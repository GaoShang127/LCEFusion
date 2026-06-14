import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from utils import *


# -----------1.Basic convolution layer------------
# 3×3Conv + BN + LeakyReLU
class Conv3BnLRelu2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding,  bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x):
        out = self.act(self.bn(self.conv(x)))
        return out


# 1×1Conv + LeakyReLU
class Conv1LRelu2d(nn.Module):
    def __init__(self, in_channels, out_channels=None, kernel_size=1, padding=0):
        super().__init__()
        out_channels = out_channels or in_channels
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=True)
        self.act = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x):
        out = self.act(self.conv(x))
        return out


# 1×1Conv + BN + LeakyReLU
class Conv1BnLRelu2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x):
        out = self.act(self.bn(self.conv(x)))
        return out


# Used to implement transposed convolution upsampling.
class TConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.tconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x):
        return self.tconv(x)


# -----------2.Input-output convolution layer------------
# Input convolution block
class InConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_block = nn.Sequential(
            Conv3BnLRelu2d(in_channels, out_channels),
            Conv3BnLRelu2d(out_channels, out_channels),
            Conv3BnLRelu2d(out_channels, out_channels)
        )

    def forward(self, x):
        out = self.conv_block(x)
        return out


# Output convolution block
class OutConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=True)
        self.out_act = nn.Sigmoid()

    def forward(self, x):
        x = self.conv(x)
        x = self.out_act(x)
        return x


# -----------3. Network encoding layer------------
# Token channel gating in CA_Swin.
class QKVChannelGate(nn.Module):
    def __init__(self, dim, reduction=4):
        super().__init__()
        hidden_dim = max(dim // reduction, 1)
        self.shared_fc = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU()
        )
        self.q_gate = nn.Linear(hidden_dim, dim)
        self.k_gate = nn.Linear(hidden_dim, dim)
        self.v_gate = nn.Linear(hidden_dim, dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        g = x.mean(dim=1)  # (B*num_windows, C)
        g = self.shared_fc(g)  # (B*num_windows, hidden_dim)
        q_gate = self.sigmoid(self.q_gate(g)).unsqueeze(1).unsqueeze(2)  # (B*num_windows, 1, 1, C)
        k_gate = self.sigmoid(self.k_gate(g)).unsqueeze(1).unsqueeze(2)
        v_gate = self.sigmoid(self.v_gate(g)).unsqueeze(1).unsqueeze(2)
        return q_gate, k_gate, v_gate


# Window-based multi-head self-attention with channel gating
class CA_MSA(nn.Module):
    def __init__(self, dim, num_heads=4, window_size=8, qkv_bias=True, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads."
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.qkv_channel_gate = QKVChannelGate(dim)

        relative_bias_size = (2 * window_size - 1) * (2 * window_size - 1)
        self.relative_position_bias_table = nn.Parameter(torch.zeros(relative_bias_size, num_heads))

        coords_h = torch.arange(window_size)
        coords_w = torch.arange(window_size)
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing='ij'))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += window_size - 1
        relative_coords[:, :, 1] += window_size - 1
        relative_coords[:, :, 0] *= 2 * window_size - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

    def forward(self, x, H, W):
        B, N, C = x.shape
        assert N == H * W, "The number of input tokens is inconsistent with H * W"
        x = x.view(B, H, W, C)

        pad_h = (self.window_size - H % self.window_size) % self.window_size
        pad_w = (self.window_size - W % self.window_size) % self.window_size
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))

        Hp, Wp = x.shape[1], x.shape[2]
        x_windows = window_partition(x, self.window_size)
        qkv = self.qkv(x_windows)
        qkv = qkv.reshape(-1, self.window_size * self.window_size, 3, C)
        q_gate, k_gate, v_gate = self.qkv_channel_gate(x_windows)
        qkv = qkv * torch.cat([q_gate, k_gate, v_gate], dim=2)
        qkv = qkv.reshape(-1, self.window_size * self.window_size, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size * self.window_size,
            self.window_size * self.window_size,
            self.num_heads
        )
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = attn @ v
        out = out.transpose(1, 2).reshape(x_windows.shape[0], x_windows.shape[1], C)
        out = self.proj(out)
        out = self.proj_drop(out)

        out = window_reverse(out, self.window_size, Hp, Wp, B)
        if pad_h > 0 or pad_w > 0:
            out = out[:, :H, :W, :].contiguous()

        out = out.view(B, H * W, C)
        return out


# Feed-forward network in CA_Swin
class SwinMlp(nn.Module):
    def __init__(self, dim, hidden_dim=None, out_dim=None, drop=0.0):
        super().__init__()
        hidden_dim = hidden_dim or dim * 4
        out_dim = out_dim or dim
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


# Swin Transformer with channel gating.
class CA_Swin(nn.Module):
    def __init__(self, dim, num_heads=4, window_size=8, mlp_ratio=4.0, drop=0.0, attn_drop=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = CA_MSA(dim=dim, num_heads=num_heads, window_size=window_size, qkv_bias=True, attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = SwinMlp(dim=dim, hidden_dim=int(dim * mlp_ratio), out_dim=dim, drop=drop)

    def forward(self, x):
        B, C, H, W = x.shape
        x_tokens = x.flatten(2).transpose(1, 2)
        x_tokens = x_tokens + self.attn(self.norm1(x_tokens), H, W)
        x_tokens = x_tokens + self.mlp(self.norm2(x_tokens))
        out = x_tokens.transpose(1, 2).reshape(B, C, H, W)
        return out


# UNet Encoder-CA_SwinTransformer
class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=True, window_size=8, num_heads=4):
        super().__init__()
        self.proj = Conv1LRelu2d(in_channels, out_channels)
        self.attn = CA_Swin(dim=out_channels, num_heads=num_heads, window_size=window_size)
        self.downsample = downsample
        self.pool = nn.AvgPool2d(2) if downsample else nn.Identity()

    def forward(self, x):
        feat = self.proj(x)
        feat = self.attn(feat)
        down = self.pool(feat)
        return feat, down


# -----------4. Network decoding layer------------
class DecoderConvs(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.dconv = Conv3BnLRelu2d(channels, channels)
        self.conv1 = Conv1BnLRelu2d(channels, channels)
        self.conv2 = Conv3BnLRelu2d(channels, channels)

    def forward(self, x):
        residual = x
        x = self.dconv(x)
        x = self.conv1(x)
        x = self.conv2(x)
        out = x + residual
        return out


# UNet Decoder-Convs
class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, upsample=True):
        super().__init__()
        self.deconvs = DecoderConvs(in_channels)
        self.conv = Conv1LRelu2d(in_channels, out_channels)
        self.upsample = TConv(out_channels, out_channels) if upsample else nn.Identity()

    def forward(self, x):
        x = self.deconvs(x)
        x = self.conv(x)
        x = self.upsample(x)
        return x


# -----------5. Modality feature/encoder-decoder interaction layer------------
# Stochastic depth
class DropPath(nn.Module):
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor = random_tensor.floor()
        return x.div(keep_prob) * random_tensor


# Spatial dimension unembedding
class SpatialDePatch(nn.Module):
    def __init__(self, channel=16, embed_dim=128, patch_size=16):
        super().__init__()
        self.patch_size = patch_size
        self.projection = nn.Linear(embed_dim, patch_size * patch_size * channel)

    def forward(self, x, ori_shape):
        b, c, h, w = ori_shape
        h_ = h // self.patch_size
        w_ = w // self.patch_size
        x = self.projection(x)
        x = rearrange(x,'b (h w) (p1 p2 c) -> b c (h p1) (w p2)', h=h_, w=w_, p1=self.patch_size, p2=self.patch_size )
        return x


# Channel dimension unembedding
class ChannelDePatch(nn.Module):
    def __init__(self, embed_dim=128, patch_size=16):
        super().__init__()
        self.patch_size = patch_size
        self.projection = nn.Linear(embed_dim, patch_size * patch_size)

    def forward(self, x, ori_shape):
        b, c, h, w = ori_shape
        h_ = h // self.patch_size
        w_ = w // self.patch_size
        x = self.projection(x)
        x = rearrange(x,'(b h w) c (p1 p2) -> b c (h p1) (w p2)', b=b, h=h_, w=w_, p1=self.patch_size, p2=self.patch_size)
        return x


# Feed-forward MLP
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


# Transformer Cross Attention
class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, y):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, 1, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        kv = self.kv(y).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q = q[0]
        k = kv[0]
        v = kv[1]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out


# Cross-modal interaction Transformer block
class CrossTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, qkv_bias=False, qk_scale=None, drop=0.0, attn_drop=0.0, drop_path=0.0,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.norm1 = norm_layer(dim)
        self.attn = CrossAttention(dim=dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop,
                                   proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, y):
        x = x + self.drop_path(self.attn(self.norm1(x), self.norm1(y)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


# Feature interaction encoder
class CrossTransEncoder(nn.Module):
    def __init__(self, embed_dim=256, depth=2, num_heads=4, mlp_ratio=2.0, qkv_bias=False, qk_scale=None, drop_rate=0.0,
                 attn_drop_rate=0.0, drop_path_rate=0.0, norm_layer=nn.LayerNorm):
        super().__init__()
        dpr = torch.linspace(0, drop_path_rate, depth).tolist()
        self.pos_drop = nn.Dropout(p=drop_rate)
        self.blocks = nn.ModuleList([
            CrossTransformerBlock(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                  drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)
        ])
        self.norm = norm_layer(embed_dim)

    def forward(self, x, y):
        x = self.pos_drop(x)
        y = self.pos_drop(y)
        for blk in self.blocks:
            x = blk(x, y)
        x = self.norm(x)
        return x


# Spatial dimension interaction
class CrossSpatial(nn.Module):
    def __init__(self, embed_dim=256, depth=2, channel=16, num_heads=4, mlp_ratio=2.0, patch_size=16, qkv_bias=False,
                 qk_scale=None, drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.0, norm_layer=nn.LayerNorm):
        super().__init__()
        self.patch_size = patch_size
        self.embedding = nn.Linear(patch_size * patch_size * channel, embed_dim)
        self.encoder = CrossTransEncoder(
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            norm_layer=norm_layer
        )
        self.depatch = SpatialDePatch(channel=channel, embed_dim=embed_dim, patch_size=patch_size)

    def forward(self, x, y):
        ori_shape = x.shape
        x = rearrange(x,'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=self.patch_size, p2=self.patch_size)
        y = rearrange(y,'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=self.patch_size, p2=self.patch_size)
        x = self.embedding(x)
        y = self.embedding(y)
        x = self.encoder(x, y)
        out = self.depatch(x, ori_shape)
        return out


# Channel dimension interaction
class CrossChannel(nn.Module):
    def __init__(self, embed_dim=256, depth=2, channel=16, num_heads=4, mlp_ratio=2.0, patch_size=16, qkv_bias=False,
                 qk_scale=None, drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.0, norm_layer=nn.LayerNorm):
        super().__init__()
        self.patch_size = patch_size
        self.embedding = nn.Linear(patch_size * patch_size, embed_dim)
        self.encoder = CrossTransEncoder(
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            norm_layer=norm_layer
        )
        self.depatch = ChannelDePatch(embed_dim=embed_dim, patch_size=patch_size)

    def forward(self, x, y):
        ori_shape = x.shape
        x = rearrange(x,'b c (h p1) (w p2) -> (b h w) c (p1 p2)', p1=self.patch_size, p2=self.patch_size)
        y = rearrange(y,'b c (h p1) (w p2) -> (b h w) c (p1 p2)', p1=self.patch_size, p2=self.patch_size)
        x = self.embedding(x)
        y = self.embedding(y)
        x = self.encoder(x, y)
        out = self.depatch(x, ori_shape)
        return out


# Complete CFIM module
class CFIM(nn.Module):
    def __init__(self, spatial_embed_dim=64, spatial_patch_size=8, channel_embed_dim=128, channel_patch_size=16,
                 channel=32, depth=2, num_heads=4, mlp_ratio=2.0, qkv_bias=False, qk_scale=None, drop_rate=0.0,
                 attn_drop_rate=0.0, drop_path_rate=0.0, norm_layer=nn.LayerNorm):
        super().__init__()
        self.cross_spatial = CrossSpatial(embed_dim=spatial_embed_dim, depth=depth, channel=channel, num_heads=num_heads,
                                          mlp_ratio=mlp_ratio, patch_size=spatial_patch_size, qkv_bias=qkv_bias,
                                          qk_scale=qk_scale, drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
                                          drop_path_rate=drop_path_rate, norm_layer=norm_layer)
        self.cross_channel = CrossChannel(embed_dim=channel_embed_dim, depth=depth, channel=channel, num_heads=num_heads,
                                          mlp_ratio=mlp_ratio, patch_size=channel_patch_size, qkv_bias=qkv_bias,
                                          qk_scale=qk_scale, drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
                                          drop_path_rate=drop_path_rate, norm_layer=norm_layer)

    def forward(self, ir_feature, vi_feature):
        # Spatial dimension interaction
        ir_spatial = self.cross_spatial(ir_feature, vi_feature)
        vi_spatial = self.cross_spatial(vi_feature, ir_feature)
        # # Channel dimension interaction
        ir_channel = self.cross_channel(ir_spatial, vi_spatial)
        vi_channel = self.cross_channel(vi_spatial, ir_spatial)
        # Residual connection
        ir_out = ir_feature + ir_channel
        vi_out = vi_feature + vi_channel
        out = torch.cat([ir_out, vi_out], dim=1)
        return out


class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()
        hidden_channels = max(channels // reduction, 1)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden_channels, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, channels, kernel_size=1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Global average pooling (GAP) and global max pooling (GMP)
        gap = F.adaptive_avg_pool2d(x, 1)
        gmp = F.adaptive_max_pool2d(x, 1)
        out = self.mlp(gap) + self.mlp(gmp)
        return self.sigmoid(out)


class GateWeight(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()
        self.attn_low = ChannelAttention(channels, reduction)
        self.attn_high = ChannelAttention(channels, reduction)

    def forward(self, f_low, f_high):
        w_low = self.attn_low(f_low)
        w_high = self.attn_high(f_high)
        return w_low, w_high


class DFTDecomposition(nn.Module):
    def __init__(self, alpha=0.25):
        super().__init__()
        self.alpha = alpha

    def get_dct_matrix(self, N, device):
        n = torch.arange(N, dtype=torch.float32, device=device)
        k = n.unsqueeze(1)
        dct = torch.cos(math.pi / N * (n + 0.5) * k)
        dct[0] *= 1.0 / math.sqrt(2.0)
        dct *= math.sqrt(2.0 / N)
        return dct

    def dct_2d(self, x):
        B, C, H, W = x.shape
        D_H = self.get_dct_matrix(H, x.device)
        D_W = self.get_dct_matrix(W, x.device)
        # Y = D_H @ X @ D_W^T
        out = torch.einsum('mh, bchw -> bcmw', D_H, x)
        out = torch.einsum('bcmw, nw -> bcmn', out, D_W)
        return out

    def idct_2d(self, x):
        B, C, H, W = x.shape
        D_H = self.get_dct_matrix(H, x.device)
        D_W = self.get_dct_matrix(W, x.device)
        # X = D_H^T @ Y @ D_W
        out = torch.einsum('hm, bcmw -> bchw', D_H, x)
        out = torch.einsum('bchw, wn -> bchn', out, D_W)
        return out

    def forward(self, x):
        B, C, H, W = x.shape
        F = self.dct_2d(x)
        v = torch.arange(H, dtype=torch.float32, device=x.device).unsqueeze(1) / H
        u = torch.arange(W, dtype=torch.float32, device=x.device).unsqueeze(0) / W
        D2 = v ** 2 + u ** 2
        # H(u,v) = exp(-D^2 / (2 * D0^2))
        D0 = self.alpha
        mask_low = torch.exp(-D2 / (2 * (D0 ** 2)))
        mask_low = mask_low.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
        mask_high = 1.0 - mask_low
        # Frequency Disentanglement
        F_low = self.idct_2d(F * mask_low)
        F_high = self.idct_2d(F * mask_high)
        return F_low, F_high


class DFGF(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.dft = DFTDecomposition()
        self.gw = GateWeight(channels)
        self.align_low = nn.Sequential(
            nn.Conv2d(channels*2, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
        self.align_high = nn.Sequential(
            nn.Conv2d(channels*2, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(channels*2, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, f_en, f_de):
        f = torch.cat([f_en, f_de], dim=1)
        f_low, f_high = self.dft(f_en)
        w_low, w_high = self.gw(f_low, f_high)
        feat_low = self.align_low(f)
        feat_high = self.align_high(f)
        feat_low_gated = w_low * feat_low
        feat_high_gated = w_high * feat_high
        feat_concat = torch.cat([feat_low_gated, feat_high_gated], dim=1)
        out = self.fusion_conv(feat_concat)
        return out


# -----------6.Complete fusion network------------
class FusionNet(nn.Module):
    def __init__(self, vi_channels=1, ir_channels=1, out_channels=1, stem_channels=32):
        super().__init__()
        # Input conv
        self.vi_inconv = InConvBlock(vi_channels, stem_channels)
        self.ir_inconv = InConvBlock(ir_channels, stem_channels)

        # Visible-light branch encoder
        self.vi_encoder1 = EncoderBlock(stem_channels, 64, downsample=True, window_size=8, num_heads=4)
        self.vi_encoder2 = EncoderBlock(64, 128, downsample=True, window_size=8, num_heads=4)
        self.vi_encoder3 = EncoderBlock(128, 128, downsample=True, window_size=4, num_heads=4)
        # Infrared branch encoder
        self.ir_encoder1 = EncoderBlock(stem_channels, 64, downsample=True, window_size=8, num_heads=4)
        self.ir_encoder2 = EncoderBlock(64, 128, downsample=True, window_size=8, num_heads=4)
        self.ir_encoder3 = EncoderBlock(128, 128, downsample=True, window_size=4, num_heads=4)
        # bottleneck CA_Swin
        self.bottleneck = CA_Swin(dim=256, num_heads=4, window_size=4)

        # Encoder feature interaction
        self.cfim1 = CFIM(spatial_embed_dim=64, spatial_patch_size=8, channel_embed_dim=128, channel_patch_size=16, channel=64)
        self.cfim2 = CFIM(spatial_embed_dim=128, spatial_patch_size=4, channel_embed_dim=256, channel_patch_size=8, channel=128)
        self.cfim3 = CFIM(spatial_embed_dim=128, spatial_patch_size=2, channel_embed_dim=256, channel_patch_size=4, channel=128)
        # Encoder-decoder feature interaction
        self.dfgf3 = DFGF(256)
        self.dfgf2 = DFGF(256)
        self.dfgf1 = DFGF(128)
        self.dfgf0 = DFGF(64)

        # Decoder
        self.decoder3 = DecoderBlock(256, 256, upsample=True)
        self.decoder2 = DecoderBlock(256, 128, upsample=True)
        self.decoder1 = DecoderBlock(128, 64, upsample=False)
        self.decoder0 = DecoderBlock(64, 32, upsample=False)

        # Output conv
        self.outconv = OutConvBlock(32, out_channels)

    def forward(self, vi, ir):
        # Input conv
        vi0 = self.vi_inconv(vi)
        ir0 = self.ir_inconv(ir)
        skip0 = torch.cat([vi0, ir0], dim=1)

        # Encoder-Feature extraction
        vi1, vi1_down = self.vi_encoder1(vi0)
        vi2, vi2_down = self.vi_encoder2(vi1_down)
        vi3, vi3_down = self.vi_encoder3(vi2_down)
        ir1, ir1_down = self.ir_encoder1(ir0)
        ir2, ir2_down = self.ir_encoder2(ir1_down)
        ir3, ir3_down = self.ir_encoder3(ir2_down)
        # Encoder-Feature interaction
        skip1 = self.cfim1(ir1, vi1)
        skip2 = self.cfim2(ir2, vi2)
        skip3 = self.cfim3(ir3, vi3)

        # Bottleneck
        x = self.bottleneck(skip3)

        # Decoder
        x = self.dfgf3(skip3, x)
        x = self.decoder3(x)
        x = self.dfgf2(skip2, x)
        x = self.decoder2(x)
        x = self.dfgf1(skip1, x)
        x = self.decoder1(x)
        x = self.dfgf0(skip0, x)
        x = self.decoder0(x)

        # Output conv
        out = self.outconv(x)
        return out


if __name__ == '__main__':
    vi = torch.randn((1, 1, 224, 224))
    ir = torch.randn((1, 1, 224, 224))
    model_fusion = FusionNet()

    fused = model_fusion(vi, ir)
    print(fused.shape)