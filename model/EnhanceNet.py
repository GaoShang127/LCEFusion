# coding:utf-8
import torch.nn as nn
from utils import *
from model.DenoiseNet import NAFNet


class DenoiseNet(nn.Module):
    def __init__(self, device='cuda'):
        super().__init__()
        self.device = device
        # Initialize the network for RGB images with channels=3.
        self.model = NAFNet(img_channel=3,  width=64,  middle_blk_num=12,  enc_blk_nums=[2, 2, 4, 8], dec_blk_nums=[2, 2, 2, 2])
        # Load the weight file
        weight_path = '/home/kemove/Documents/Pycharm_Document/LCEFusion/weight/denoise_model.pth'
        self.model = load_denoisenet_weights(self.model, weight_path, device)
        self.model.eval()
        # Freeze the weights so they are not updated during training
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, x):
        is_single_channel = (x.size(1) == 1)
        if is_single_channel:
            x_input = x.repeat(1, 3, 1, 1)
        else:
            x_input = x
        with torch.no_grad():
            denoised_img = self.model(x_input)
            denoised_img = torch.clamp(denoised_img, 0, 1)
        return denoised_img


# Coordinate attention module
class CoordinateAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=8):
        super().__init__()
        self.in_channels = in_channels
        self.reduced_channels = max(in_channels // reduction_ratio, 8)  # Avoid too few channels.
        # Global average pooling in the horizontal and vertical directions.
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        # Shared convolution layer.
        self.conv1 = nn.Conv2d(in_channels, self.reduced_channels, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(self.reduced_channels)
        self.act = nn.ReLU(inplace=True)
        # Direction-specific convolution layers.
        self.conv_h = nn.Conv2d(self.reduced_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(self.reduced_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        identity = x
        b, c, h, w = x.size()
        # Embed coordinate information.
        x_h = self.pool_h(x)  # [B, C, H, 1]
        x_w = self.pool_w(x).permute(0, 1, 3, 2)  # [B, C, 1, W] -> [B, C, W, 1]
        # Concatenate features and apply the shared convolution.
        y = torch.cat([x_h, x_w], dim=2)  # [B, C, H+W, 1]
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)
        # Split features along the horizontal and vertical directions.
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)  # Restore dimensions to [B, C, 1, W].
        # Generate attention weights.
        a_h = self.sigmoid(self.conv_h(x_h))
        a_w = self.sigmoid(self.conv_w(x_w))
        # Apply attention weighting.
        out = identity * a_h * a_w
        return out


# U-Net Encoder
class EnhanceEncoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(EnhanceEncoder, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x_pool = self.maxpool(x)
        return x, x_pool


# U-Net Decoder
class EnhanceDecoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(EnhanceDecoder, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2, padding=0)
        self.channel = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.conv1 = nn.Conv2d(2*out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, x_encoder):
        x = self.up(x)
        if x.shape[2:] != x_encoder.shape[2:]:
            x = F.interpolate(x, size=x_encoder.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, x_encoder], dim=1)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        return x


# U-Net BottleNeck
class BottleNeck(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(BottleNeck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        return x


class LightBootNet(nn.Module):
    def __init__(self, in_channels=1, iter_num=8, base_channels=32):
        super(LightBootNet, self).__init__()
        self.iter_num = iter_num
        out_channels = in_channels * self.iter_num
        self.encoder = nn.ModuleList([
            EnhanceEncoder(in_channels, base_channels),
            EnhanceEncoder(base_channels, base_channels * 2),
            EnhanceEncoder(base_channels * 2, base_channels * 4),
            EnhanceEncoder(base_channels * 4, base_channels * 8),
        ])
        self.bottleneck = BottleNeck(base_channels * 8, base_channels * 16)
        self.decoder = nn.ModuleList([
            EnhanceDecoder(base_channels * 16, base_channels * 8),
            EnhanceDecoder(base_channels * 8, base_channels * 4),
            EnhanceDecoder(base_channels * 4, base_channels * 2),
            EnhanceDecoder(base_channels * 2, base_channels),
        ])
        self.coord_attn = CoordinateAttention(base_channels)
        self.out_conv = nn.Conv2d(base_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # Encoder
        enc1, x = self.encoder[0](x)
        enc2, x = self.encoder[1](x)
        enc3, x = self.encoder[2](x)
        enc4, x = self.encoder[3](x)
        # Bottleneck
        bottleneck = self.bottleneck(x)
        # Decoder
        dec4 = self.decoder[0](bottleneck, enc4)
        dec3 = self.decoder[1](dec4, enc3)
        dec2 = self.decoder[2](dec3, enc2)
        dec1 = self.decoder[3](dec2, enc1)
        # coord_attn
        attn_feat = self.coord_attn(dec1)
        # Output curve
        a_map = self.tanh(self.out_conv(attn_feat))
        return a_map


class EnhancementNet(nn.Module):
    def __init__(self, in_channels=1, iter_num=8, base_channels=32, device="cuda"):
        super().__init__()
        self.in_channels = in_channels
        self.iter_num = iter_num
        self.tanh = nn.Tanh()
        # Illumination adjustment curve estimation network.
        self.illumination_estimator = LightBootNet(in_channels=in_channels, iter_num=iter_num, base_channels=base_channels)
        # Denoising module
        self.denoiser = DenoiseNet(device=device)

    def enhance_with_curve(self, x, a_map):
        enhanced = x
        a_maps = torch.split(a_map, self.in_channels, dim=1)
        x_org = x.clone()
        x_cur = x_org
        for i in range(self.iter_num//2):
            x_cur = x_cur + a_maps[i] * ((x_cur ** 2 - x_cur) / torch.exp(x_cur))
        enhance_image_l = x_cur.clone()
        for i in range(self.iter_num//2, self.iter_num):
            x_cur = x_cur + a_maps[i] * ((x_cur ** 2 - x_cur) / torch.exp(x_cur))
        enhance_image = x_cur
        a = torch.cat(a_maps, dim=1)
        return enhance_image_l, enhance_image, a

    def forward(self, x):
        # Predict illumination curve parameters
        a_map = self.illumination_estimator(x)
        if not self.training:
            intensity = torch.mean(x, dim=1, keepdim=True)
            dark_suppress_mask = torch.sigmoid(30 * (intensity - 0.01))
            bright_suppress_mask = 1.0 - torch.sigmoid(20 * (intensity - 0.9))
            attention_mask = dark_suppress_mask * bright_suppress_mask
            a_map = a_map * attention_mask
        # Curve enhance
        enhanced_image_l, enhanced_image, a = self.enhance_with_curve(x, a_map)
        # enhanced_image = self.illumination_estimator.usm(enhanced_image)
        return enhanced_image_l, enhanced_image, a


if __name__ == '__main__':
    input = torch.randn((1, 1, 640, 480))
    model_enhance = EnhancementNet()
    _, enhanced_img, a_map = model_enhance(input)
    print(enhanced_img.shape)