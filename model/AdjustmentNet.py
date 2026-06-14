import torch.nn as nn
from torchvision import transforms
from PIL import Image
from utils import *

def conv3x3(in_planes, out_planes, stride=1):
	return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
	return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class ResBlk(nn.Module):
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(ResBlk, self).__init__()
        self.bias1a = nn.Parameter(torch.zeros(1))
        self.actv1 = nn.PReLU()
        self.bias1b = nn.Parameter(torch.zeros(1))
        self.conv1 = conv3x3(inplanes, planes, stride)

        self.bias2a = nn.Parameter(torch.zeros(1))
        self.actv2 = nn.PReLU()
        self.bias2b = nn.Parameter(torch.zeros(1))
        self.conv2 = conv3x3(planes, planes)
        self.scale = nn.Parameter(torch.ones(1))

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.actv1(x + self.bias1a)
        if self.downsample is not None:
            identity = self.downsample(out)

        out = self.conv1(out + self.bias1b)
        out = self.actv2(out + self.bias2a)
        out = self.conv2(out + self.bias2b)
        out = out * self.scale
        out += identity
        return out


class ColorAdjustmentEncoder(nn.Module):
    def __init__(self, out_dims=3):
        super(ColorAdjustmentEncoder, self).__init__()
        self.planes = [64, 128, 256, 512]
        self.inplanes = self.planes[0]

        self.conv1 = nn.Conv2d(3, self.planes[0], kernel_size=7, stride=2, padding=3, bias=False)
        self.actv = nn.PReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = nn.Sequential(*self._make_layer(ResBlk, self.planes[0], 2))
        self.layer2 = nn.Sequential(*self._make_layer(ResBlk, self.planes[1], 2, stride=2))
        self.layer3 = nn.Sequential(*self._make_layer(ResBlk, self.planes[2], 2, stride=2))
        self.layer4 = nn.Sequential(*self._make_layer(ResBlk, self.planes[3], 2, stride=2))
        self.gap = nn.AdaptiveAvgPool2d(1)

        # Predict the control point parameters for the final 9 sets of curves
        self.fc_curve = nn.Linear(self.planes[3], out_dims)
        nn.init.normal_(self.fc_curve.weight, mean=0.0, std=1e-3)
        nn.init.constant_(self.fc_curve.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = conv1x1(self.inplanes, planes, stride)
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return layers

    def forward(self, x):
        x = self.maxpool(self.actv(self.conv1(x)))
        f1 = self.layer1(x)
        f2 = self.layer2(f1)
        f3 = self.layer3(f2)
        f4 = self.layer4(f3)
        g = self.gap(f4).flatten(1)
        curve_params = self.fc_curve(g)
        return curve_params, f4


class SpatialCorrectionHead(nn.Module):
    def __init__(self, in_channels=512, hidden_channels=128, out_channels=3):
        super(SpatialCorrectionHead, self).__init__()
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1, bias=True),
            nn.PReLU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1, bias=True),
            nn.PReLU(),
            nn.Conv2d(hidden_channels, out_channels, kernel_size=1, bias=True)
        )

        nn.init.normal_(self.head[-1].weight, mean=0.0, std=1e-3)
        nn.init.constant_(self.head[-1].bias, 0.0)

    def forward(self, feat, out_h, out_w):
        local_map = self.head(feat)
        local_map = F.interpolate(local_map, size=(out_h, out_w), mode='bilinear', align_corners=False)
        return local_map


class ColorAdjustmentNet(nn.Module):
    def __init__(self, num_knots=16, cd=256, local_strength=0.1, global_strength=0.1):
        super(ColorAdjustmentNet, self).__init__()
        self.cd = cd                  # Lookup table length.
        self.knots = num_knots        # Number of control points.
        # Nine mapping relationships: (R,G,B) -> (R,G,B)
        self.cl = [self.knots] * 9
        self.encoder = ColorAdjustmentEncoder(sum(self.cl))
        self.spatial_head = SpatialCorrectionHead(in_channels=512, hidden_channels=128, out_channels=3)
        self.local_strength = local_strength
        self.global_strength = global_strength

    def interp_curves(self, params, batch_size):
        # Interpolate the predicted control points into smooth curves of length 256
        params = params.view(batch_size * 9, 1, 1, self.knots)
        curves = F.interpolate(params, (1, self.cd), mode='bilinear', align_corners=False)
        return curves.view(batch_size, 9, self.cd)

    def apply_curve(self, x, curve):
        # x: [B, 1, H, W], curve: [B, cd]
        B, _, H, W = x.size()
        x_ind = (torch.clamp(x, 0, 1) * (self.cd - 1)).long().flatten(1).detach()
        out = torch.gather(curve, 1, x_ind)
        return out.reshape(B, 1, H, W)

    def global_color_correction(self, x, curves):
        # Each output channel is jointly determined by the input RGB channels
        out_list = []
        for i in range(3):
            out_i = self.apply_curve(x[:, [0], ...], curves[:, i * 3 + 0]) + \
                    self.apply_curve(x[:, [1], ...], curves[:, i * 3 + 1]) + \
                    self.apply_curve(x[:, [2], ...], curves[:, i * 3 + 2])
            out_list.append(out_i)
        out = torch.cat(out_list, dim=1)
        out = x + torch.tanh(out) * self.global_strength
        return out

    def forward(self, x, return_curves=False):
        B, _, H, W = x.size()

        # Extract global color and deep spatial features from the low-resolution input encoder.
        x_low = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        curve_params, feat = self.encoder(x_low)

        # Global curve interpolation
        curves = self.interp_curves(curve_params, B)
        x_global = self.global_color_correction(x, curves)
        # Local color correction
        local_map = self.spatial_head(feat, H, W)
        local_gain = 1.0 + self.local_strength * torch.tanh(local_map)

        x_out = x_global * local_gain
        x_out = torch.clamp(x_out, 0.0, 1.0)
        # Used for curve plotting
        if return_curves:
            effective_curves = torch.tanh(curves) * self.global_strength
            return x_out, {
                "raw_curves": curves,
                "effective_curves": effective_curves,
                "curve_params": curve_params,
                "local_map": local_map,
                "local_gain": local_gain
            }
        return x_out


if __name__ == '__main__':
    img_path = "../test_imgs/Correction/LLVIP/test/1.jpg"
    transform = transforms.Compose([transforms.ToTensor(),])

    img = Image.open(img_path).convert("RGB")
    input = transform(img).unsqueeze(0)
    model_enhance = ColorAdjustmentNet()
    path = '../result_imgs/correct/train/exp42/color_correction_model.pt'
    model_enhance.load_state_dict(torch.load(path))
    enhanced_img, info = model_enhance(input, return_curves=True)
    plot_rgb_separate_curves(info["effective_curves"], sample_idx=0, save_dir="curve_results")