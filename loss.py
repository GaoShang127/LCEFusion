#!/usr/bin/python
# -*- encoding: utf-8 -*-
import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision.models as models
from torch.autograd import Variable
from math import exp
from utils import *


# ----------------1.Build the EnhanceLoss enhancement loss function----------------
class SpatialConsistencyLoss(nn.Module):
    def __init__(self):
        super(SpatialConsistencyLoss, self).__init__()
        # Define eight convolution kernels left、right、up、down、upleft、upright、downleft、downright
        # (H, W)->(1, H, W)->(1, 1, H, W)   (1, 1, 3, 3)
        kernel_left = torch.FloatTensor([[0, 0, 0], [-1, 1, 0], [0, 0, 0]]).cuda().unsqueeze(0).unsqueeze(0)
        kernel_right = torch.FloatTensor([[0, 0, 0], [0, 1, -1], [0, 0, 0]]).cuda().unsqueeze(0).unsqueeze(0)
        kernel_up = torch.FloatTensor([[0, -1, 0], [0, 1, 0], [0, 0, 0]]).cuda().unsqueeze(0).unsqueeze(0)
        kernel_down = torch.FloatTensor([[0, 0, 0], [0, 1, 0], [0, -1, 0]]).cuda().unsqueeze(0).unsqueeze(0)
        kernel_upleft = torch.FloatTensor([[-1, 0, 0], [0, 1, 0], [0, 0, 0]]).cuda().unsqueeze(0).unsqueeze(0)
        kernel_upright = torch.FloatTensor([[0, 0, -1], [0, 1, 0], [0, 0, 0]]).cuda().unsqueeze(0).unsqueeze(0)
        kernel_downleft = torch.FloatTensor([[0, 0, 0], [0, 1, 0], [-1, 0, 0]]).cuda().unsqueeze(0).unsqueeze(0)
        kernel_downright = torch.FloatTensor([[0, 0, 0], [0, 1, -1], [0, 0, -1]]).cuda().unsqueeze(0).unsqueeze(0)
        # Define the above convolution kernels as fixed kernels without gradient computation
        self.weight_left = nn.Parameter(data=kernel_left, requires_grad=False)
        self.weight_right = nn.Parameter(data=kernel_right, requires_grad=False)
        self.weight_up = nn.Parameter(data=kernel_up, requires_grad=False)
        self.weight_down = nn.Parameter(data=kernel_down, requires_grad=False)
        self.weight_upleft = nn.Parameter(data=kernel_upleft, requires_grad=False)
        self.weight_upright = nn.Parameter(data=kernel_upright, requires_grad=False)
        self.weight_downleft = nn.Parameter(data=kernel_downleft, requires_grad=False)
        self.weight_downright = nn.Parameter(data=kernel_downright, requires_grad=False)
        # Define an average pooling layer with a 4x4 window size.
        self.pool = nn.AvgPool2d(4)

    def forward(self, img_low, img_enhanced):
        # (B, 3, H, W)->(B, 1, H, W)
        img_low_y = torch.mean(img_low, 1, keepdim=True)
        img_enhanced_y = torch.mean(img_enhanced, 1, keepdim=True)
        # (B, 1, H, W)->(B, 1, H/4, W/4)
        img_low_pool = self.pool(img_low_y)
        img_enhanced_pool = self.pool(img_enhanced_y)
        # Compute local gradient differences of the source image in eight directions
        gard_low_left = F.conv2d(img_low_pool, self.weight_left, padding=1)
        gard_low_right = F.conv2d(img_low_pool, self.weight_right, padding=1)
        gard_low_up = F.conv2d(img_low_pool, self.weight_up, padding=1)
        gard_low_down = F.conv2d(img_low_pool, self.weight_down, padding=1)
        gard_low_upleft = F.conv2d(img_low_pool, self.weight_upleft, padding=1)
        gard_low_upright = F.conv2d(img_low_pool, self.weight_upright, padding=1)
        gard_low_downleft = F.conv2d(img_low_pool, self.weight_downleft, padding=1)
        gard_low_downright = F.conv2d(img_low_pool, self.weight_downright, padding=1)
        # Compute local gradient differences of the enhanced image in eight directions
        gard_enhance_left = F.conv2d(img_enhanced_pool, self.weight_left, padding=1)
        gard_enhance_right = F.conv2d(img_enhanced_pool, self.weight_right, padding=1)
        gard_enhance_up = F.conv2d(img_enhanced_pool, self.weight_up, padding=1)
        gard_enhance_down = F.conv2d(img_enhanced_pool, self.weight_down, padding=1)
        gard_enhance_upleft = F.conv2d(img_enhanced_pool, self.weight_upleft, padding=1)
        gard_enhance_upright = F.conv2d(img_enhanced_pool, self.weight_upright, padding=1)
        gard_enhance_downleft = F.conv2d(img_enhanced_pool, self.weight_downleft, padding=1)
        gard_enhance_downright = F.conv2d(img_enhanced_pool, self.weight_downright, padding=1)
        # Compute the squared gradient difference between the original and enhanced images in each direction.
        diff_left = torch.pow(gard_low_left - gard_enhance_left, 2)
        diff_right = torch.pow(gard_low_right - gard_enhance_right, 2)
        diff_up = torch.pow(gard_low_up - gard_enhance_up, 2)
        diff_down = torch.pow(gard_low_down - gard_enhance_down, 2)
        diff_upleft = torch.pow(gard_low_upleft - gard_enhance_upleft, 2)
        diff_upright = torch.pow(gard_low_upright - gard_enhance_upright, 2)
        diff_downleft = torch.pow(gard_low_downleft - gard_enhance_downleft, 2)
        diff_downright = torch.pow(gard_low_downright - gard_enhance_downright, 2)
        # Sum the squared gradient differences to compute the loss
        loss = torch.mean(diff_left + diff_right + diff_up + diff_down + diff_upleft + diff_upright + diff_downleft + diff_downright)
        return loss


class TotalVariationLoss(nn.Module):
    def __init__(self, loss_weight=2):
        super(TotalVariationLoss, self).__init__()
        self.loss_weight = loss_weight

    def forward(self, img_illu):
        batch_size = img_illu.size(0)
        img_illu_h = img_illu.size(2)
        img_illu_w = img_illu.size(3)
        # Compute the number of adjacent pixel pairs in the vertical and horizontal directions
        count_h = (img_illu_h - 1) * img_illu_w
        count_w = img_illu_h * (img_illu_w - 1)
        # Compute the squared pixel differences in the horizontal and vertical directions
        # [:, :, 1:, :]From the first row to the last row
        # [:, :, :img_illu_h - 1, :] From the 0th row to the second-to-last row.
        h_tv = torch.pow((img_illu[:, :, 1:, :] - img_illu[:, :, :img_illu_h - 1, :]), 2).sum()
        w_tv = torch.pow((img_illu[:, :, :, 1:] - img_illu[:, :, :, :img_illu_w - 1]), 2).sum()
        loss = self.loss_weight * (h_tv / count_h + w_tv / count_w) / batch_size
        return loss


class ExposureLoss(nn.Module):
    def __init__(self, patch_size=16, target_exp=0.5):
        super(ExposureLoss, self).__init__()
        self.target_exp = target_exp
        self.pool = nn.AvgPool2d(kernel_size=patch_size, stride=patch_size)

    def forward(self, img_enhanced):
        img_enhanced_y = torch.mean(img_enhanced, 1, keepdim=True)
        img_enhanced_pool = self.pool(img_enhanced_y)
        loss = torch.mean(torch.pow((img_enhanced_pool - self.target_exp), 2))
        return loss


class GlobalContrastLoss(nn.Module):
    def __init__(self, gain=1.8):
        super().__init__()
        self.gain = gain

    def forward(self, enhanced, low):
        std_enh = torch.std(enhanced, dim=[2, 3])
        std_low = torch.std(low, dim=[2, 3]).detach()
        target_std = std_low * self.gain
        loss = torch.mean(F.relu(target_std - std_enh) ** 2)
        return loss


class LightBoostLoss(nn.Module):
    def __init__(self):
        super(LightBoostLoss, self).__init__()
        self.spa_loss = SpatialConsistencyLoss()
        self.tv_loss = TotalVariationLoss()
        self.exp_loss = ExposureLoss()
        self.contrast_loss = GlobalContrastLoss()

    def forward(self, img_low, img_illu, img_enhanced_final):
        # Compute each loss
        loss_spa = self.spa_loss(img_low, img_enhanced_final)
        loss_tv = 200 * self.tv_loss(img_illu)
        loss_exp = 10 * self.exp_loss(img_enhanced_final)
        loss_contrast = 10*self.contrast_loss(img_enhanced_final, img_low)
        # Compute total loss
        loss_total = loss_spa + loss_tv + loss_exp + loss_contrast
        return loss_total, loss_spa, loss_tv, loss_exp, loss_contrast


# ----------------2.Build the AdjustmentLoss Adjustment loss function----------------
class RGBL1Loss(nn.Module):
    def __init__(self):
        super(RGBL1Loss, self).__init__()
        self.l1_loss = nn.L1Loss()

    def forward(self, img_1, img_2, img_gt):
        loss_rgb = self.l1_loss(img_1, img_gt) + self.l1_loss(img_2, img_gt)
        return loss_rgb


class ColorABLoss(nn.Module):
    def __init__(self):
        super(ColorABLoss, self).__init__()
        self.l1_loss = nn.L1Loss()

    def forward(self, img_1, img_2, img_gt):
        l_1, a_1, b_1 = rgb2lab(img_1)
        l_2, a_2, b_2 = rgb2lab(img_2)
        l_gt, a_gt, b_gt = rgb2lab(img_gt)

        ab_1 = torch.cat([a_1, b_1], dim=1) / 128.0
        ab_2 = torch.cat([a_2, b_2], dim=1) / 128.0
        ab_gt = torch.cat([a_gt, b_gt], dim=1) / 128.0

        loss_ab = self.l1_loss(ab_1, ab_gt) + self.l1_loss(ab_2, ab_gt)
        return loss_ab


class SaturationLoss(nn.Module):
    def __init__(self):
        super(SaturationLoss, self).__init__()
        self.l1_loss = nn.L1Loss()

    def forward(self, img_1, img_2, img_gt):
        hsv_1 = rgb2hsv(img_1)
        hsv_2 = rgb2hsv(img_2)
        hsv_gt = rgb2hsv(img_gt)

        s_1 = hsv_1[:, 1:2, :, :]
        s_2 = hsv_2[:, 1:2, :, :]
        s_gt = hsv_gt[:, 1:2, :, :]

        loss_sat = self.l1_loss(s_1, s_gt) + self.l1_loss(s_2, s_gt)
        return loss_sat


class ReconstructionLoss(nn.Module):
    def __init__(self, w_rgb=1.0, w_ab=0.5, w_sat=0.1):
        super(ReconstructionLoss, self).__init__()
        self.w_rgb = w_rgb
        self.w_ab = w_ab
        self.w_sat = w_sat
        self.rgb_loss = RGBL1Loss()
        self.ab_loss = ColorABLoss()
        self.sat_loss = SaturationLoss()

    def forward(self, img_1, img_2, img_gt):
        loss_rgb = self.rgb_loss(img_1, img_2, img_gt)
        loss_ab = self.ab_loss(img_1, img_2, img_gt)
        loss_sat = self.sat_loss(img_1, img_2, img_gt)
        loss_rec = self.w_rgb * loss_rgb + self.w_ab * loss_ab +  self.w_sat * loss_sat
        return loss_rec, loss_rgb, loss_ab, loss_sat


class BranchConsistencyLoss(nn.Module):
    def __init__(self):
        super(BranchConsistencyLoss, self).__init__()
        self.mse_loss = nn.MSELoss()

    def forward(self, img_1, img_2):
        loss_con = self.mse_loss(torch.clamp(img_1, 0, 1), torch.clamp(img_2, 0, 1))
        return loss_con


class AdjustmentLoss(nn.Module):
    def __init__(self, w_rgb=5.0, w_ab=1.0, w_sat=1.0):
        super(AdjustmentLoss, self).__init__()
        self.recon_loss = ReconstructionLoss(w_rgb=w_rgb, w_ab=w_ab, w_sat=w_sat)
        self.branch_loss = BranchConsistencyLoss()

    def forward(self, img_1, img_2, img_gt):
        loss_rec, loss_rgb, loss_ab, loss_sat = self.recon_loss(img_1, img_2, img_gt)
        loss_con = self.branch_loss(img_1, img_2)
        loss_rec = loss_rec
        loss_con = 10 * loss_con
        loss_total = loss_rec + loss_con
        return loss_total, loss_rec, loss_con, loss_rgb, loss_ab, loss_sat


# ----------------3.Build the FusionLoss Fusion loss function----------------
class Sobelxy(nn.Module):
    def __init__(self):
        super(Sobelxy, self).__init__()
        kernelx = [[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]]
        kernely = [[1, 2, 1],
                   [0, 0, 0],
                   [-1, -2, -1]]
        kernelx = torch.FloatTensor(kernelx).unsqueeze(0).unsqueeze(0)
        kernely = torch.FloatTensor(kernely).unsqueeze(0).unsqueeze(0)
        self.weightx = nn.Parameter(data=kernelx, requires_grad=False).cuda()
        self.weighty = nn.Parameter(data=kernely, requires_grad=False).cuda()

    def forward(self, x):
        sobelx = F.conv2d(x, self.weightx, padding=1)
        sobely = F.conv2d(x, self.weighty, padding=1)
        return torch.abs(sobelx) + torch.abs(sobely)


class StructuralSimilarityLoss(nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(StructuralSimilarityLoss, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = self.create_window(window_size, self.channel)
        self.sobelconv = Sobelxy()

    def create_window(self, window_size, channel):
        _1D_window = self.gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
        return window

    @staticmethod
    def gaussian(window_size, sigma):
        gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
        return gauss / gauss.sum()

    @staticmethod
    def _ssim(img1, img2, window, window_size, channel, size_average=True):
        mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
        mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        if size_average:
            return ssim_map.mean()
        else:
            return ssim_map.mean(1).mean(1).mean(1)

    def ssim(self, img1, img2, window_size=11, size_average=True):
        (_, channel, _, _) = img1.size()
        window = self.create_window(window_size, channel)
        if img1.is_cuda:
            window = window.cuda(img1.get_device())
        window = window.type_as(img1)
        out = self._ssim(img1, img2, window, window_size, channel, size_average)
        return out

    def forward(self, image_vis, image_ir, image_fused):
        gradient_vis = self.sobelconv(image_vis)
        gradient_ir = self.sobelconv(image_ir)
        weight_vis = torch.mean(gradient_vis) / (torch.mean(gradient_vis) + torch.mean(gradient_ir))
        weight_ir = torch.mean(gradient_ir) / (torch.mean(gradient_vis) + torch.mean(gradient_ir))
        loss_ssim = weight_vis * self.ssim(image_vis, image_fused) + weight_ir * self.ssim(image_ir, image_fused)
        return loss_ssim


class IntensityLoss(nn.Module):
    def __init__(self):
        super(IntensityLoss, self).__init__()

    def forward(self, image_vis, image_ir, image_fused):
        intensity_joint = torch.max(image_vis, image_ir)
        loss_intensity = F.l1_loss(image_fused, intensity_joint)
        return loss_intensity


class GradientLoss(nn.Module):
    def __init__(self):
        super(GradientLoss, self).__init__()
        self.sobelconv = Sobelxy()

    def forward(self, image_vis, image_ir, image_fused):
        gradient_vis = self.sobelconv(image_vis)
        gradient_ir = self.sobelconv(image_ir)
        gradient_fused = self.sobelconv(image_fused)
        gradient_joint = torch.max(gradient_vis, gradient_ir)
        loss_gradient = F.l1_loss(gradient_fused, gradient_joint)
        return loss_gradient


class ColorConstancyLoss(nn.Module):
    def __init__(self):
        super(ColorConstancyLoss, self).__init__()

    @staticmethod
    def forward(img_enhanced):
        b, c, h, w = img_enhanced.shape
        mean_rgb = torch.mean(img_enhanced, [2, 3], keepdim=True)
        mean_r, mean_g, mean_b = torch.split(mean_rgb, 1, dim=1)
        diff_rg = torch.pow(mean_r - mean_g, 2)
        diff_rb = torch.pow(mean_r - mean_b, 2)
        diff_gb = torch.pow(mean_g - mean_b, 2)
        loss = torch.mean(torch.pow(torch.pow(diff_rg, 2) + torch.pow(diff_rb, 2) + torch.pow(diff_gb, 2), 0.5))
        return loss


class FusionLoss(nn.Module):
    def __init__(self):
        super(FusionLoss, self).__init__()
        self.int_loss = IntensityLoss()
        self.gard_loss = GradientLoss()
        self.color_loss = ColorConstancyLoss()

    def forward(self, image_vis, image_ir, image_fused, image_fused_rgb):
        loss_int = 5*self.int_loss(image_vis, image_ir, image_fused)
        loss_grad = 10 * self.gard_loss(image_vis, image_ir, image_fused)
        loss_color = 1 * self.color_loss(image_fused_rgb)
        loss_total = loss_grad + loss_int + loss_color
        return loss_total, loss_int, loss_grad, loss_color


if __name__ == '__main__':
    pass

