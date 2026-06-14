import torch
import os
import math
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F


# -------------------Color space conversion---------------------
def rgb2hsv(img, eps=1e-6):
    r, g, b = img[:, 0:1, :, :], img[:, 1:2, :, :], img[:, 2:3, :, :]
    c_max, max_idx = torch.max(img, dim=1, keepdim=True)
    c_min, _ = torch.min(img, dim=1, keepdim=True)
    delta = c_max - c_min

    v = c_max
    s = delta / (c_max + eps)
    s = torch.where(c_max == 0.0, torch.zeros_like(s), s)
    h = torch.zeros_like(v)

    mask_r = (max_idx == 0) & (delta > 0)
    mask_g = (max_idx == 1) & (delta > 0)
    mask_b = (max_idx == 2) & (delta > 0)

    h = torch.where(mask_r, ((g - b) / (delta + eps)) % 6.0, h)
    h = torch.where(mask_g, ((b - r) / (delta + eps)) + 2.0, h)
    h = torch.where(mask_b, ((r - g) / (delta + eps)) + 4.0, h)
    h = h / 6.0
    h = h % 1.0
    return torch.cat([h, s, v], dim=1)


def rgb2ycrcb(img):
    img = img.float()
    R = img[:, 0:1, :, :]
    G = img[:, 1:2, :, :]
    B = img[:, 2:3, :, :]
    Y = 0.299 * R + 0.587 * G + 0.114 * B
    Cr = 0.713 * (R - Y) + 0.5
    Cb = 0.564 * (B - Y) + 0.5
    return torch.cat([Y, Cr, Cb], dim=1)


def ycrcb2rgb(img):
    img = img.float()
    Y = img[:, 0:1, :, :]
    Cr = img[:, 1:2, :, :]
    Cb = img[:, 2:3, :, :]
    R = Y + 1.403 * (Cr - 0.5)
    G = Y - 0.344 * (Cb - 0.5) - 0.714 * (Cr - 0.5)
    B = Y + 1.773 * (Cb - 0.5)
    return torch.clamp(torch.cat([R, G, B], dim=1), 0.0, 1.0)


def rgb2lab(img):
    mask = (img > 0.04045).float()
    linear_rgb = mask * torch.pow((img + 0.055) / 1.055, 2.4) + (1 - mask) * (img / 12.92)
    R, G, B = linear_rgb[:, 0:1, :, :], linear_rgb[:, 1:2, :, :], linear_rgb[:, 2:3, :, :]

    X = 0.412453 * R + 0.357580 * G + 0.180423 * B
    Y = 0.212671 * R + 0.715160 * G + 0.072169 * B
    Z = 0.019334 * R + 0.119193 * G + 0.950227 * B
    X = X / 0.950456
    Y = Y / 1.000000
    Z = Z / 1.088754

    def f(t):
        mask = (t > 0.008856).float()
        return mask * torch.pow(torch.abs(t) + 1e-8, 1 / 3) + (1 - mask) * (7.787 * t + 16 / 116)

    fX, fY, fZ = f(X), f(Y), f(Z)
    L = 116 * fY - 16
    a = 500 * (fX - fY)
    b = 200 * (fY - fZ)
    return L, a, b


def lab2rgb(l, a, b):
    fY = (l + 16) / 116
    fX = a / 500 + fY
    fZ = fY - b / 200

    def f_inv(t):
        mask = (t > 0.206893).float()  # 0.206893 = 0.008856^(1/3)
        return mask * torch.pow(t, 3) + (1 - mask) * ((t - 16 / 116) / 7.787)

    X = f_inv(fX) * 0.950456
    Y = f_inv(fY) * 1.000000
    Z = f_inv(fZ) * 1.088754

    R = 3.2404542 * X - 1.5371385 * Y - 0.4985314 * Z
    G = -0.9692660 * X + 1.8760108 * Y + 0.0415560 * Z
    B = 0.0556434 * X - 0.2040259 * Y + 1.0572252 * Z
    rgb = torch.cat([R, G, B], dim=1)

    rgb_clamped = torch.clamp(rgb, min=1e-8)
    mask = (rgb_clamped > 0.0031308).float()
    srgb = mask * (1.055 * torch.pow(rgb_clamped, 1 / 2.4) - 0.055) + (1 - mask) * (12.92 * rgb_clamped)
    return torch.clamp(srgb, 0.0, 1.0)


# -------Perform patch-wise inference with the fusion network-----------
def pad_to_multiple(x, multiple=8, mode='reflect'):
    b, c, h, w = x.shape
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    x_pad = F.pad(x, (0, pad_w, 0, pad_h), mode=mode)
    return x_pad, h, w


def crop_back_to_original(padded_x, pad_info, original_H, original_W):
    pad_left, pad_right, pad_top, pad_bottom = pad_info
    start_h = pad_top
    end_h = start_h + original_H
    start_w = pad_left
    end_w = start_w + original_W
    cropped_x = padded_x[:, :, start_h:end_h, start_w:end_w]
    return cropped_x


def process_one_patch(img_vis_patch, img_ir_patch, enhance_model, adjustment_model, fusion_model):
    # Perform the full fusion process for a single patch
    # Visible image enhance
    img_low_l, img_low_a, img_low_b = rgb2lab(img_vis_patch)
    _, img_enhanced_gray, img_illu = enhance_model(img_low_l / 100.0)
    img_enhanced_l = img_enhanced_gray * 100.0
    img_enhanced_a, img_enhanced_b = adaptive_color_enhance(img_low_l, img_low_a, img_low_b, img_enhanced_l)
    img_vis_en = lab2rgb(img_enhanced_l, img_enhanced_a, img_enhanced_b)

    # Color adjustment
    img_vis_enca = adjustment_model(img_vis_en)

    # Pre-process the image
    img_vis_ycrcb = rgb2ycrcb(img_vis_enca)
    img_vis_y = img_vis_ycrcb[:, :1, :, :]
    _, _, h_ori, w_ori = img_vis_y.shape
    img_vis_y_pad, _, _ = pad_to_multiple(img_vis_y, multiple=16)
    img_ir_pad, _, _ = pad_to_multiple(img_ir_patch, multiple=16)

    # Image fusion
    img_fused_y_pad = fusion_model(img_vis_y_pad, img_ir_pad)

    # Post-process the image
    img_fused_y = img_fused_y_pad[:, :, :h_ori, :w_ori]
    fusion_ycrcb = torch.cat((img_fused_y, img_vis_ycrcb[:, 1:2, :, :], img_vis_ycrcb[:, 2:, :, :]),dim=1)
    img_fused_patch = ycrcb2rgb(fusion_ycrcb)
    img_fused_patch = torch.clamp(img_fused_patch, 0, 1)
    return img_fused_patch


def smooth_ramp(length, device, dtype):
    # Generate cosine-smoothed weights from 0 to 1
    if length <= 1:
        return torch.ones(length, device=device, dtype=dtype)
    t = torch.linspace(0, 1, steps=length, device=device, dtype=dtype)
    pi = torch.tensor(math.pi, device=device, dtype=dtype)
    ramp = 0.5 - 0.5 * torch.cos(pi * t)
    return ramp


def build_blend_weight(patch_h, patch_w, y0, y1, x0, x1, full_h, full_w, overlap_h, overlap_w, device, dtype):
    weight = torch.ones((1, 1, patch_h, patch_w), device=device, dtype=dtype)
    # Vertical weight
    if overlap_h > 0:
        blend_h = min(2 * overlap_h, patch_h)
        if y0 > 0:
            ramp = smooth_ramp(blend_h, device, dtype)
            ramp = ramp.view(1, 1, blend_h, 1)
            weight[:, :, :blend_h, :] *= ramp
        if y1 < full_h:
            ramp = smooth_ramp(blend_h, device, dtype)
            ramp = torch.flip(ramp, dims=[0])
            ramp = ramp.view(1, 1, blend_h, 1)
            weight[:, :, -blend_h:, :] *= ramp

    # Horizontal weight
    if overlap_w > 0:
        blend_w = min(2 * overlap_w, patch_w)
        if x0 > 0:
            ramp = smooth_ramp(blend_w, device, dtype)
            ramp = ramp.view(1, 1, 1, blend_w)
            weight[:, :, :, :blend_w] *= ramp
        if x1 < full_w:
            ramp = smooth_ramp(blend_w, device, dtype)
            ramp = torch.flip(ramp, dims=[0])
            ramp = ramp.view(1, 1, 1, blend_w)
            weight[:, :, :, -blend_w:] *= ramp
    return weight


def fusion_by_patches(img_vis, img_ir, enhance_model, adjustment_model, fusion_model, overlap=64, use_overlap_fusion=True):
    # Fuse directly without patching
    if not use_overlap_fusion:
        return process_one_patch(img_vis, img_ir, enhance_model, adjustment_model, fusion_model)

    # Split the image into four patches for fusion
    b, c, h, w = img_vis.shape
    device = img_vis.device
    dtype = img_vis.dtype
    h_mid = h // 2
    w_mid = w // 2
    overlap_h = min(overlap, max(0, h_mid - 1))
    overlap_w = min(overlap, max(0, w_mid - 1))

    # Divide the image into patches
    patches = [
        (0, min(h, h_mid + overlap_h), 0, min(w, w_mid + overlap_w)),
        (0, min(h, h_mid + overlap_h), max(0, w_mid - overlap_w), w),
        (max(0, h_mid - overlap_h), h, 0, min(w, w_mid + overlap_w)),
        (max(0, h_mid - overlap_h), h, max(0, w_mid - overlap_w), w)
    ]

    output_sum = torch.zeros((b, 3, h, w), device=device, dtype=dtype)
    weight_sum = torch.zeros((b, 1, h, w), device=device, dtype=dtype)

    # Fuse patch by patch.
    for idx, (y0, y1, x0, x1) in enumerate(patches):
        img_vis_patch = img_vis[:, :, y0:y1, x0:x1]
        img_ir_patch = img_ir[:, :, y0:y1, x0:x1]
        fused_patch = process_one_patch(img_vis_patch, img_ir_patch, enhance_model, adjustment_model, fusion_model)
        patch_h = y1 - y0
        patch_w = x1 - x0
        weight = build_blend_weight(patch_h=patch_h, patch_w=patch_w, y0=y0, y1=y1, x0=x0, x1=x1, full_h=h, full_w=w,
                                    overlap_h=overlap_h, overlap_w=overlap_w, device=device, dtype=dtype)
        output_sum[:, :, y0:y1, x0:x1] += fused_patch * weight
        weight_sum[:, :, y0:y1, x0:x1] += weight
        # Clear the cache.
        del img_vis_patch
        del img_ir_patch
        del fused_patch
        del weight
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Normalize the data
    img_fused = output_sum / weight_sum.clamp_min(1e-6)
    img_fused = torch.clamp(img_fused, 0, 1)
    return img_fused


def fusion_inference(img_vis, img_ir, enhance_model, adjustment_model, fusion_model, use_overlap_fusion=False, patch_overlap=64):
    return fusion_by_patches(
        img_vis=img_vis,
        img_ir=img_ir,
        enhance_model=enhance_model,
        adjustment_model=adjustment_model,
        fusion_model=fusion_model,
        overlap=patch_overlap,
        use_overlap_fusion=use_overlap_fusion
    )


# --------------CA-SwinTransformer------------------
def window_partition(x, window_size):
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    windows = windows.view(-1, window_size * window_size, C)
    return windows


def window_reverse(windows, window_size, H, W, B):
    C = windows.shape[-1]
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, C)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    x = x.view(B, H, W, C)
    return x


# --------------Data augmentation-----------------
def augment_img(img, mode=0):
    '''Kai Zhang (github: https://github.com/cszn)'''
    if mode == 0:
        return img
    elif mode == 1:
        return np.flipud(np.rot90(img))
    elif mode == 2:
        return np.flipud(img)
    elif mode == 3:
        return np.rot90(img, k=3)
    elif mode == 4:
        return np.flipud(np.rot90(img, k=2))
    elif mode == 5:
        return np.rot90(img)
    elif mode == 6:
        return np.rot90(img, k=2)
    elif mode == 7:
        return np.flipud(np.rot90(img, k=3))


# --------------adaptive_color_enhance-----------------
def adaptive_color_enhance(img_low_l, img_low_a, img_low_b, img_enhanced_l, strength=0.7, eps=1e-6):
    # Clamp the brightness range
    img_low_l = torch.clamp(img_low_l, 0.0, 100.0)
    img_enhanced_l = torch.clamp(img_enhanced_l, 0.0, 100.0)

    # Brightness-guided weight
    dark_weight = 1.0 - img_low_l / 100.0
    dark_weight = torch.clamp(dark_weight, 0.0, 1.0)
    dark_weight = torch.pow(dark_weight + eps, 0.55)
    # Brightness-gain-guided weight
    delta_l = img_enhanced_l - img_low_l
    enhance_weight = torch.clamp(delta_l / 60.0, 0.0, 1.0)
    enhance_weight = torch.pow(enhance_weight + eps, 0.65)
    # Base brightness-gain weight
    base_weight = 0.18 * dark_weight
    # Brightness-guided composite weight
    light_weight = torch.sqrt(dark_weight * enhance_weight + eps)
    light_weight = torch.clamp(light_weight + base_weight, 0.0, 1.25)

    # Chroma-preserving weight
    chroma = torch.sqrt(img_low_a ** 2 + img_low_b ** 2 + eps)
    chroma_weight = torch.clamp(1.0 - chroma / 100.0, 0.55, 1.0)
    # Final chroma gain
    chroma_gain = 1.0 + strength * light_weight * chroma_weight
    img_enhanced_a = img_low_a * chroma_gain
    img_enhanced_b = img_low_b * chroma_gain

    # Limit the maximum chroma to avoid over-saturation clipping during Lab-to-RGB conversion
    chroma_enhanced = torch.sqrt(img_enhanced_a ** 2 + img_enhanced_b ** 2 + eps)
    clip_scale = torch.clamp(95.0 / (chroma_enhanced + eps), max=1.0)
    img_enhanced_a = img_enhanced_a * clip_scale
    img_enhanced_b = img_enhanced_b * clip_scale
    return img_enhanced_a, img_enhanced_b


# --------------other-----------------
def load_denoisenet_weights(model, weight_path, device):
    checkpoint = torch.load(weight_path, map_location=device)
    if 'params' in checkpoint:
        state_dict = checkpoint['params']
    else:
        state_dict = checkpoint
    model.load_state_dict(state_dict, strict=True)
    return model


def plot_rgb_separate_curves(curves, sample_idx=0, save_dir=None):
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["axes.unicode_minus"] = False
    curves_np = curves[sample_idx].detach().cpu().numpy()
    x = np.linspace(0, 1, curves_np.shape[-1])
    font_name = "Times New Roman"

    groups = {
        "R": {
            "indices": [0, 1, 2],
            "labels": ["R→R", "G→R", "B→R"],
            "title": "Adjustment Curves for Output R Channel"
        },
        "G": {
            "indices": [3, 4, 5],
            "labels": ["R→G", "G→G", "B→G"],
            "title": "Adjustment Curves for Output G Channel"
        },
        "B": {
            "indices": [6, 7, 8],
            "labels": ["R→B", "G→B", "B→B"],
            "title": "Adjustment Curves for Output B Channel"
        }
    }

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
    for channel_name, group in groups.items():
        plt.figure(figsize=(5, 3.5), dpi=150)
        colors = ["#D62728", "#2CA02C", "#1F77B4"]
        for idx, label, color in zip(group["indices"], group["labels"], colors):
            plt.plot(x, curves_np[idx], linewidth=1.5, label=label, color=color)
        zero_line_colors = {"R": "#D62728", "G": "#2CA02C", "B": "#1F77B4",}
        plt.axhline(0, linestyle="--", linewidth=1.0, color=zero_line_colors[channel_name], alpha=0.8)
        plt.xlim(0, 1)

        y_abs = max(abs(curves_np[group["indices"]]).max(), 1e-4)
        plt.ylim(-y_abs * 1.2, y_abs * 1.2)
        plt.xlabel("Input intensity", fontname=font_name, fontsize=10)
        plt.ylabel("Adjustment value", fontname=font_name, fontsize=10)
        plt.title(group["title"], fontname=font_name, fontsize=10)
        plt.xticks(fontname=font_name, fontsize=10)
        plt.yticks(fontname=font_name, fontsize=10)
        plt.legend(prop={"family": font_name, "size": 10}, loc = "upper left")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        if save_dir is not None:
            save_path = os.path.join(save_dir, f"{channel_name}_adjustment_curves.png")
            plt.savefig(save_path, bbox_inches="tight", dpi=300)
        plt.show()


def increment_path(base_dir="runs/train", prefix="exp"):
    os.makedirs(base_dir, exist_ok=True)
    dirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    existing = [d for d in dirs if d.startswith(prefix)]
    # find the maximum sequence number in the directory
    if prefix in existing:
        nums = [
            int(d.replace(prefix, ""))
            for d in existing
            if d.replace(prefix, "").isdigit()
        ]
        next_num = max(nums) + 1 if nums else 2
        new_dir = os.path.join(base_dir, f"{prefix}{next_num}")
    else:
        new_dir = os.path.join(base_dir, prefix)
    # create a file directory
    os.makedirs(new_dir, exist_ok=True)
    return new_dir


def plot_loss_curves(loss_dict, save_dir="loss_plots"):
    os.makedirs(save_dir, exist_ok=True)
    steps = range(1, len(next(iter(loss_dict.values()))) + 1)
    # total loss drawing
    plt.figure(figsize=(10, 6))
    for name, values in loss_dict.items():
        plt.plot(steps, values, label=name)
    plt.xlabel("Steps")
    plt.ylabel("Loss Value")
    plt.title("All Loss Curves")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, "loss_all.png"), dpi=300, bbox_inches="tight")
    plt.close()
    # draw sub-losses one by one
    for name, values in loss_dict.items():
        plt.figure(figsize=(8, 5))
        plt.plot(steps, values, label=name, color="blue")
        plt.xlabel("Steps")
        plt.ylabel("Loss Value")
        plt.title(f"{name} Curve")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(save_dir, f"{name}.png"), dpi=300, bbox_inches="tight")
        plt.close()