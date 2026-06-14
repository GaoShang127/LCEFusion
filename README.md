# LCEFusion
Official PyTorch implementation of LCEFusion: Luminance-Color Decoupled Enhancement for Low-Light Infrared-Visible Image Fusion. LCEFusion is a low-light infrared and visible image fusion framework designed to improve luminance, color naturalness, texture preservation, and infrared target saliency in nighttime perception scenarios.

## Repository Structure

```text
LCEFusion/
├── dataset/                 # Dataset root
├── logs/                    # Training and testing logs
├── model/                   # Network architectures
│   ├── AdjustmentNet.py     # Image-level color adjustment network, ICAN
│   ├── DenoiseNet.py        # Denoising network modules
│   ├── EnhanceNet.py        # Luminance-color enhancement network
│   ├── FusionNet.py         # Infrared-visible image fusion network
│   └── ResNet.py            # Residual network blocks
├── result_imgs/             # Saved enhancement and fusion results
├── test_imgs/               # Input images for testing
├── weight/                  # Pretrained model weights
├── datasets.py              # Dataset loading and preprocessing
├── logger.py                # Logging utilities
├── loss.py                  # Loss functions
├── optimizer.py             # Optimizer settings
├── test.py                  # Testing script
├── train_adjustment.py      # Training script for ICAN
├── train_enhancement.py     # Training script for enhancement network
├── train_fusion.py          # Training script for fusion network
├── utils.py                 # Utility functions
└── README.md
```

## Installation and Requirements

This project is recommended to run on Linux with an independent Conda environment. The main experimental environment is listed below:

```text
Python              3.8.18
PyTorch             2.0.0+cu118
torchvision         0.15.0+cu118
CUDA                11.8
OpenCV              4.8.1.78
NumPy               1.21.6
SciPy               1.9.1
scikit-image        0.21.0
Pillow              9.5.0
Matplotlib          3.5.0
Kornia              0.7.3
einops              0.8.1
timm                1.0.19
TensorBoard         2.14.0
tqdm                4.66.1
```

### 1. Clone the Repository

```bash
git clone https://github.com/GaoShang127/LCEFusion.git
cd LCEFusion
```

### 2. Create a Conda Environment

```bash
conda create -n lcefusion python=3.8
conda activate lcefusion
```

### 3. Install PyTorch

This project uses PyTorch with CUDA 11.8:

```bash
pip install torch==2.0.0+cu118 torchvision==0.15.0+cu118 \
  --extra-index-url https://download.pytorch.org/whl/cu118
```

If CUDA is not available on your device, you can install the CPU version of PyTorch instead, but training and inference will be slower.

### 4. Install Other Dependencies

```bash
pip install opencv-python==4.8.1.78
pip install opencv-contrib-python==4.8.1.78
pip install numpy==1.21.6
pip install scipy==1.9.1
pip install scikit-image==0.21.0
pip install pillow==9.5.0
pip install matplotlib==3.5.0
pip install kornia==0.7.3
pip install einops==0.8.1
pip install timm==1.0.19
pip install tensorboard==2.14.0
pip install tqdm==4.66.1
pip install loguru==0.7.3
pip install pytorch-msssim==1.0.0
```

### 5. Verify the Environment

After installation, run the following command to check whether PyTorch and CUDA are available:

```bash
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('CUDA version:', torch.version.cuda)"
```

If `CUDA available` returns `True`, the GPU environment has been successfully configured.


## Dataset Preparation

Please organize infrared and visible image pairs as follows:

```text
dataset/
├── correct/                         # Color correction / ICAN training data
│   └── Flickr2K/
│       ├── train/
│       │   ├── ir/
│       │   │   ├── 00001.png
│       │   │   ├── 00002.png
│       │   │   └── ...
│       │   └── vi/
│       │       ├── 00001.png
│       │       ├── 00002.png
│       │       └── ...
│       └── test/
│           ├── ir/
│           │   ├── 00001.png
│           │   ├── 00002.png
│           │   └── ...
│           └── vi/
│               ├── 00001.png
│               ├── 00002.png
│               └── ...
│
├── enhance/                         # Low-light visible image enhancement data
│   └── LLVIP/
│       ├── train/
│       │   ├── ir/
│       │   │   ├── 00001.png
│       │   │   ├── 00002.png
│       │   │   └── ...
│       │   └── vi/
│       │       ├── 00001.png
│       │       ├── 00002.png
│       │       └── ...
│       └── test/
│           ├── ir/
│           │   ├── 00001.png
│           │   ├── 00002.png
│           │   └── ...
│           └── vi/
│               ├── 00001.png
│               ├── 00002.png
│               └── ...
│
└── fusion/                          # Infrared-visible image fusion data
    └── LLVIP/
        ├── train/
        │   ├── ir/
        │   │   ├── 00001.png
        │   │   ├── 00002.png
        │   │   └── ...
        │   └── vi/
        │       ├── 00001.png
        │       ├── 00002.png
        │       └── ...
        └── test/
            ├── ir/
            │   ├── 00001.png
            │   ├── 00002.png
            │   └── ...
            └── vi/
                ├── 00001.png
                ├── 00002.png
                └── ...
```


The infrared and visible images should be spatially aligned and have the same file names.

## Training

LCEFusion contains three training stages: color adjustment, low-light enhancement, and infrared-visible image fusion. Please make sure the datasets are prepared according to the required directory structure before training.

### 1. Train the Image-level Color Adjustment Network

The color adjustment network is trained using the data under:

```text
dataset/correct/Flickr2K/
```

Run:

```bash
python train_adjustment.py
```

The trained weights will be saved in the `weight/` directory, and the training logs will be saved in `logs/`.

### 2. Train the Enhancement Network

The enhancement network is trained using the low-light visible image data under:

```text
dataset/enhance/LLVIP/
```

Run:

```bash
python train_enhancement.py
```

This stage aims to enhance the luminance and color representation of low-light visible images. The generated model weights will be saved in `weight/`.

### 3. Train the Fusion Network

The fusion network is trained using aligned infrared and visible image pairs under:

```text
dataset/fusion/LLVIP/
```

Run:

```bash
python train_fusion.py
```

This stage learns to fuse the infrared image and the enhanced visible image. The final fusion model will be saved in the `weight/` directory.

During training, intermediate logs and training information are stored in:

```text
logs/
```

## Testing

After training or downloading the pretrained weights, place the model weights in the following directory:

```text
weight/
```

Test images can be placed under:

```text
test_imgs/
```

or organized according to the dataset testing structure:

```text
dataset/fusion/LLVIP/test/
├── ir/
│   ├── 00001.png
│   ├── 00002.png
│   └── ...
└── vi/
    ├── 00001.png
    ├── 00002.png
    └── ...
```

To generate fused images, run:

```bash
python test.py
```

The fused results will be saved in:

```text
result_imgs/
```

Please make sure that the infrared and visible images are spatially aligned and share the same file names before testing.


## Pretrained Model

Pretrained weights will be released at:

```text
(https://pan.baidu.com/s/16Cintf-sPJgf0qHPhRtXWA?pwd=jfg1/weight)
```

Please download the checkpoint and place it under the `weight/` directory.
