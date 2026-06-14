# LCEFusion
Official PyTorch implementation of LCEFusion: Luminance-Color Decoupled Enhancement for Low-Light Infrared-Visible Image Fusion.

LCEFusion is a low-light infrared and visible image fusion framework designed to improve luminance, color naturalness, texture preservation, and infrared target saliency in nighttime perception scenarios.

## Highlights

* **Luminance–Color Decoupled Enhancement**
  Separates luminance enhancement and color restoration for low-light visible images.

* **Adaptive Color Enhancement**
  Enhances chrominance information in dark regions while preserving hue consistency.

* **Image-level Color Adjustment Network (ICAN)**
  Corrects global and local color imbalance to improve naturalness and stability.

* **Cross-modal Image Fusion Network**
  Fuses enhanced visible images and infrared images with complementary feature interaction.

* **Dual-Frequency Gated Fusion Module (DFGF)**
  Preserves high-frequency texture details and low-frequency structural information during fusion.

## Framework

The proposed framework contains two main stages:

1. **Luminance-Color Decoupled Enhancement Network (LCDEN)**
   The visible image is converted into the CIELAB color space. The luminance component is enhanced independently, while the chrominance components are adaptively restored and further adjusted by ICAN.

2. **Image Fusion Network (IFN)**
   The enhanced visible image and the infrared image are fed into a dual-branch fusion network. Cross-modal features are interacted through CFIM, and texture/structure information is preserved by DFGF.

The final fused image combines infrared target saliency, visible texture details, enhanced brightness, and natural color representation.

## Repository Structure

```text
LCEFusion/
├── configs/                 # Configuration files
├── datasets/                # Dataset preparation scripts or dataset root
├── models/                  # Network architecture
│   ├── enhancement/         # LCDEN and ICAN
│   ├── fusion/              # IFN, CFIM, DFGF
│   └── modules/             # Basic network blocks
├── losses/                  # Enhancement and fusion loss functions
├── scripts/                 # Training and testing scripts
├── utils/                   # Utility functions
├── checkpoints/             # Pretrained models
├── results/                 # Fusion results
├── train.py                 # Training entry
├── test.py                  # Testing entry
└── README.md
```

## Installation

```bash
git clone https://github.com/your-username/LCEFusion.git
cd LCEFusion

conda create -n lcefusion python=3.8
conda activate lcefusion

pip install -r requirements.txt
```

## Requirements

The code is expected to run with the following environment:

```text
Python >= 3.8
PyTorch >= 1.10
torchvision
opencv-python
numpy
scipy
scikit-image
tqdm
Pillow
matplotlib
```

You can install the dependencies with:

```bash
pip install -r requirements.txt
```

## Dataset Preparation

Please organize infrared and visible image pairs as follows:

```text
datasets/
├── train/
│   ├── ir/
│   │   ├── 00001.png
│   │   ├── 00002.png
│   │   └── ...
│   └── vi/
│       ├── 00001.png
│       ├── 00002.png
│       └── ...
├── test/
│   ├── ir/
│   └── vi/
```

The infrared and visible images should be spatially aligned and have the same file names.

## Training

To train LCEFusion, run:

```bash
python train.py \
  --config configs/lcefusion.yaml \
  --train_ir datasets/train/ir \
  --train_vi datasets/train/vi \
  --save_dir checkpoints/lcefusion
```

You may modify the training settings in the configuration file:

```yaml
epochs: 100
batch_size: 4
learning_rate: 1e-4
image_size: 256
save_interval: 10
```

## Testing

To generate fused images using a trained model:

```bash
python test.py \
  --checkpoint checkpoints/lcefusion/best.pth \
  --ir_dir datasets/test/ir \
  --vi_dir datasets/test/vi \
  --save_dir results/lcefusion
```

The fused results will be saved in:

```text
results/lcefusion/
```

## Pretrained Model

Pretrained weights will be released at:

```text
checkpoints/lcefusion/best.pth
```

Please download the checkpoint and place it under the `checkpoints/` directory.

## Evaluation

The fusion results can be evaluated using common infrared-visible image fusion metrics, including:

* EN: Entropy
* AG: Average Gradient
* SF: Spatial Frequency
* VIF: Visual Information Fidelity
* EI: Edge Intensity
* QCB: Chen-Blum metric
* CQE: Color Quality Evaluation
* CM: Colorfulness Metric

Example:

```bash
python scripts/evaluate.py \
  --fused_dir results/lcefusion \
  --ir_dir datasets/test/ir \
  --vi_dir datasets/test/vi
```

## Results

LCEFusion aims to produce fused images with:

* enhanced luminance in low-light scenes;
* natural and stable color appearance;
* clear texture and structural details;
* salient infrared targets;
* robust performance in nighttime perception scenarios.

Qualitative and quantitative results will be updated after the release of the complete code and pretrained models.

## Citation

If this work is helpful for your research, please cite:

```bibtex
@article{lcefusion2026,
  title   = {LCEFusion: Luminance-Color Decoupled Enhancement for Low-Light Infrared and Visible Image Fusion},
  author  = {Gao, Shang and Yan, Aiyun and Wang, Xu and Meng, LongYue and Wang, Qi and Jin, Shuowei},
  journal = {Preprint},
  year    = {2026}
}
```

## Acknowledgements

This project is developed for low-light infrared and visible image fusion research. We thank the authors of related infrared-visible fusion and low-light enhancement methods for their open-source contributions.

## License

This repository is released for academic research only. For commercial use, please contact the authors.

## Contact

For questions, please contact:

```text
Gao Shang: gaoshang_neu@163.com
```
