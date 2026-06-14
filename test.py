# coding:utf-8

import argparse
import warnings
import matplotlib
matplotlib.use('Agg')
from model.FusionNet import FusionNet
from model.EnhanceNet import EnhancementNet
from model.AdjustmentNet import ColorAdjustmentNet
from datasets import FusionDataset
from torch.utils.data import DataLoader
from PIL import Image
from utils import *
warnings.filterwarnings('ignore')


def main():
    # Network model loading
    fusion_model = FusionNet()
    fusion_model.eval()
    fusion_model.load_state_dict(torch.load(args.model_path))
    if args.gpu >= 0 and torch.cuda.is_available():
        fusion_model.cuda(args.gpu)

    enhance_model = EnhancementNet()
    enhance_model.eval().cuda()
    enhance_model.load_state_dict(torch.load('./weight/enhancement_model.pt'))

    adjustment_model = ColorAdjustmentNet()
    adjustment_model.eval().cuda()
    adjustment_model.load_state_dict(torch.load('./weight/color_adjustment_model.pt'))

    root_path = increment_path(base_dir="./result_imgs/fusion/test")
    print('done!')

    # Datasets loading
    test_dataset = FusionDataset(split='test', dataset_path=args.test_path)
    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=1,
                             shuffle=False,
                             num_workers=args.num_workers,
                             pin_memory=True,
                             drop_last=False,
    )
    test_loader.n_iter = len(test_loader)

    # Specific test process
    with torch.no_grad():
        for it, (img_vis, img_ir, name) in enumerate(test_loader):
            # Move the data to the GPU
            if args.gpu >= 0 and torch.cuda.is_available():
                img_vis = img_vis.to(f"cuda:{args.gpu}")
                img_ir = img_ir.to(f"cuda:{args.gpu}")

            # Fusion inference
            img_fused = fusion_inference(
                img_vis=img_vis,
                img_ir=img_ir,
                enhance_model=enhance_model,
                adjustment_model=adjustment_model,
                fusion_model=fusion_model,
                use_overlap_fusion=args.use_overlap_fusion,
                patch_overlap=args.patch_overlap)

            # Post-process the fused image
            img_fused = torch.clamp(img_fused, 0, 1)
            img_fused = img_fused.permute((0, 2, 3, 1)).cpu().numpy()
            img_fused = np.uint8(255.0 * img_fused)
            # save image
            for k in range(len(name)):
                image = img_fused[k, :, :, :]
                image = image.squeeze()
                image = Image.fromarray(image)
                save_path = os.path.join(root_path, name[k])
                image.save(save_path)
                print('Fusion {0} Sucessfully!'.format(save_path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train with pytorch')
    # Set general parameters
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--num_workers', type=int, default=8)
    # Set parameters related to fusion testing
    parser.add_argument('--model_path', '-F', type=str, default='./weight/fusion_model.pt')
    parser.add_argument('--test_path', type=str, default='./test_imgs')
    parser.add_argument('--use_overlap_fusion', type=bool, default=True, help='use overlap crop and smooth fusion')
    parser.add_argument('--patch_overlap', type=int, default=64, help='overlap size for patch-based fusion')
    args = parser.parse_args()

    main()