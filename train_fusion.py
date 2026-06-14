#!/usr/bin/python
# -*- encoding: utf-8 -*-

import time
import datetime
import argparse
import warnings
import logging
import matplotlib
matplotlib.use('Agg')
from model.FusionNet import FusionNet
from model.EnhanceNet import EnhancementNet
from model.AdjustmentNet import ColorAdjustmentNet
from datasets import FusionDataset
from torch.utils.data import DataLoader
from loss import FusionLoss
from logger import setup_logger
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from utils import *
warnings.filterwarnings('ignore')


def parse_args():
    parse = argparse.ArgumentParser()
    return parse.parse_args()


def train_fusion(logger=None):
    # Network model loading
    fusion_model = FusionNet()
    fusion_model.cuda()
    enhance_model = EnhancementNet(in_channels=1, iter_num=8)
    enhance_model.eval().cuda()

    # Loss function and optimizer loading
    fusion_loss = FusionLoss()
    optimizer = torch.optim.Adam(fusion_model.parameters(), lr=args.f_lr, betas=(0.9, 0.999), weight_decay=1e-4)

    # Datasets loading
    fusion_dataset = FusionDataset(split='train', dataset_names=args.f_dataset)
    print("the training fusion dataset is length:{}".format(fusion_dataset.length))
    train_loader = DataLoader(dataset=fusion_dataset,
                              batch_size=args.f_batch_size,
                              shuffle=True,
                              num_workers=0,
                              pin_memory=True,
                              drop_last=True,)
    train_loader.n_iter = len(train_loader)

    # Specific training process
    f_root_path = increment_path(base_dir="./result_imgs/fusion/train")
    print(f"Training results in {f_root_path}")
    epoch = args.f_epoch
    st = glob_st = time.time()     # record the start time
    logger.info('Training Fusion Model start~')
    writer = SummaryWriter(log_dir=os.path.join(f_root_path, "loss/tensorboard"))
    global_step = 0
    loss_history = {"loss_total": [], "loss_int": [], "loss_gard": []}

    for epo in range(0, epoch):
        lr_start = args.f_lr
        lr_decay = 0.9
        lr_this_epo = lr_start * lr_decay ** epo
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr_this_epo

        for it, (img_vis, img_ir, name) in enumerate(train_loader):
            fusion_model.train()
            optimizer.zero_grad(set_to_none=True)
            # Move the data to the GPU
            if args.gpu >= 0 and torch.cuda.is_available():
                img_vis = img_vis.to(f"cuda:{args.gpu}")
                img_ir = img_ir.to(f"cuda:{args.gpu}")
            # Convert the RGB color space to YCrCb
            img_vis_ycrcb = rgb2ycrcb(img_vis)
            img_vis_y = img_vis_ycrcb[:, :1]

            # EnhanceNet & FusionNet
            _, img_vi_en, img_illu = enhance_model(img_vis_y)
            img_fused_y = fusion_model(img_vi_en, img_ir)

            # Convert the YCrCb color space to RGB
            fusion_ycrcb = torch.cat((img_fused_y, img_vis_ycrcb[:, 1:2, :, :], img_vis_ycrcb[:, 2:, :, :]), dim=1)
            img_fused = ycrcb2rgb(fusion_ycrcb)
            # Compute the loss
            loss_total, loss_int, loss_gard, loss_color = fusion_loss(img_vi_en, img_ir, img_fused_y, img_fused)
            loss_history["loss_total"].append(loss_total.item())
            loss_history["loss_int"].append(loss_int.item())
            loss_history["loss_gard"].append(loss_gard.item())
            loss_total.backward()
            torch.nn.utils.clip_grad_norm_(fusion_model.parameters(), max_norm=1.0)
            optimizer.step()
            writer.add_scalar("Loss/train", loss_total.item(), global_step)
            global_step += 1

            # Estimate the remaining training time
            ed = time.time()        # Record the end time.
            t_intv, glob_t_intv = ed - st, ed - glob_st
            now_it = train_loader.n_iter * epo + it + 1
            eta = int((train_loader.n_iter * epoch - now_it) * (glob_t_intv / now_it))
            eta = str(datetime.timedelta(seconds=eta))

            # Print the information
            if now_it % 10 == 0:
                msg = (', '.join(
                    [
                        'step: {it}/{max_it}',
                        'loss_learn:{lr_this_epo:.4F}',
                        'loss_total: {loss_total:.4f}',
                        'loss_int: {loss_int:.4f}',
                        'loss_gard: {loss_gard:.4f}',
                        'loss_color: {loss_color:.4f}',
                        'eta: {eta}',
                        'time: {time:.4f}',
                    ]).format(
                        it=now_it,
                        max_it=train_loader.n_iter * epoch,
                        lr_this_epo=lr_this_epo,
                        loss_total=loss_total.item(),
                        loss_int=loss_int.item(),
                        loss_gard=loss_gard.item(),
                        loss_color=loss_color.item(),
                        time=t_intv,
                        eta=eta,
                    ))
                logger.info(msg)
                st = ed
    writer.close()
    plot_loss_curves(loss_history, save_dir=os.path.join(f_root_path, "loss"))

    # Save the weights
    fusion_model_file = os.path.join(f_root_path, "fusion_model.pt")
    torch.save(fusion_model.state_dict(), fusion_model_file)
    logger.info("Fusion Model Save to: {}".format(fusion_model_file))
    logger.info('\n')


def test_fusion():
    # Network model loading
    fusion_model = FusionNet()
    fusion_model.eval()
    fusion_model.load_state_dict(torch.load(args.f_model_path))
    if args.gpu >= 0 and torch.cuda.is_available():
        fusion_model.cuda(args.gpu)

    enhance_model = EnhancementNet()
    enhance_model.eval().cuda()
    enhance_model.load_state_dict(torch.load('./weight/enhancement_model.pt'))

    adjustment_model = ColorAdjustmentNet()
    adjustment_model.eval().cuda()
    adjustment_model.load_state_dict(torch.load('./weight/color_adjustment_model.pt'))

    f_root_path = increment_path(base_dir="./result_imgs/fusion/test")
    print('done!')

    # Datasets loading
    test_dataset = FusionDataset(split='test', dataset_path=args.f_test_path)
    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=1,
                             shuffle=False,
                             num_workers=args.num_workers,
                             pin_memory=True,
                             drop_last=False,)
    test_loader.n_iter = len(test_loader)

    # Specific test process
    with torch.no_grad():
        for it, (img_vis, img_ir, name) in enumerate(test_loader):
            # Move the data to the GPU
            if args.gpu >= 0 and torch.cuda.is_available():
                img_vis = img_vis.to(f"cuda:{args.gpu}")
                img_ir = img_ir.to(f"cuda:{args.gpu}")

            # Fusion inference
            img_fused = fusion_inference(img_vis=img_vis,
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
                save_path = os.path.join(f_root_path, name[k])
                image.save(save_path)
                print('Fusion {0} Sucessfully!'.format(save_path))
            del img_fused
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train with pytorch')
    # Set general parameters
    parser.add_argument('--seed', type=int, default=3, help='random seed')
    parser.add_argument('--model_name', type=str, default='LCEFusion')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--num_workers', type=int, default=8)
    # Set parameters related to fusion training and testing
    parser.add_argument('--f_lr', type=float, default=0.0001)
    parser.add_argument('--f_dataset', type=list, default=['LLVIP', 'M3FD', 'MSRS', 'RoadScene'])
    parser.add_argument('--f_batch_size', type=int, default=4)
    parser.add_argument('--f_epoch', type=int, default=10)
    # parser.add_argument('--f_model_path', '-F', type=str, default='./weight/fusion_model.pt')
    parser.add_argument('--f_model_path', '-F', type=str, default='./weight/fusion_model.pt')
    parser.add_argument('--f_test_path', type=str, default='./test_imgs')
    parser.add_argument('--use_overlap_fusion', type=bool, default=True, help='use overlap crop and smooth fusion')
    parser.add_argument('--patch_overlap', type=int, default=64, help='overlap size for patch-based fusion')
    args = parser.parse_args()

    # Set up logging
    logpath = './logs'
    logger = logging.getLogger()
    setup_logger(logpath)

    # network train
    train_fusion(logger)
    # network test
    # test_fusion()