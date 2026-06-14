#!/usr/bin/python
# -*- encoding: utf-8 -*-

import os
import time
import datetime
import torch
import warnings
import argparse
import logging
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from logger import setup_logger
from loss import AdjustmentLoss
from torch.utils.data import DataLoader
from model.AdjustmentNet import ColorAdjustmentNet
from datasets import AdjustmentDataset
from utils import *
warnings.filterwarnings('ignore')


def parse_args():
    parse = argparse.ArgumentParser()
    return parse.parse_args()


def train_color_adjustment():
    # Network model loading
    adjustment_model = ColorAdjustmentNet()
    adjustment_model.cuda()

    # Loss function and optimizer loading
    adjustment_loss = AdjustmentLoss()
    optimizer = torch.optim.Adam(adjustment_model.parameters(), lr=args.ca_lr, weight_decay=1e-4)

    # Datasets loading
    adjustment_dataset = AdjustmentDataset(split='train', dataset_name=args.ca_dataset, img_color=True)
    print("the training adjustment dataset is length:{}".format(adjustment_dataset.length))
    train_loader = DataLoader(dataset=adjustment_dataset,
                              batch_size=args.ca_batch_size,
                              shuffle=True,
                              num_workers=0,
                              pin_memory=True,
                              drop_last=True,)
    train_loader.n_iter = len(train_loader)

    # Specific training process
    ca_root_path = increment_path(base_dir="./result_imgs/correct/train")
    print(f"Training results in {ca_root_path}")
    epoch = args.ca_epoch
    # Set up logging
    log_path = os.path.join(ca_root_path, "log")
    os.makedirs(log_path, exist_ok=True)
    logger = logging.getLogger()
    setup_logger(log_path)
    logger.info('Training Adjustment Model start~')
    # Log the loss
    global_step = 0
    loss_history = {"loss_total": []}
    st = glob_st = time.time()

    for epo in range(0, epoch):
        lr_start = args.ca_lr
        lr_decay = 0.75
        lr_this_epo = lr_start * (lr_decay ** epo)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr_this_epo

        for it, (img_color_bais_1, img_color_bais_2, img_gt, name) in enumerate(train_loader):
            adjustment_model.train()
            optimizer.zero_grad(set_to_none=True)

            # Move the data to the GPU
            if args.gpu >= 0 and torch.cuda.is_available():
                img_gt = img_gt.to(f"cuda:{args.gpu}")
                img_color_bais_1 = img_color_bais_1.to(f"cuda:{args.gpu}")
                img_color_bais_2 = img_color_bais_2.to(f"cuda:{args.gpu}")

            # Color Adjustment Net
            img_color_adjustment_1 = adjustment_model(img_color_bais_1)
            img_color_adjustment_2 = adjustment_model(img_color_bais_2)

            # Compute the loss
            loss_total, loss_rec, loss_con, loss_rgb, loss_ab, loss_sat = adjustment_loss(
                img_color_adjustment_1,
                img_color_adjustment_2,
                img_gt)
            loss_history["loss_total"].append(loss_total.item())
            loss_total.backward()
            optimizer.step()
            global_step += 1

            # Estimate the remaining training time
            ed = time.time()
            t_intv, glob_t_intv = ed - st, ed - glob_st
            now_it = train_loader.n_iter * epo + it + 1
            eta = int((train_loader.n_iter * epoch - now_it) * (glob_t_intv / (now_it)))
            eta = str(datetime.timedelta(seconds=eta))

            # Print the information
            if now_it % 1 == 0:
                msg = ', '.join(
                    [
                        'step: {it}/{max_it}',
                        'loss_learn:{lr_this_epo:.4F}',
                        'loss_total: {loss_total:.4f}',
                        'loss_rec: {loss_rec:.4f}',
                        'loss_con: {loss_con:.4f}',
                        'loss_rgb: {loss_rgb:.4f}',
                        'loss_ab: {loss_ab:.4f}',
                        'loss_sat: {loss_sat:.4f}',
                        'eta: {eta}',
                        'time: {time:.4f}',
                    ]).format(
                        it=now_it,
                        max_it=train_loader.n_iter * epoch,
                        lr_this_epo=lr_this_epo,
                        loss_total=loss_total.item(),
                        loss_rec=loss_rec.item(),
                        loss_con=loss_con.item(),
                        loss_rgb=loss_rgb.item(),
                        loss_ab=loss_ab.item(),
                        loss_sat=loss_sat.item(),
                        time=t_intv,
                        eta=eta,
                    )
                logger.info(msg)
                st = ed
    plot_loss_curves(loss_history, save_dir=os.path.join(ca_root_path, "loss"))

    # Save the weights
    adjustment_model_file = os.path.join(ca_root_path, "color_adjustment_model.pt")
    torch.save(adjustment_model.state_dict(), adjustment_model_file)
    logger.info("Correct Model Save to: {}".format(adjustment_model_file))
    logger.info('\n')

    # Visualize and save the training results
    adjustment_model.eval()
    save_dir = os.path.join(ca_root_path, "images")
    os.makedirs(save_dir, exist_ok=True)
    visual_loader = DataLoader(dataset=adjustment_dataset,
                               batch_size=1,
                               shuffle=False,
                               num_workers=0,
                               pin_memory=True)
    with torch.no_grad():
        for it, (img_color_bais_1, img_color_bais_2, img_gt, name) in enumerate(visual_loader):
            if args.gpu >= 0 and torch.cuda.is_available():
                img_gt = img_gt.to(f"cuda:{args.gpu}")
                img_color_bais_1 = img_color_bais_1.to(f"cuda:{args.gpu}")
                img_color_bais_2 = img_color_bais_2.to(f"cuda:{args.gpu}")

            img_color_adjustment_1 = adjustment_model(img_color_bais_1)
            img_color_adjustment_2 = adjustment_model(img_color_bais_2)

            img_color_bais_1 = torch.clamp(img_color_bais_1, 0, 1).permute((0, 2, 3, 1)).cpu().numpy()
            img_color_bais_2 = torch.clamp(img_color_bais_2, 0, 1).permute((0, 2, 3, 1)).cpu().numpy()
            img_color_adjustment_1 = torch.clamp(img_color_adjustment_1, 0, 1).permute((0, 2, 3, 1)).cpu().numpy()
            img_color_adjustment_2 = torch.clamp(img_color_adjustment_2, 0, 1).permute((0, 2, 3, 1)).cpu().numpy()

            img_color_bais_1 = np.uint8(255.0 * img_color_bais_1)
            img_color_bais_2 = np.uint8(255.0 * img_color_bais_2)
            img_color_adjustment_1 = np.uint8(255.0 * img_color_adjustment_1)
            img_color_adjustment_2 = np.uint8(255.0 * img_color_adjustment_2)
            for k in range(len(name)):
                image_cb_1 = img_color_bais_1[k, :, :, :]
                image_cb_2 = img_color_bais_2[k, :, :, :]
                image_ca_1 = img_color_adjustment_1[k, :, :, :]
                image_ca_2 = img_color_adjustment_2[k, :, :, :]

                image_cb_1 = image_cb_1.squeeze()
                image_cb_2 = image_cb_2.squeeze()
                image_ca_1 = image_ca_1.squeeze()
                image_ca_2 = image_ca_2.squeeze()

                image_cb_1 = Image.fromarray(image_cb_1)
                image_cb_2 = Image.fromarray(image_cb_2)
                image_ca_1 = Image.fromarray(image_ca_1)
                image_ca_2 = Image.fromarray(image_ca_2)

                base_name = os.path.splitext(name[k])[0]
                save_path_cb_1 = os.path.join(save_dir, base_name + "_cb1.png")
                save_path_cb_2 = os.path.join(save_dir, base_name + "_cb2.png")
                save_path_ca_1 = os.path.join(save_dir, base_name + "_ca1.png")
                save_path_ca_2 = os.path.join(save_dir, base_name + "_ca2.png")

                image_cb_1.save(save_path_cb_1)
                image_cb_2.save(save_path_cb_2)
                image_ca_1.save(save_path_ca_1)
                image_ca_2.save(save_path_ca_2)
                print(f"Saved enhanced image: {os.path.join(save_dir, name[k])}")
    print(f"Adjustment training images saved to: {save_dir}")


def test_color_adjustment():
    # Network model loading
    adjustment_model = ColorAdjustmentNet()
    adjustment_model.eval()
    if args.gpu >= 0:
        adjustment_model.cuda(args.gpu)
    adjustment_model.load_state_dict(torch.load(args.ca_model_path))

    ca_root_path = increment_path(base_dir="./result_imgs/correct/test")
    print(f"Testing results in {ca_root_path}")

    # Datasets loading
    test_dataset = AdjustmentDataset(split='test', img_dir=args.ca_test_path, img_color=True, is_train=False)
    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=1,
                             shuffle=False,
                             num_workers=args.num_workers,
                             pin_memory=True,
                             drop_last=False, )
    test_loader.n_iter = len(test_loader)

    # Specific test process
    with torch.no_grad():
        for it, (img_gt, name) in enumerate(test_loader):
            # Move the data to the GPU
            if args.gpu >= 0 and torch.cuda.is_available():
                img_gt = img_gt.to(f"cuda:{args.gpu}")

            # Adjustment inference
            img_color_adjustment = adjustment_model(img_gt)

            # Post-process the fused image
            img_color_adjustment = torch.clamp(img_color_adjustment, 0, 1).permute((0, 2, 3, 1)).cpu().numpy()
            img_color_adjustment = np.uint8(255.0 * img_color_adjustment)
            for k in range(len(name)):
                image = img_color_adjustment[k, :, :, :]
                image = image.squeeze()
                image = Image.fromarray(image)
                save_path = os.path.join(ca_root_path, name[k])
                image.save(save_path)
                print(f"Saved enhanced image: {os.path.join(ca_root_path, name[k])}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train with pytorch')
    # Set general parameters
    parser.add_argument('--seed', type=int, default=3, help='random seed')
    parser.add_argument('--model_name', type=str, default='LCEFusion')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--num_workers', type=int, default=8)
    # Set parameters related to enhancement training and testing
    parser.add_argument('--ca_lr', type=float, default=0.0005)
    parser.add_argument('--ca_dataset', type=str, default='Flickr2K')
    parser.add_argument('--ca_batch_size', type=int, default=32)
    parser.add_argument('--ca_epoch', type=int, default=1)
    parser.add_argument('--ca_model_path', type=str, default='./weight/color_adjustment_model.pt')
    parser.add_argument('--ca_test_path', type=str, default='./test_imgs/MSRS/vi')
    args = parser.parse_args()

    # train_color_adjustment()
    test_color_adjustment()