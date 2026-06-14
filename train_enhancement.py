#!/usr/bin/python
# -*- encoding: utf-8 -*-

import time
import datetime
import logging
import argparse
import warnings
import matplotlib
matplotlib.use('Agg')
from logger import setup_logger
from model.EnhanceNet import EnhancementNet
from PIL import Image
from datasets import EnhanceDataset
from torch.utils.data import DataLoader
from loss import LightBoostLoss
from torch.utils.tensorboard import SummaryWriter
from utils import *
warnings.filterwarnings('ignore')


def parse_args():
    parse = argparse.ArgumentParser()
    return parse.parse_args()


def train_enhancement():
    # Network model loading
    if args.en_mode == 'Color':
        enhance_model = EnhancementNet(in_channels=3)
    else:
        enhance_model = EnhancementNet(in_channels=1)
    enhance_model.cuda()

    # Loss function and optimizer loading
    enhance_loss = LightBoostLoss()
    optimizer = torch.optim.Adam(enhance_model.parameters(), lr=args.en_lr, weight_decay=1e-4)

    # Datasets loading
    enhance_dataset = EnhanceDataset(split='train', dataset_name=args.en_dataset, img_color=args.en_mode)
    print("the training enhance dataset is length:{}".format(enhance_dataset.length))
    train_loader = DataLoader(dataset=enhance_dataset,
                              batch_size=args.en_batch_size,
                              shuffle=True,
                              num_workers=0,
                              pin_memory=True,
                              drop_last=True,)
    train_loader.n_iter = len(train_loader)

    # Specific training process
    en_root_path = increment_path(base_dir="./result_imgs/enhance/train")
    print(f"Training results in {en_root_path}")
    epoch = args.en_epoch
    st = glob_st = time.time()
    # Set up logging
    log_path = os.path.join(en_root_path, "log")
    os.makedirs(log_path, exist_ok=True)
    logger = logging.getLogger()
    setup_logger(log_path)
    logger.info('Training Enhancement Model start~')
    # Log the loss
    writer = SummaryWriter(log_dir=os.path.join(en_root_path, "loss/tensorboard"))
    global_step = 0
    loss_history = {"loss_total": [], "loss_spa": [], "loss_tv": [], "loss_exp": [], "loss_contrast": []}

    for epo in range(0, epoch):
        lr_start = args.en_lr
        lr_decay = 0.9
        lr_this_epo = lr_start * (lr_decay ** epo)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr_this_epo

        for it, (img_low, name) in enumerate(train_loader):
            enhance_model.train()
            optimizer.zero_grad(set_to_none=True)
            # Move the data to the GPU
            if args.gpu >= 0 and torch.cuda.is_available():
                img_low = img_low.to(f"cuda:{args.gpu}")

            # EnhanceNet
            _, img_enhanced, img_illu = enhance_model(img_low)

            # Compute the loss
            loss_total, loss_spa, loss_tv, loss_exp, loss_contrast = enhance_loss(img_low, img_illu, img_enhanced)
            loss_history["loss_total"].append(loss_total.item())
            loss_history["loss_spa"].append(loss_spa.item())
            loss_history["loss_tv"].append(loss_tv.item())
            loss_history["loss_exp"].append(loss_exp.item())
            loss_history["loss_contrast"].append(loss_contrast.item())
            loss_total.backward()
            optimizer.step()
            writer.add_scalar("Loss/train", loss_total.item(), global_step)
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
                        'loss_spa: {loss_spa:.4f}',
                        'loss_tv: {loss_tv:.4f}',
                        'loss_exp: {loss_exp:.4f}',
                        'loss_contrast: {loss_contrast:.4f}',
                        'eta: {eta}',
                        'time: {time:.4f}',
                    ]).format(
                        it=now_it,
                        max_it=train_loader.n_iter * epoch,
                        lr_this_epo=lr_this_epo,
                        loss_total=loss_total.item(),
                        loss_spa=loss_spa.item(),
                        loss_tv=loss_tv.item(),
                        loss_exp=loss_exp.item(),
                        loss_contrast=loss_contrast.item(),
                        time=t_intv,
                        eta=eta,
                    )
                logger.info(msg)
                st = ed
    plot_loss_curves(loss_history, save_dir=os.path.join(en_root_path, "loss"))

    # Save the weights
    enhancement_model_file = os.path.join(en_root_path, "enhancement_model.pt")
    torch.save(enhance_model.state_dict(), enhancement_model_file)
    logger.info("Enhance Model Save to: {}".format(enhancement_model_file))
    logger.info('\n')


def test_enhancement():
    # Network model loading
    if args.en_mode == 'Color':
        enhance_model = EnhancementNet(in_channels=3)
    else:
        enhance_model = EnhancementNet(in_channels=1)
    enhance_model.eval()
    if args.gpu >= 0:
        enhance_model.cuda(args.gpu)
    enhance_model.load_state_dict(torch.load(args.en_model_path))

    en_root_path = increment_path(base_dir="./result_imgs/enhance/test")
    print(f"Testing results in {en_root_path}")

    # Datasets loading
    test_dataset = EnhanceDataset(split='test', lowimg_dir=args.en_test_path, img_color='Gray')
    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=1,
                             shuffle=False,
                             num_workers=args.num_workers,
                             pin_memory=True,
                             drop_last=False, )
    test_loader.n_iter = len(test_loader)

    # Specific test process
    with torch.no_grad():
        for it, (img_low, name) in enumerate(test_loader):
            # Move the data to the GPU
            if args.gpu >= 0 and torch.cuda.is_available():
                img_low = img_low.to(f"cuda:{args.gpu}")

            # Enhance inference
            _, img_enhanced, img_illu = enhance_model(img_low)

            # Post-process the fused image
            img_enhanced = img_enhanced.permute((0, 2, 3, 1)).cpu().numpy()
            img_enhanced = (img_enhanced - np.min(img_enhanced)) / (
                np.max(img_enhanced) - np.min(img_enhanced))
            img_enhanced = np.uint8(255.0 * img_enhanced)
            # save image
            for k in range(len(name)):
                image = img_enhanced[k, :, :, :]
                image = image.squeeze()
                image = Image.fromarray(image)
                save_path = os.path.join(en_root_path, name[k])
                image.save(save_path)
                print('Enhanced Test {0} Sucessfully!'.format(save_path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train with pytorch')
    # Set general parameters
    parser.add_argument('--seed', type=int, default=3, help='random seed')
    parser.add_argument('--model_name', type=str, default='LCDEN')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--num_workers', type=int, default=8)
    # Set parameters related to enhancement training and testing
    parser.add_argument('--en_mode', type=str, default='Gray', help='Color, Gray')
    parser.add_argument('--en_lr', type=float, default=0.0005)
    parser.add_argument('--en_dataset', type=str, default='MSRS')
    parser.add_argument('--en_batch_size', type=int, default=8)
    parser.add_argument('--en_epoch', type=int, default=10)
    parser.add_argument('--en_model_path', type=str, default='./weight/enhancement_model.pt')
    parser.add_argument('--en_test_path', type=str, default='./test_imgs/LLVIP/vi')
    args = parser.parse_args()

    # train_enhancement()
    test_enhancement()