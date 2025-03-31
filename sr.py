import sys
import os
import time
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
sys.path.insert(0, os.path.abspath("D:/FYP-001"))

import torch
from torch.utils.data import DataLoader
from dataloader.MS2_dataset import DataLoader_MS2

import data as Data
import matplotlib.pyplot as plt
import model as Model
import argparse
import logging
import core.logger as Logger
import core.metrics as Metrics
from core.wandb_logger import WandbLogger
from torch.utils.tensorboard import SummaryWriter
import os
import numpy as np
import wandb
import utils
import random
from model.sr3_modules import transformer

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config/shadow.json',
                        help='JSON file for configuration')
    parser.add_argument('-p', '--phase', type=str, choices=['train', 'val'],
                        help='Run either train(training) or val(generation)', default='train')
    parser.add_argument('-gpu', '--gpu_ids', type=str, default=None)
    parser.add_argument('-debug', '-d', action='store_true')
    parser.add_argument('-enable_wandb', action='store_true')
    parser.add_argument('-log_wandb_ckpt', action='store_true')
    parser.add_argument('-log_eval', action='store_true')
    parser.add_argument('--dataset_dir', type=str, default='D:/FYP-001/MS2dataset', help='Root directory of the MS2 dataset')

    args = parser.parse_args()
    opt = Logger.parse(args)
    opt = Logger.dict_to_nonedict(opt)

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    Logger.setup_logger(None, opt['path']['log'], 'train', level=logging.INFO, screen=True)
    Logger.setup_logger('val', opt['path']['log'], 'val', level=logging.INFO)
    logger = logging.getLogger('base')
    logger.info(Logger.dict2str(opt))
    tb_logger = SummaryWriter(log_dir=opt['path']['tb_logger'])

    if opt['enable_wandb']:
        wandb_logger = WandbLogger(opt)
        wandb.define_metric('validation/val_step')
        wandb.define_metric('epoch')
        wandb.define_metric("validation/*", step_metric="val_step")
        val_step = 0
    else:
        wandb_logger = None

    # Fix random seed for reproducibility
    random.seed(1234)
    np.random.seed(1234)
    torch.manual_seed(1234)
    torch.cuda.manual_seed_all(1234)

    # Assign device (GPU/CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if torch.cuda.is_available():
        logger.info(f"Using GPU: {torch.cuda.get_device_name(device)} (ID: {torch.cuda.current_device()})")
    else:
        logger.info("Using CPU")

    # Load datasets
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train' and args.phase != 'val':
            train_set = DataLoader_MS2(
                dataset_opt['dataroot'],
                data_split='train',
                data_format='MonoDepth',
                modality='thr',
                sampling_step=3,
                set_length=1,
                set_interval=1
            )
            train_loader = torch.utils.data.DataLoader(train_set, batch_size=1, shuffle=True)

        elif phase == 'val':
            val_set = DataLoader_MS2(
                dataset_opt['dataroot'],
                data_split='val',
                data_format='MonoDepth',
                modality='thr',
                sampling_step=3,
                set_length=1,
                set_interval=1
            )
            val_loader = torch.utils.data.DataLoader(val_set, batch_size=1, shuffle=False)

    logger.info('Initial Dataset Loaded')

    # Load Model to Device
    diffusion = Model.create_model(opt).to(device)
    logger.info('Model Created')

    current_step = diffusion.begin_step
    current_epoch = diffusion.begin_epoch
    n_iter = opt['train']['n_iter']

    diffusion.set_new_noise_schedule(opt['model']['beta_schedule'][opt['phase']], schedule_phase=opt['phase'])

    # Define Saving Path
    save_path = os.path.join(opt['path']['checkpoint'], 'final_model.pth')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Save the model immediately without further training
    logger.info(f"Manually stopping at iteration {current_step}...")
    torch.save(diffusion.state_dict(), save_path)  # Save model weights
    diffusion.save_network(current_epoch, current_step)  # Save network configuration
    logger.info(f"Model saved manually at {save_path}")

    logger.info("Training is stopped immediately. The model has been saved.")

    # Optionally, you can perform evaluation or other tasks after saving the model
    if args.phase == 'val':
        logger.info('Begin Model Evaluation...')
        avg_mae = 0.0
        avg_rmse = 0.0
        idx = 0

        for _, val_data in enumerate(val_loader):
            idx += 1
            diffusion.feed_data(val_data)
            diffusion.test(continuous=True)
            visuals = diffusion.get_current_visuals()

            pred_depth = visuals['SR'].squeeze().cpu().numpy()
            gt_depth = visuals['HR'].squeeze().cpu().numpy()
            
            mae = np.mean(np.abs(pred_depth - gt_depth))
            rmse = np.sqrt(np.mean((pred_depth - gt_depth) ** 2))
            avg_mae += mae
            avg_rmse += rmse

        avg_mae /= idx
        avg_rmse /= idx
        logger.info(f'# Final Evaluation # MAE: {avg_mae:.4e}, RMSE: {avg_rmse:.4e}')
