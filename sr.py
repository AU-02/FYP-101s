import sys
import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
sys.path.append(os.path.abspath("D:/FYP-001"))
import torch
from torch.utils.data import DataLoader
from dataloader.MS2_dataset import DataLoader_MS2

import data as Data
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
    parser.add_argument('-c', '--config', type=str, default='config/sr_sr3_16_128.json',
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

    random.seed(1234)
    np.random.seed(1234)
    torch.manual_seed(1234)
    torch.cuda.manual_seed_all(1234)

    # ✅ Updated to use MS2 dataloader for thermal data
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
            train_loader = torch.utils.data(train_set, batch_size=1, shuffle=True)
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
            val_loader = torch.utils.data(val_set, batch_size=1, shuffle=False)

    logger.info('Initial Dataset Finished')

    diffusion = Model.create_model(opt)
    logger.info('Initial Model Finished')

    current_step = diffusion.begin_step
    current_epoch = diffusion.begin_epoch
    n_iter = opt['train']['n_iter']

    diffusion.set_new_noise_schedule(opt['model']['beta_schedule'][opt['phase']], schedule_phase=opt['phase'])

    if opt['phase'] == 'train':
        while current_step < n_iter:
            current_epoch += 1
            for _, train_data in enumerate(train_loader):
                current_step += 1
                if current_step > n_iter:
                    break

                # ✅ Fix for thermal input and depth map
                input_img = train_data['tgt_image'].cuda()
                gt_depth = train_data['tgt_depth_gt'].cuda()

                diffusion.feed_data({'input': input_img, 'target': gt_depth})
                diffusion.optimize_parameters()

                if current_step % opt['train']['print_freq'] == 0:
                    logs = diffusion.get_current_log()
                    message = '<epoch:{:3d}, iter:{:8,d}> '.format(current_epoch, current_step)
                    for k, v in logs.items():
                        message += '{:s}: {:.4e} '.format(k, v)
                        tb_logger.add_scalar(k, v, current_step)
                    logger.info(message)

                    if wandb_logger:
                        wandb_logger.log_metrics(logs)

                # ✅ Fix for depth map evaluation
                if current_step % opt['train']['val_freq'] == 0:
                    avg_mae = 0.0
                    avg_rmse = 0.0
                    idx = 0
                    result_path = '{}/{}'.format(opt['path']['results'], current_epoch)
                    os.makedirs(result_path, exist_ok=True)

                    diffusion.set_new_noise_schedule(opt['model']['beta_schedule']['val'], schedule_phase='val')
                    for _, val_data in enumerate(val_loader):
                        idx += 1
                        diffusion.feed_data(val_data)
                        diffusion.test(continous=False)
                        visuals = diffusion.get_current_visuals()

                        pred_depth = visuals['SR'].squeeze().cpu().numpy()
                        gt_depth = visuals['HR'].squeeze().cpu().numpy()

                        # ✅ Save as depth maps
                        np.save(f'{result_path}/{current_step}_{idx}_depth_pred.npy', pred_depth)
                        np.save(f'{result_path}/{current_step}_{idx}_depth_gt.npy', gt_depth)

                        mae = np.mean(np.abs(pred_depth - gt_depth))
                        rmse = np.sqrt(np.mean((pred_depth - gt_depth) ** 2))

                        avg_mae += mae
                        avg_rmse += rmse

                    avg_mae /= idx
                    avg_rmse /= idx

                    logger.info('# Validation # MAE: {:.4e}, RMSE: {:.4e}'.format(avg_mae, avg_rmse))

                    if wandb_logger:
                        wandb_logger.log_metrics({
                            'validation/mae': avg_mae,
                            'validation/rmse': avg_rmse,
                            'validation/val_step': val_step
                        })
                        val_step += 1

                if current_step % opt['train']['save_checkpoint_freq'] == 0:
                    logger.info('Saving models and training states.')
                    diffusion.save_network(current_epoch, current_step)

            if wandb_logger:
                wandb_logger.log_metrics({'epoch': current_epoch-1})

        logger.info('End of training.')

    else:
        logger.info('Begin Model Evaluation.')
        # ✅ Evaluation phase preserved for consistency
        avg_mae = 0.0
        avg_rmse = 0.0
        idx = 0
        result_path = '{}'.format(opt['path']['results'])
        os.makedirs(result_path, exist_ok=True)
        for _, val_data in enumerate(val_loader):
            idx += 1
            diffusion.feed_data(val_data)
            diffusion.test(continous=True)
            visuals = diffusion.get_current_visuals()
            pred_depth = visuals['SR'].squeeze().cpu().numpy()
            gt_depth = visuals['HR'].squeeze().cpu().numpy()
            mae = np.mean(np.abs(pred_depth - gt_depth))
            rmse = np.sqrt(np.mean((pred_depth - gt_depth) ** 2))
            avg_mae += mae
            avg_rmse += rmse

        avg_mae /= idx
        avg_rmse /= idx
        logger.info('# Final Evaluation # MAE: {:.4e}, RMSE: {:.4e}'.format(avg_mae, avg_rmse))
