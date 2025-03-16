import torch
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
from MS2_dataset import DataLoader_MS2  # ✅ Changed to MS2 dataset loader

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config/sr_sr3_64_512.json',
                        help='JSON file for configuration')
    parser.add_argument('-p', '--phase', type=str, choices=['val'], help='val(generation)', default='val')
    parser.add_argument('-gpu', '--gpu_ids', type=str, default=None)
    parser.add_argument('-debug', '-d', action='store_true')
    parser.add_argument('-enable_wandb', action='store_true')
    parser.add_argument('-log_infer', action='store_true')

    args = parser.parse_args()
    opt = Logger.parse(args)
    opt = Logger.dict_to_nonedict(opt)

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    Logger.setup_logger(None, opt['path']['log'],
                        'train', level=logging.INFO, screen=True)
    Logger.setup_logger('val', opt['path']['log'], 'val', level=logging.INFO)
    logger = logging.getLogger('base')
    logger.info(Logger.dict2str(opt))
    tb_logger = SummaryWriter(log_dir=opt['path']['tb_logger'])

    # ✅ Initialize WandbLogger
    if opt['enable_wandb']:
        wandb_logger = WandbLogger(opt)
    else:
        wandb_logger = None

    # ✅ Updated to use MS2 dataset for thermal + depth data
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'val':
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
    logger.info('Initial Dataset Finished')

    # ✅ Initialize model
    diffusion = Model.create_model(opt)
    logger.info('Initial Model Finished')

    diffusion.set_new_noise_schedule(
        opt['model']['beta_schedule']['val'], schedule_phase='val')

    logger.info('Begin Model Inference.')
    current_step = 0
    idx = 0

    result_path = '{}'.format(opt['path']['results'])
    os.makedirs(result_path, exist_ok=True)

    for _, val_data in enumerate(val_loader):
        idx += 1
        diffusion.feed_data(val_data)
        diffusion.test(continous=True)
        visuals = diffusion.get_current_visuals(need_LR=False)

        # ✅ Extract predicted and ground truth depth
        gt_depth = visuals['HR'].squeeze().cpu().numpy()
        pred_depth = visuals['SR'].squeeze().cpu().numpy()

        # ✅ Save depth maps as .npy files instead of .png
        np.save(f'{result_path}/{current_step}_{idx}_depth_pred.npy', pred_depth)
        np.save(f'{result_path}/{current_step}_{idx}_depth_gt.npy', gt_depth)

        # ✅ Optional: Log images using matplotlib (for visualization)
        if wandb_logger and opt['log_infer']:
            wandb_logger.log_image(f'Inference_{idx}', pred_depth)

    if wandb_logger and opt['log_infer']:
        wandb_logger.log_eval_table(commit=True)

    logger.info('Inference Completed.')
