import torch
import data as Data
import model as Model
import argparse
import logging
import core.logger as Logger
import core.metrics as Metrics
from core.wandb_logger import WandbLogger
from tensorboardX import SummaryWriter
import os
import numpy as np
import wandb
from MS2_dataset import DataLoader_MS2  # ✅ Changed to MS2 dataset loader

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config/sample_sr3_128.json',
                        help='JSON file for configuration')
    parser.add_argument('-p', '--phase', type=str, choices=['train', 'val'],
                        help='Run either train(training) or val(generation)', default='train')
    parser.add_argument('-gpu', '--gpu_ids', type=str, default=None)
    parser.add_argument('-debug', '-d', action='store_true')
    parser.add_argument('-enable_wandb', action='store_true')
    parser.add_argument('-log_wandb_ckpt', action='store_true')

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

    # ✅ Initialize WandbLogger
    if opt['enable_wandb']:
        wandb_logger = WandbLogger(opt)
        val_step = 0
    else:
        wandb_logger = None

    # ✅ Updated to use MS2 dataset for thermal + depth data
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
    logger.info('Initial Dataset Finished')

    # ✅ Initialize model
    diffusion = Model.create_model(opt)
    logger.info('Initial Model Finished')

    current_step = diffusion.begin_step
    current_epoch = diffusion.begin_epoch
    n_iter = opt['train']['n_iter']
    sample_sum = opt['datasets']['val']['data_len']

    if opt['path']['resume_state']:
        logger.info('Resuming training from epoch: {}, iter: {}.'.format(current_epoch, current_step))

    diffusion.set_new_noise_schedule(opt['model']['beta_schedule'][opt['phase']], schedule_phase=opt['phase'])

    if opt['phase'] == 'train':
        while current_step < n_iter:
            current_epoch += 1
            for _, train_data in enumerate(train_loader):
                current_step += 1
                if current_step > n_iter:
                    break

                # ✅ Feed thermal input and ground truth depth
                input_img = train_data['tgt_image'].cuda()
                gt_depth = train_data['tgt_depth_gt'].cuda()

                diffusion.feed_data({'input': input_img, 'target': gt_depth})
                diffusion.optimize_parameters()

                # ✅ Logging
                if current_step % opt['train']['print_freq'] == 0:
                    logs = diffusion.get_current_log()
                    message = '<epoch:{:3d}, iter:{:8,d}> '.format(current_epoch, current_step)
                    for k, v in logs.items():
                        message += '{:s}: {:.4e} '.format(k, v)
                        tb_logger.add_scalar(k, v, current_step)
                    logger.info(message)

                    if wandb_logger:
                        wandb_logger.log_metrics(logs)

                # ✅ Validation
                if current_step % opt['train']['val_freq'] == 0:
                    result_path = '{}/{}'.format(opt['path']['results'], current_epoch)
                    os.makedirs(result_path, exist_ok=True)

                    diffusion.set_new_noise_schedule(opt['model']['beta_schedule']['val'], schedule_phase='val')

                    for idx in range(sample_sum):
                        diffusion.sample(continous=False)
                        visuals = diffusion.get_current_visuals(sample=True)

                        # ✅ Save as .npy instead of .png
                        pred_depth = visuals['SR'].squeeze().cpu().numpy()
                        np.save(f'{result_path}/{current_step}_{idx}_depth_pred.npy', pred_depth)

                        tb_logger.add_image(
                            'Iter_{}'.format(current_step),
                            np.expand_dims(pred_depth, axis=0),
                            idx
                        )

                        if wandb_logger:
                            wandb_logger.log_image(f'validation_{idx}', pred_depth)

                    diffusion.set_new_noise_schedule(opt['model']['beta_schedule']['train'], schedule_phase='train')

                # ✅ Save Model
                if current_step % opt['train']['save_checkpoint_freq'] == 0:
                    logger.info('Saving models and training states.')
                    diffusion.save_network(current_epoch, current_step)

                    if wandb_logger and opt['log_wandb_ckpt']:
                        wandb_logger.log_checkpoint(current_epoch, current_step)

        logger.info('End of training.')

    else:
        logger.info('Begin Model Evaluation.')

        result_path = '{}'.format(opt['path']['results'])
        os.makedirs(result_path, exist_ok=True)
        sample_depth_maps = []

        for idx in range(sample_sum):
            idx += 1
            diffusion.sample(continous=True)
            visuals = diffusion.get_current_visuals(sample=True)

            # ✅ Handle depth maps directly
            pred_depth = visuals['SR'].squeeze().cpu().numpy()
            np.save(f'{result_path}/{current_step}_{idx}_depth_pred.npy', pred_depth)

            sample_depth_maps.append(pred_depth)

        if wandb_logger:
            wandb_logger.log_images('eval_depth_maps', sample_depth_maps)
