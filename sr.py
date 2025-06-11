import sys
import os
import time
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
sys.path.insert(0, os.path.abspath("D:/FYP-001"))

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
import numpy as np
import wandb
import random
import torchvision.transforms.functional as TF
import json  # for saving effective config

# ✅ Define local image saving function
def save_tensor_image(tensor, path):
    # Check if the tensor is empty
    if tensor.numel() == 0:
        raise ValueError(f"Tensor is empty. Cannot save image. Tensor shape: {tensor.shape}")
    
    # Ensure the tensor is on CPU
    tensor = tensor.detach().cpu()
    
    # Debug: Print the original tensor shape
    print("Original tensor shape:", tensor.shape)
    
    # If tensor is 4D but in channel-last format (e.g., [N, H, W, C]), convert to channel-first
    if tensor.ndimension() == 4 and tensor.shape[-1] == 1:
        print("Detected channel-last format in a 4D tensor. Permuting to channel-first...")
        tensor = tensor.permute(0, 3, 1, 2)
        print("New tensor shape after permute:", tensor.shape)
    
    # Handle 2D (H, W) by unsqueezing to (1, H, W)
    if tensor.ndimension() == 2:
        tensor = tensor.unsqueeze(0)
    
    # Handle 4D (N, C, H, W) -> take first image
    if tensor.ndimension() == 4:
        tensor = tensor[0]
    
    # Now tensor should be 3D (C, H, W)
    if tensor.ndimension() != 3:
        raise ValueError(f"Unexpected tensor shape: {tensor.shape}. Expected 3D tensor.")
    
    # Debug: Print shape before potential transpose
    print("Shape before any transpose:", tensor.shape)
    
    # Optionally, if you need to check for swapped dimensions, add a conditional transpose here.
    # For example:
    C, H, W = tensor.shape
    # if H > W:
    #     print(f"Detected H > W (H={H}, W={W}). Transposing spatial dimensions.")
    #     tensor = tensor.transpose(-2, -1)
    #     print("New tensor shape after transpose:", tensor.shape)
    
    # If >3 channels, select first channel only (grayscale)
    if tensor.shape[0] > 3:
        tensor = tensor[:1]
    
    # If grayscale (1-channel), expand to RGB for saving
    if tensor.shape[0] == 1:
        tensor = tensor.expand(3, -1, -1)
    
    # Normalize to [0, 1] range
    tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min() + 1e-5)
    
    # Convert to PIL image
    from torchvision.transforms import functional as TF
    img = TF.to_pil_image(tensor)
    
    # Ensure the directory exists
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    # Save the image
    img.save(path)
    print(f"Saved image to {path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument('-c', '--config', type=str, default='D:/FYP-300-NIR/FYP-101s/config/shadow.json',
    #                     help='JSON file for configuration')
    parser.add_argument('-c', '--config', type=str, default='D:/FYP-101s/config/shadow.json',
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
    # ✅ FORCE overwrite to match old checkpoint model
    opt['model']['unet']['inner_channel'] = 64
    opt['model']['unet']['channel_multiplier'] = [1, 2, 4, 8]
    opt['path']['resume_state'] = "D:/FYP-300-NIR/experiments/depth_estimation_sr3_250426_222341/experiments/depth_estimation_sr3_250424_102919/checkpoints/I106000_E29"




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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train' and args.phase != 'val':
            train_set = DataLoader_MS2(
                dataset_opt['dataroot'],
                data_split='train',
                data_format='MonoDepth',
                modality='nir',
                sampling_step=3,
                set_length=1,
                set_interval=1
            )
            train_loader = torch.utils.data.DataLoader(
                train_set,
                batch_size=dataset_opt['batch_size'],
                shuffle=dataset_opt['use_shuffle'],
                num_workers=dataset_opt.get('num_workers', 0),
                pin_memory=True
            )

        elif phase == 'val':
            val_set = DataLoader_MS2(
                dataset_opt['dataroot'],
                data_split='val',
                data_format='MonoDepth',
                modality='nir',
                sampling_step=3,
                set_length=1,
                set_interval=1
            )
            val_loader = torch.utils.data.DataLoader(
                val_set,
                batch_size=1,  # or dataset_opt.get('batch_size', 1) if you later want to configure this too
                shuffle=False,
                num_workers=dataset_opt.get('num_workers', 0),
                pin_memory=True
            )


    logger.info('Initial Dataset Loaded')
    print(opt['model']['unet'])
    diffusion = Model.create_model(opt).to(device)
    logger.info('Model Created')

    current_step = diffusion.begin_step
    current_epoch = diffusion.begin_epoch
    num_epochs = opt['train']['num_epochs']

    diffusion.set_new_noise_schedule(opt['model']['beta_schedule'][opt['phase']], schedule_phase=opt['phase'])

    logger.info("Start Training Loop...")
    start_time = time.time()

    for epoch in range(current_epoch, num_epochs):
        current_epoch = epoch
        logger.info(f"Epoch [{epoch + 1}/{num_epochs}]")
        for _, train_data in enumerate(train_loader):
            current_step += 1

            diffusion.feed_data(train_data)
            diffusion.optimize_parameters()

            if current_step % opt['train']['print_freq'] == 0:
                visuals = diffusion.get_current_visuals()
                logger.info(f"Predicted shape: {visuals['Predicted'].shape}")  # Log the predicted shape

                tb_logger.add_scalar('loss/l_pix', diffusion.log_dict['l_pix'], current_step)
                logger.info(f"[Epoch {epoch + 1}/{num_epochs}] [Step {current_step}] l_pix: {diffusion.log_dict['l_pix']:.6f}")


                save_tensor_image(visuals['Input'], f"{opt['path']['results']}/input_{current_step}.png")
                save_tensor_image(visuals['GT'], f"{opt['path']['results']}/gt_{current_step}.png")
                pred_to_save = visuals['Predicted']
                save_tensor_image(pred_to_save, f"{opt['path']['results']}/pred_{current_step}.png")

            if current_step % opt['train']['val_freq'] == 0:
                logger.info(f"Validation at step {current_step}... (Epoch {epoch + 1})")
                avg_mae, avg_rmse = 0.0, 0.0
                val_step = 0
                for val_data in val_loader:
                    diffusion.feed_data(val_data)
                    diffusion.test(continuous=True)
                    visuals = diffusion.get_current_visuals()
                    pred = visuals['Predicted'].squeeze().cpu().numpy()
                    gt = visuals['GT'].squeeze().cpu().numpy()
                    print(f"📏 pred shape: {pred.shape}, gt shape: {gt.shape}")
                    val_mae = np.mean(np.abs(pred - gt))
                    val_rmse = np.sqrt(np.mean((pred - gt) ** 2))
                    avg_mae += val_mae
                    avg_rmse += val_rmse
                    val_step += 1
                avg_mae /= val_step
                avg_rmse /= val_step
                logger.info(f"VAL --> MAE: {avg_mae:.4f}, RMSE: {avg_rmse:.4f}")
                tb_logger.add_scalar("val/mae", avg_mae, current_step)
                tb_logger.add_scalar("val/rmse", avg_rmse, current_step)

            if current_step % opt['train']['save_checkpoint_freq'] == 0:
                diffusion.save_network(current_epoch, current_step)
                
                # Save the effective config at this checkpoint
                effective_config_path = os.path.join(opt['path']['checkpoint'], f'effective_opt_{current_step}.json')
                with open(effective_config_path, 'w') as f:
                    json.dump(opt, f, indent=4)
                logger.info("Saved effective configuration to %s", effective_config_path)

    save_path = os.path.join(opt['path']['checkpoint'], 'final_model.pth')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(diffusion.state_dict(), save_path)
    diffusion.save_network(current_epoch, current_step)
    logger.info(f"Model saved at final step {current_step} to {save_path}")

    elapsed = time.time() - start_time
    logger.info(f"Finished Training in {elapsed / 60:.2f} minutes.")

    # Save the effective configuration to a JSON file after training
    effective_config_path = os.path.join(opt['path']['checkpoint'], 'effective_opt.json')
    with open(effective_config_path, 'w') as f:
        json.dump(opt, f, indent=4)
    logger.info("Saved effective configuration to %s", effective_config_path)

    # Save the full model for future loading without rebuilding the architecture
    full_model_path = os.path.join(opt['path']['checkpoint'], 'full_model.pth')
    torch.save(diffusion, full_model_path)
    logger.info("Saved full model to %s", full_model_path)

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
            pred_depth = visuals['Predicted'].squeeze().cpu().numpy()
            gt_depth = visuals['GT'].squeeze().cpu().numpy()
            mae = np.mean(np.abs(pred_depth - gt_depth))
            rmse = np.sqrt(np.mean((pred_depth - gt_depth) ** 2))
            avg_mae += mae
            avg_rmse += rmse
        avg_mae /= idx
        avg_rmse /= idx
        logger.info(f'# Final Evaluation # MAE: {avg_mae:.4e}, RMSE: {avg_rmse:.4e}')