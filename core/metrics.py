import os
import math
import numpy as np
import cv2
from torchvision.utils import make_grid

# ✅ Updated to handle single-channel depth maps
def tensor2img(tensor, out_type=np.uint8, min_max=(0, 1)):
    '''
    Converts a torch Tensor into an image Numpy array
    Input: 3D(C,H,W) or 2D(H,W), single channel depth maps
    Output: 2D(H,W), [0,255], np.uint8 (default)
    '''
    tensor = tensor.squeeze().float().cpu().clamp_(*min_max)
    tensor = (tensor - min_max[0]) / (min_max[1] - min_max[0])

    if tensor.dim() == 2:
        img_np = tensor.numpy()
    else:
        raise TypeError(
            'Only support 2D tensor for depth maps. Received: {:d} dimensions.'.format(tensor.dim()))

    if out_type == np.uint8:
        img_np = (img_np * 255.0).round()
    return img_np.astype(out_type)

# ✅ Updated to save depth maps as .png for visualization
def save_img(img, img_path, mode='GRAY'):
    cv2.imwrite(img_path, img)

# ✅ Added Mean Absolute Error (MAE) for depth estimation
def calculate_mae(depth_pred, depth_gt):
    '''
    Calculate Mean Absolute Error (MAE) between predicted and ground truth depth maps
    '''
    mae = np.mean(np.abs(depth_pred - depth_gt))
    return mae

# ✅ Added Root Mean Squared Error (RMSE) for depth estimation
def calculate_rmse(depth_pred, depth_gt):
    '''
    Calculate Root Mean Squared Error (RMSE) between predicted and ground truth depth maps
    '''
    rmse = np.sqrt(np.mean((depth_pred - depth_gt) ** 2))
    return rmse

# ❌ Removed PSNR and SSIM since they are not relevant for depth estimation
# def calculate_psnr(img1, img2):
#     pass

# def calculate_ssim(img1, img2):
#     pass
