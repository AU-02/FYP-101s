import os
import torch
from PIL import Image
import numpy as np
import torchvision.transforms.functional as TF
from skimage.color import rgb2gray
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
import matplotlib.cm as cm  # For applying colormap

def load_thermal_image(image_path, target_size=(640, 256)):
    print(f"Loading thermal image from: {image_path}")
    img = Image.open(image_path).convert("L")
    img = img.resize(target_size, Image.BILINEAR)
    tensor = TF.to_tensor(img)  # shape: [1, H, W]
    tensor = tensor.unsqueeze(0)  # -> [1, 1, H, W]
    print(f"Thermal image tensor shape: {tensor.shape}")
    return tensor

def save_tensor_image(tensor, save_path):
    if tensor.numel() == 0:
        raise ValueError(f"Tensor is empty. Cannot save image. Tensor shape: {tensor.shape}")
    tensor = tensor.detach().cpu()
    if tensor.ndimension() == 4:
        tensor = tensor[0]
    if tensor.ndimension() == 2:
        tensor = tensor.unsqueeze(0)
    if tensor.shape[0] == 1:
        tensor = tensor.expand(3, -1, -1)
    tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min() + 1e-5)
    img = TF.to_pil_image(tensor)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    img.save(save_path)
    print(f"Saved image to {save_path}")

def fill_depth_colorization(imgRgb, imgDepth, alpha=1):
    # This function is your provided KITTI depth post-processing.
    if len(imgRgb.shape) == 2:
        imgRgb = np.stack([imgRgb] * 3, axis=-1)
    if imgRgb.shape[-1] == 1:
        imgRgb = np.repeat(imgRgb, 3, axis=-1)
    
    grayImg = rgb2gray(imgRgb)
    if len(grayImg.shape) != 2:
        raise ValueError(f"grayImg has an unexpected shape: {grayImg.shape}")
    
    imgIsNoise = (imgDepth == 0) | (imgDepth == 10)
    maxImgAbsDepth = np.max(imgDepth[~imgIsNoise])
    imgDepth = imgDepth / 10000.0
    imgDepth[imgDepth > 1] = 1
    H, W = imgDepth.shape
    numPix = H * W
    indsM = np.arange(numPix).reshape(H, W)
    knownValMask = ~imgIsNoise
    winRad = 1
    
    rows, cols, vals = [], [], []
    for j in range(W):
        for i in range(H):
            absImgNdx = indsM[i, j]
            gvals = []
            for ii in range(max(0, i - winRad), min(i + winRad + 1, H)):
                for jj in range(max(0, j - winRad), min(j + winRad + 1, W)):
                    if ii == i and jj == j:
                        continue
                    rows.append(absImgNdx)
                    cols.append(indsM[ii, jj])
                    gvals.append(grayImg[ii, jj])
            curVal = grayImg[i, j]
            gvals.append(curVal)
            c_var = np.mean((np.array(gvals) - np.mean(gvals)) ** 2)
            csig = c_var * 0.6
            mgv = np.min((np.array(gvals[:-1]) - curVal) ** 2)
            if csig < (-mgv / np.log(0.01)):
                csig = -mgv / np.log(0.01)
            if csig < 0.000002:
                csig = 0.000002
            gvals[:-1] = np.exp(-((np.array(gvals[:-1]) - curVal) ** 2) / csig)
            if np.sum(gvals[:-1]) != 0:
                gvals[:-1] /= np.sum(gvals[:-1])
            vals.extend(-np.array(gvals[:-1]))
            rows.append(absImgNdx)
            cols.append(absImgNdx)
            vals.append(1)
    
    A = csr_matrix((vals, (rows, cols)), shape=(numPix, numPix))
    G = csr_matrix((knownValMask.ravel() * alpha, (np.arange(numPix), np.arange(numPix))), shape=(numPix, numPix))
    new_vals = spsolve(A + G, (knownValMask.ravel() * alpha * imgDepth.ravel()))
    denoisedDepthImg = new_vals.reshape(H, W) * maxImgAbsDepth
    return denoisedDepthImg

def main():
    full_model_path = "experiments/depth_estimation_sr3_250407_120336/checkpoints/full_model.pth"
    input_image_path = "D:/FYP-200_git/FYP-101s/4000_1_depth_gt.png"
    output_raw_path = "D:/FYP-200_git/FYP-101s/generated_depth_raw.png"
    output_final_path = "D:/FYP-200_git/FYP-101s/generated_depth.png"
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    print(f"Loading full model from: {full_model_path}")
    loaded_obj = torch.load(full_model_path, map_location=device)
    print("Loaded model type:", type(loaded_obj))
    if isinstance(loaded_obj, dict):
        print("Error: The loaded checkpoint is a state_dict, not the full model!")
        return
    model = loaded_obj
    model.eval()
    model.to(device)
    print("Full model loaded successfully.")
    
    thermal_tensor = load_thermal_image(input_image_path, target_size=(640, 256))
    thermal_tensor = thermal_tensor.to(device)
    
    input_dict = {"HR": thermal_tensor, "SR": thermal_tensor}
    
    print("Feeding data to model...")
    model.feed_data(input_dict)
    print("Running test() method...")
    model.test(continuous=True)
    
    # Try to retrieve predicted depth from candidate attributes.
    predicted_depth = None
    if hasattr(model, "netG"):
        if hasattr(model.netG, "module") and hasattr(model.netG.module, "predicted_depth"):
            predicted_depth = model.netG.module.predicted_depth
            print("Retrieved predicted depth from model.netG.module.predicted_depth:", predicted_depth.shape)
        elif hasattr(model, "predicted_depth"):
            predicted_depth = model.predicted_depth
            print("Retrieved predicted depth from model.predicted_depth:", predicted_depth.shape)
    
    if predicted_depth is None or predicted_depth.numel() == 0:
        visuals = model.get_current_visuals()
        print("Visual keys from get_current_visuals():", list(visuals.keys()))
        if "Predicted" in visuals and visuals["Predicted"].ndimension() >= 2:
            predicted_depth = visuals["Predicted"]
            if predicted_depth.shape[1] == 2:
                predicted_depth = predicted_depth[:, 0:1, :, :]
            print("Retrieved predicted depth from visuals:", predicted_depth.shape)
        else:
            print("Error: Could not retrieve predicted output.")
            return

    # Explicitly extract single channel: if shape is [1, 2, 256, 640], then take channel 0.
    if predicted_depth.ndimension() == 4 and predicted_depth.shape[1] == 2:
        depth_tensor = predicted_depth[:, 0, :, :].unsqueeze(0)  # shape: [1, 1, 256, 640]
    else:
        depth_tensor = predicted_depth.squeeze()
    print("Final predicted depth tensor shape (after extraction):", depth_tensor.shape)
    
    # Save the raw predicted depth map.
    save_tensor_image(depth_tensor, output_raw_path)
    
    # Post-process using KITTI depth colorization.
    # Convert the raw depth tensor to a 2D NumPy array.
    depth_np = depth_tensor.squeeze().cpu().numpy()
    print("Raw predicted depth numpy shape:", depth_np.shape)
    
    # Load the corresponding RGB image (for guidance in colorization)
    rgb_img = np.array(Image.open(input_image_path).convert("RGB"))
    densified_depth = fill_depth_colorization(rgb_img, depth_np, alpha=1)
    
    # Normalize densified depth for visualization.
    densified_depth_norm = (densified_depth - densified_depth.min()) / (densified_depth.max() - densified_depth.min())
    
    # Apply Viridis colormap using matplotlib.
    colored = cm.viridis(densified_depth_norm)  # This returns an RGBA image in float [0, 1]
    colored = (colored[:, :, :3] * 255).astype(np.uint8)  # Drop alpha channel and convert to uint8
    colored_img = Image.fromarray(colored)
    colored_img.save(output_final_path)
    print(f"Post-processed (Viridis colored) depth map saved to {output_final_path}")

if __name__ == "__main__":
    main()
