import argparse
import core.metrics as Metrics
import numpy as np
import glob

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path', type=str,
                        default='experiments/basic_sr_ffhq_210809_142238/results')
    args = parser.parse_args()

    # ✅ Load .npy depth maps instead of .png images
    real_names = list(glob.glob('{}/*_depth_gt.npy'.format(args.path)))
    fake_names = list(glob.glob('{}/*_depth_pred.npy'.format(args.path)))

    real_names.sort()
    fake_names.sort()

    avg_mae = 0.0
    avg_rmse = 0.0
    idx = 0

    for rname, fname in zip(real_names, fake_names):
        idx += 1
        ridx = rname.rsplit("_depth_gt")[0]
        fidx = fname.rsplit("_depth_pred")[0]

        assert ridx == fidx, 'Mismatch: GT index:{ridx} != Prediction index:{fidx}'.format(
            ridx=ridx, fidx=fidx)

        # ✅ Load depth maps from .npy
        gt_depth = np.load(rname)
        pred_depth = np.load(fname)

        # ✅ Use MAE and RMSE instead of PSNR/SSIM
        mae = np.mean(np.abs(pred_depth - gt_depth))
        rmse = np.sqrt(np.mean((pred_depth - gt_depth) ** 2))

        avg_mae += mae
        avg_rmse += rmse

        if idx % 20 == 0:
            print('Sample:{}, MAE:{:.4f}, RMSE:{:.4f}'.format(idx, mae, rmse))

    avg_mae = avg_mae / idx
    avg_rmse = avg_rmse / idx

    # ✅ Updated logging for depth evaluation
    print('# Validation # MAE: {:.4e}'.format(avg_mae))
    print('# Validation # RMSE: {:.4e}'.format(avg_rmse))
