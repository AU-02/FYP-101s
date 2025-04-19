import math
import torch
from torch import device, nn, einsum
import torch.nn.functional as F
from inspect import isfunction
from functools import partial
import numpy as np
from tqdm import tqdm
import os
import utils
from PIL import Image
import torchvision.transforms as transforms



def _warmup_beta(linear_start, linear_end, n_timestep, warmup_frac):
    betas = linear_end * np.ones(n_timestep, dtype=np.float64)
    warmup_time = int(n_timestep * warmup_frac)
    betas[:warmup_time] = np.linspace(
        linear_start, linear_end, warmup_time, dtype=np.float64)
    return betas


def make_beta_schedule(schedule, n_timestep, linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
    if schedule == 'quad':
        betas = np.linspace(linear_start ** 0.5, linear_end ** 0.5,
                            n_timestep, dtype=np.float64) ** 2
    elif schedule == 'linear':
        betas = np.linspace(linear_start, linear_end,
                            n_timestep, dtype=np.float64)
    elif schedule == 'warmup10':
        betas = _warmup_beta(linear_start, linear_end,
                             n_timestep, 0.1)
    elif schedule == 'warmup50':
        betas = _warmup_beta(linear_start, linear_end,
                             n_timestep, 0.5)
    elif schedule == 'const':
        betas = linear_end * np.ones(n_timestep, dtype=np.float64)
    elif schedule == 'jsd':  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1. / np.linspace(n_timestep,
                                 1, n_timestep, dtype=np.float64)
    elif schedule == "cosine":
        timesteps = (
            torch.arange(n_timestep + 1, dtype=torch.float64) /
            n_timestep + cosine_s
        )
        alphas = timesteps / (1 + cosine_s) * math.pi / 2
        alphas = torch.cos(alphas).pow(2)
        alphas = alphas / alphas[0]
        betas = 1 - alphas[1:] / alphas[:-1]
        betas = betas.clamp(max=0.999)
    else:
        raise NotImplementedError(schedule)
    return betas


# gaussian diffusion trainer class

def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        denoise_fn,
        image_size,
        channels=3,
        loss_type='l1',
        conditional=True,
        schedule_opt=None
    ):
        super().__init__()
        self.channels = 1
        self.image_size = image_size
        self.denoise_fn = denoise_fn
        self.loss_type = loss_type
        self.conditional = conditional

        if schedule_opt is not None:
            pass
            # self.set_new_noise_schedule(schedule_opt)

    def set_loss(self, device):
        # not l1 or l2
        if self.loss_type == 'mae':
            self.loss_func = nn.L1Loss(reduction='sum').to(device)
        elif self.loss_type == 'huber':
            self.loss_func = nn.SmoothL1Loss(reduction='sum').to(device)

        else:
            raise NotImplementedError()

    def set_new_noise_schedule(self, schedule_opt, device):
        to_torch = partial(torch.tensor, dtype=torch.float32, device=device)

        betas = make_beta_schedule(
            schedule=schedule_opt['schedule'],
            n_timestep=schedule_opt['n_timestep'],
            linear_start=schedule_opt['linear_start'],
            linear_end=schedule_opt['linear_end'])
        betas = betas.detach().cpu().numpy() if isinstance(
            betas, torch.Tensor) else betas
        # self.betas = betas
        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])
        self.sqrt_alphas_cumprod_prev = np.sqrt(
            np.append(1., alphas_cumprod))

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev',
                             to_torch(alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod',
                             to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod',
                             to_torch(np.sqrt(1. - alphas_cumprod)))
        self.register_buffer('log_one_minus_alphas_cumprod',
                             to_torch(np.log(1. - alphas_cumprod)))
        self.register_buffer('sqrt_recip_alphas_cumprod',
                             to_torch(np.sqrt(1. / alphas_cumprod)))
        self.register_buffer('sqrt_recipm1_alphas_cumprod',
                             to_torch(np.sqrt(1. / alphas_cumprod - 1)))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * \
            (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.register_buffer('posterior_variance',
                             to_torch(posterior_variance))
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped', to_torch(
            np.log(np.maximum(posterior_variance, 1e-20))))
        self.register_buffer('posterior_mean_coef1', to_torch(
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)))
        self.register_buffer('posterior_mean_coef2', to_torch(
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)))

    def compute_alpha(self, beta, t):
        beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
        a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1)
        return a

    def predict_start_from_noise(self, x_t, t, noise):
        return self.sqrt_recip_alphas_cumprod[t] * x_t - \
            self.sqrt_recipm1_alphas_cumprod[t] * noise

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = self.posterior_mean_coef1[t] * \
            x_start + self.posterior_mean_coef2[t] * x_t
        posterior_log_variance_clipped = self.posterior_log_variance_clipped[t]
        return posterior_mean, posterior_log_variance_clipped


        #TODO KO CONDITION_X :p
    def p_mean_variance(self, x, t, clip_denoised: bool, condition_x=None):
        batch_size = x.shape[0]
        noise_level = torch.FloatTensor(
            [self.sqrt_alphas_cumprod_prev[t+1]]).repeat(batch_size, 1).to(x.device)
        if condition_x is not None:
            x_recon = self.predict_start_from_noise(
                x, t=t, noise=self.denoise_fn(torch.cat([x], dim=1), noise_level)).squeeze(1)
        else:
            x_recon = self.predict_start_from_noise(
                x, t=t, noise=self.denoise_fn(torch.cat([x], dim=1), noise_level))


        if clip_denoised:
            x_recon.clamp_(0., 1.)# ???

        model_mean, posterior_log_variance = self.q_posterior(
            x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, x, t, clip_denoised=True, condition_x=None):
        model_mean, model_log_variance = self.p_mean_variance(
            x=x, t=t, clip_denoised=clip_denoised, condition_x=condition_x)
        noise = torch.randn_like(x) if t > 0 else torch.zeros_like(x)
        return model_mean + noise * (0.5 * model_log_variance).exp()

    @torch.no_grad()
    def p_sample_loop(self, x_lr, continous=False):
        device = self.betas.device
        n = x_lr.size(0)
        noise = torch.randn((n, self.channels, x_lr.size(2), x_lr.size(3))).to(device) # todo why this change ?
        skip = self.num_timesteps // 5
        # skip = self.num_timesteps // 25
        seq = range(0, self.num_timesteps, skip)
        x0_preds = []
        xs = [noise]
        #mask removed
        b = self.betas
        eta = 0.
        gamma_ori = 0.1
        idx = 0
        # sample_inter = (1 | (self.num_timesteps//10))
        seq_next = [-1] + list(seq[:-1])
        # mask = mask_0  REMOVE
        for i, j in zip(reversed(seq), reversed(seq_next)):
            t = (torch.ones(n) * i).to(device)
            next_t = (torch.ones(n) * j).to(device)
            at = self.compute_alpha(b, t.long())
            at_next = self.compute_alpha(b, next_t.long())
            xt = xs[-1].to('cuda')
            
            if i >= len(b)*0.2:
                et= self.denoise_fn(torch.cat([x_lr, xt], dim=1), t)
            else:
                et= self.denoise_fn(torch.cat([x_lr, xt], dim=1), t)
            x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()
            x0_preds.append(x0_t.to('cpu'))
            c1 = eta * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
            c2 = ((1 - at_next) - c1 ** 2).sqrt()
            xt_next = at_next.sqrt() * x0_t + c1 * torch.randn_like(x_lr) + c2 * et
            xs.append(xt_next.to('cpu'))
        ret_img = xs
        # if continous:
        #     return ret_img
        # else:
        return ret_img[-1].squeeze(1)
        # if not self.conditional:
        #     shape = x_in
        #     img = torch.randn(shape, device=device)
        #     ret_img = img
        #     for i in tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step', total=self.num_timesteps):
        #         img = self.p_sample(img, i)
        #         if i % sample_inter == 0:
        #             ret_img = torch.cat([ret_img, mask, img], dim=0)
        # else:
        #     x = x_in
        #     shape = x.shape
        #     img = torch.randn(shape, device=device)
        #     ret_img = x
        #     for i in tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step', total=self.num_timesteps):
        #         img = self.p_sample(img, mask, i, condition_x=x)
        #         if i % sample_inter == 0:
        #             ret_img = torch.cat([ret_img, mask, img], dim=0)
        # if continous:
        #     return ret_img
        # else:
        #     return ret_img[-1]

    @torch.no_grad()
    def sample(self, batch_size=1, continous=False):
        image_size = self.image_size
        #  todo KO CHANNELS = SELF.CHANNELS . THEORY 1 IS THE CHANELLS DUE TO THERMAL AS NOTES RECLERIFY 
        return self.p_sample_loop((batch_size, 1, image_size, image_size), continous)

    @torch.no_grad()
    def depth_estimation(self, x_lr, continous=False):
        return self.p_sample_loop(x_lr, continous)



    def q_sample(self, x_start, continuous_sqrt_alpha_cumprod, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        # random gama
        return (
            continuous_sqrt_alpha_cumprod * x_start +
            (1 - continuous_sqrt_alpha_cumprod**2).sqrt() * noise
        )

    def p_losses(self, x_in, noise=None):
        if isinstance(x_in, torch.Tensor):
            raise TypeError(f"Expected x_in to be a dictionary, but got {type(x_in)}")

        print(f"x_in keys: {x_in.keys()}")

        # Ensure input tensors exist and are valid
        if 'HR' not in x_in:
            raise KeyError("'HR' key is missing from x_in")
        if 'SR' not in x_in:
            raise KeyError("'SR' key is missing from x_in")

        # Handle empty tensors or None values
        if x_in['HR'] is None or x_in['HR'].numel() == 0:
            print("⚠️ x_in['HR'] is empty — replacing with zeros")
            x_in['HR'] = torch.zeros((1, 1, 256, 256)).to(self.betas.device)
        if x_in['SR'] is None or x_in['SR'].numel() == 0:
            print("⚠️ x_in['SR'] is empty — replacing with zeros")
            x_in['SR'] = torch.zeros((1, 1, 256, 256)).to(self.betas.device)

        # Convert from channels-last to channels-first if needed.
        if x_in['HR'].ndim == 4 and x_in['HR'].shape[-1] in [1, 2, 3]:
            print("Converting HR from channels-last to channels-first")
            x_in['HR'] = x_in['HR'].permute(0, 3, 1, 2)
        if x_in['SR'].ndim == 4 and x_in['SR'].shape[-1] in [1, 2, 3]:
            print("Converting SR from channels-last to channels-first")
            x_in['SR'] = x_in['SR'].permute(0, 3, 1, 2)

        # Ensure tensors are 4D -> (N, C, H, W)
        if x_in['HR'].ndim == 3:
            x_in['HR'] = x_in['HR'].unsqueeze(0)
        if x_in['SR'].ndim == 3:
            x_in['SR'] = x_in['SR'].unsqueeze(0)

        # Fix channel mismatches by repeating if needed.
        if x_in['HR'].shape[1] != 2:
            print(f"Fixing HR channel mismatch: {x_in['HR'].shape[1]} -> 2")
            x_in['HR'] = x_in['HR'].repeat(1, 2, 1, 1)
        if x_in['SR'].shape[1] != 2:
            print(f"Fixing SR channel mismatch: {x_in['SR'].shape[1]} -> 2")
            x_in['SR'] = x_in['SR'].repeat(1, 2, 1, 1)

        # Fix spatial dimensions: make SR match HR.
        if x_in['SR'].shape[-2:] != x_in['HR'].shape[-2:]:
            print(f"Resizing SR from {x_in['SR'].shape[-2:]} to {x_in['HR'].shape[-2:]}")
            x_in['SR'] = F.interpolate(
                x_in['SR'],
                size=x_in['HR'].shape[-2:],
                mode='bilinear',
                align_corners=False
            )

        print(f"x_in['HR'] shape: {x_in['HR'].shape}, type: {type(x_in['HR'])}")
        print(f"x_in['SR'] shape: {x_in['SR'].shape}, type: {type(x_in['SR'])}")

        # Store the processed HR for later use.
        self.processed_HR = x_in['HR']

        # Generate noisy version of HR for training
        x_start = x_in['SR']
        n, c, h, w = x_start.shape
        print(f"x_start shape: {x_start.shape}")
        t = torch.randint(low=0, high=self.num_timesteps, size=(n // 2 + 1,)).to(x_start.device)
        t = torch.cat([t, self.num_timesteps - t - 1], dim=0)[:n]
        print(f"t shape: {t.shape}")
        b = self.betas
        a = (1 - b).cumprod(dim=0).index_select(0, t).view(-1, 1, 1, 1)
        e = torch.randn_like(x_start)
        x_noisy = x_start * a.sqrt() + e * (1.0 - a).sqrt()
        print(f"x_noisy shape: {x_noisy.shape}")

        # Ensure x_noisy and SR have the same spatial dimensions
        if x_in['SR'].shape[-2:] != x_noisy.shape[-2:]:
            print(f"Resizing SR to match x_noisy: {x_in['SR'].shape[-2:]} -> {x_noisy.shape[-2:]}")
            x_in['SR'] = F.interpolate(
                x_in['SR'],
                size=x_noisy.shape[-2:],
                mode='bilinear',
                align_corners=False
            )
        if x_in['SR'].shape[1] != x_noisy.shape[1]:
            x_in['SR'] = x_in['SR'].repeat(1, x_noisy.shape[1] // x_in['SR'].shape[1], 1, 1)
        print(f"x_in['SR'] shape after matching x_noisy: {x_in['SR'].shape}")

        print(f"Final SR shape before concat: {x_in['SR'].shape}")
        print(f"Final noisy shape before concat: {x_noisy.shape}")
        x_cat = torch.cat([x_in['HR'], x_noisy], dim=1)
        print(f"x_cat shape (after concat): {x_cat.shape}")

        x_recon = self.denoise_fn(x_cat, t.float())
        print(f"x_recon shape after denoise_fn: {x_recon.shape}")

        # Resize x_recon to match HR spatial dimensions exactly.
        target_size = x_in['HR'].shape[-2:]  # (H, W)
        print(f"Target size for interpolation: {target_size}")
        if x_recon.numel() == 0:
            raise ValueError("x_recon is empty before interpolation!")
        x_recon = F.interpolate(x_recon, size=target_size, mode='bilinear', align_corners=False)
        print(f"x_recon shape after interpolation: {x_recon.shape}")

        if x_recon.shape[1] != x_in['HR'].shape[1]:
            print(f"Channel mismatch: x_recon channels {x_recon.shape[1]} vs HR channels {x_in['HR'].shape[1]}")
            x_recon = x_recon.repeat(1, x_in['HR'].shape[1] // x_recon.shape[1], 1, 1)
            print(f"x_recon shape after channel repeat: {x_recon.shape}")

        # Store predicted depth for logging purposes.
        self.predicted_depth = x_recon

        loss = self.loss_func(e, x_recon) / (n * h * w)
        print(f"Final loss: {loss.item()}")
        print(f"Final predicted x_recon shape (before returning loss): {x_recon.shape}")

        res = (x_in['HR'] + 1) / 2 - (x_in['SR'] + 1) / 2
        res = torch.mean(res, dim=1, keepdim=True)
        res_map = torch.where(res < 0.05, torch.zeros_like(res), torch.ones_like(res))
        
        print(f"Returning loss and storing predicted depth.")
        return loss



    def forward(self, x, *args, **kwargs):
        return self.p_losses(x, *args, **kwargs)

