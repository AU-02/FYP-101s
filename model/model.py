import logging
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

import os
import model.networks as networks
from .base_model import BaseModel
logger = logging.getLogger('base')

class EMAHelper(object):
    def __init__(self, mu=0.9999):
        self.mu = mu
        self.shadow = {}

    def register(self, module):
        if isinstance(module, nn.DataParallel):
            module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
                print(f"✅ Registering EMA shadow for {name}")
                self.shadow[name] = param.data.clone()

    def update(self, module):
        if isinstance(module, nn.DataParallel):
            module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
                if name not in self.shadow:
                    print(f"✅ Initializing EMA shadow for {name}")
                    self.shadow[name] = param.data.clone()

                # ✅ Handle shape mismatches dynamically
                if self.shadow[name].shape != param.data.shape:
                    print(f"⚠️ Shape mismatch in EMA update for {name}: {self.shadow[name].shape} -> {param.data.shape}")
                    self.shadow[name] = param.data.clone()

                # ✅ EMA Update
                self.shadow[name].data = (1. - self.mu) * param.data + self.mu * self.shadow[name].data

    def ema(self, module):
        if isinstance(module, nn.DataParallel):
            module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
                if name in self.shadow:
                    param.data.copy_(self.shadow[name].data)

    def ema_copy(self, module):
        if isinstance(module, nn.DataParallel):
            inner_module = module.module
            module_copy = type(inner_module)(inner_module.config).to(inner_module.config.device)
            module_copy.load_state_dict(inner_module.state_dict())
            module_copy = nn.DataParallel(module_copy)
        else:
            module_copy = type(module)(module.config).to(module.config.device)
            module_copy.load_state_dict(module.state_dict())
        self.ema(module_copy)
        return module_copy

    def state_dict(self):
        return self.shadow

    def load_state_dict(self, state_dict):
        self.shadow = state_dict

class DDPM(BaseModel, nn.Module):
    def __init__(self, opt):
        nn.Module.__init__(self)
        super(DDPM, self).__init__(opt)
        self.netG = self.set_device(networks.define_G(opt))
        self.schedule_phase = None
        self.ema_helper = EMAHelper()
        self.ema_helper.register(self.netG)
        self.set_loss()
        self.set_new_noise_schedule(opt['model']['beta_schedule']['train'], schedule_phase='train')
        if self.opt['phase'] == 'train':
            self.netG.train()
            optim_params = list(self.netG.parameters())
            self.optG = torch.optim.Adam(optim_params, lr=opt['train']["optimizer"]["lr"])
            self.log_dict = OrderedDict()
        self.load_network()
        self.print_network()
        
    # Inside your model's forward method
    def forward(self, x_in):
        # Print initial input shapes
        print(f"Initial input shapes: {x_in['HR'].shape}, {x_in['SR'].shape}")

        # Assuming your model uses some layers, for example, self.netG
        # If self.netG is a part of the DDPM model, print its details
        print(f"Initial input to netG: {x_in['HR'].shape}")
        
        # Process through layers
        x = self.netG(x_in)  # Process through your model's generator or network
        print(f"After netG: {x.shape}")

        # If you have any intermediate layers or transformations, check them
        # For example, if there's a convolution layer
        # x = self.conv_layer(x)
        # print(f"After conv_layer: {x.shape}")

        # Print the final output before any transformation
        final_output = x  # Replace this with your final layer output
        print(f"Final output before any post-processing: {final_output.shape}")
        
        # Check if the output is a valid tensor
        if final_output.numel() == 0:
            print("Error: Final output tensor is empty!")
        else:
            print(f"Final output shape: {final_output.shape}")
        
        return final_output

    # ✅ Updated to handle thermal + depth data
    def feed_data(self, data):
        self.data = self.set_device(data)
        logger.info(f"self.data: {self.data}")

        # Reassign keys so that thermal image is HR and depth ground truth is SR.
        if 'tgt_image' in data and 'tgt_depth_gt' in data:
            self.data['input'] = {
                'HR': self.data['tgt_image'],      # Thermal image goes to HR
                'SR': self.data['tgt_depth_gt']      # Depth ground truth goes to SR
            }
        elif 'HR' in data and 'SR' in data:
            self.data['input'] = {
                'HR': self.data['HR'],
                'SR': self.data['SR']
            }
        else:
            raise KeyError(f"Invalid data keys in self.data: {self.data.keys()}")

        print("After feed_data, HR shape:", self.data['input']['HR'].shape)
        print("After feed_data, SR shape:", self.data['input']['SR'].shape)

    # ✅ Updated to reflect depth estimation
    def optimize_parameters(self):
        self.optG.zero_grad()

        logger.info(f"self.data['input'] type: {type(self.data['input'])}")

        # Wrap the tensor in a dictionary if needed.
        if isinstance(self.data['input'], torch.Tensor):
            logger.info("Converting Tensor to dictionary format")
            self.data['input'] = {
                'HR': self.data['input'],
                'SR': self.data['input']
            }

        # Pass the input dictionary to the model.
        predicted_loss = self.netG(self.data['input'])
        
        # Retrieve predicted depth and processed HR from the underlying module if needed.
        if hasattr(self.netG, "module"):
            self.output = self.netG.module.predicted_depth
            processed_HR = self.netG.module.processed_HR
        else:
            self.output = self.predicted_depth
            processed_HR = self.processed_HR

        # Log the shape of the predicted depth tensor.
        logger.info(f"Predicted depth shape: {self.output.shape}")
        
        # Log predicted depth statistics.
        logger.info("Predicted depth - min: %s", self.output.min().item())
        logger.info("Predicted depth - max: %s", self.output.max().item())
        logger.info("Predicted depth - mean: %s", self.output.mean().item())

        # Compute the pixel loss (MAE) between predicted depth and processed HR.
        l_pix = torch.mean(torch.abs(self.output - processed_HR))

        predicted_loss.backward()
        self.optG.step()
        self.ema_helper.update(self.netG)
        self.log_dict['l_pix'] = l_pix.item()

    # ✅ Updated for depth output
    def test(self, continuous=False):
        self.netG.eval()

        with torch.no_grad():
            # ✅ Handle missing keys more gracefully
            input_data = self.data.get('input', {})
            if 'HR' not in input_data or 'SR' not in input_data:
                raise KeyError(f"Missing keys in self.data['input']: {input_data.keys()}")

            x_in = {
                'HR': input_data['HR'],
                'SR': input_data['SR']
            }

            # ✅ Ensure SR and HR are 4D tensors (N, C, H, W)
            if x_in['SR'].dim() == 3:
                print(f"Unsqueezing SR from {x_in['SR'].shape}")
                x_in['SR'] = x_in['SR'].unsqueeze(1)

            if x_in['HR'].dim() == 3:
                print(f"Unsqueezing HR from {x_in['HR'].shape}")
                x_in['HR'] = x_in['HR'].unsqueeze(1)

            # ✅ Fix channel mismatch
            if x_in['HR'].shape[1] != x_in['SR'].shape[1]:
                print(f"Fixing channel mismatch: HR {x_in['HR'].shape[1]} -> SR {x_in['SR'].shape[1]}")
                x_in['HR'] = x_in['HR'].repeat(1, x_in['SR'].shape[1] // x_in['HR'].shape[1], 1, 1)

            # ✅ Fix spatial mismatch
            if x_in['HR'].shape[-2:] != x_in['SR'].shape[-2:]:
                print(f"Resizing SR from {x_in['SR'].shape[-2:]} to {x_in['HR'].shape[-2:]}")
                x_in['SR'] = F.interpolate(
                    x_in['SR'],
                    size=x_in['HR'].shape[-2:],
                    mode='bilinear',
                    align_corners=False
                )

            # ✅ Pass through the model
            if isinstance(self.netG, nn.DataParallel):
                self.output = self.netG.module(x_in)
            else:
                self.output = self.netG(x_in)

            # ✅ Make sure output is valid before resizing
            if self.output is not None and self.output.dim() > 0 and self.output.numel() > 0:
                if self.output.shape != x_in['HR'].shape:
                    print(f"Resizing output from {self.output.shape} to {x_in['HR'].shape}")
                    self.output = F.interpolate(
                        self.output,
                        size=x_in['HR'].shape[-2:],
                        mode='bilinear',
                        align_corners=False
                    )
            else:
                print(f"⚠️ Model output is empty or invalid — output.shape = {self.output.shape if self.output is not None else 'None'}")
            

        self.netG.train()

    # ✅ Updated for depth sample generation
    def sample(self, batch_size=1, continous=False):
        self.netG.eval()
        with torch.no_grad():
            # Ensure the image size matches the expected size for the model (e.g., 256)
            image_size = 256
            # Generate random noise of the correct shape, matching the smallest resolution expected in the network
            random_noise = torch.randn((batch_size, 64, 4, 256)).to(self.device)  # Match smallest resolution here (4, 256)

            # Call the p_sample_loop with the generated random noise
            if isinstance(self.netG, nn.DataParallel):
                self.output = self.netG.module.p_sample_loop(random_noise, continous)
            else:
                self.output = self.netG.p_sample_loop(random_noise, continous)

        self.netG.train()  # Return model to training mode after inference

    def set_loss(self):
        if isinstance(self.netG, nn.DataParallel):
            self.netG.module.set_loss(self.device)
        else:
            self.netG.set_loss(self.device)

    def set_new_noise_schedule(self, schedule_opt, schedule_phase='train'):
        if self.schedule_phase is None or self.schedule_phase != schedule_phase:
            self.schedule_phase = schedule_phase
            if isinstance(self.netG, nn.DataParallel):
                self.netG.module.set_new_noise_schedule(schedule_opt, self.device)
            else:
                self.netG.set_new_noise_schedule(schedule_opt, self.device)

    # ✅ Updated for depth-based output
    def get_current_visuals(self, need_LR=True, sample=False):
        out_dict = OrderedDict()

        if sample:
            out_dict['SAM'] = self.output.detach().float().cpu()
        else:
            # Predicted output remains as is.
            if self.output is None:
                logger.warning("⚠️ 'Predicted' output is None!")
                out_dict['Predicted'] = torch.zeros_like(self.data['input']['HR']).detach().cpu()
            else:
                out_dict['Predicted'] = self.output.detach().float().cpu()

            # Ground truth (GT) should now come from the depth ground truth, which is in SR.
            if 'SR' in self.data['input']:
                gt = self.data['input']['SR'].detach().float().cpu()
                # Remove or comment out the following block if it’s causing the issue:
                # if gt.ndimension() == 4:
                #     _, _, H, W = gt.shape
                #     if H > W:
                #         print("Detected GT tensor with H > W. Transposing GT tensor dimensions.")
                #         gt = gt.transpose(-2, -1)  # swap height and width
                out_dict['GT'] = gt
            else:
                logger.warning("⚠️ 'SR' (ground truth) is missing!")
                out_dict['GT'] = torch.zeros_like(self.data['input']['HR']).detach().cpu()

            # For the input image, use HR (the thermal image)
            if 'HR' in self.data['input']:
                out_dict['Input'] = self.data['input']['HR'].detach().float().cpu()
            else:
                logger.warning("⚠️ 'HR' (input) is missing!")
                out_dict['Input'] = torch.zeros_like(self.data['input']['SR']).detach().cpu()

            # Optionally, you can still keep a key for the original SR if needed.
            out_dict['SR'] = self.data['input'].get('SR', torch.zeros_like(self.data['input']['HR'])).detach().float().cpu()

        return out_dict


    def print_network(self):
        s, n = self.get_network_description(self.netG)
        if isinstance(self.netG, nn.DataParallel):
            net_struc_str = '{} - {}'.format(self.netG.__class__.__name__, self.netG.module.__class__.__name__)
        else:
            net_struc_str = '{}'.format(self.netG.__class__.__name__)
        logger.info('Network G structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
        logger.info(s)

    # ✅ Updated checkpoint names for depth estimation
    def save_network(self, epoch, iter_step):
        gen_path = os.path.join(self.opt['path']['checkpoint'], 'I{}_E{}_depth_gen.pth'.format(iter_step, epoch))
        opt_path = os.path.join(self.opt['path']['checkpoint'], 'I{}_E{}_depth_opt.pth'.format(iter_step, epoch))
        network = self.netG
        if isinstance(self.netG, nn.DataParallel):
            network = network.module
        state_dict = network.state_dict()
        for key, param in state_dict.items():
            state_dict[key] = param.cpu()
        torch.save(state_dict, gen_path)
        opt_state = {'epoch': epoch, 'iter': iter_step, 'scheduler': None, 'optimizer': None}
        opt_state['optimizer'] = self.optG.state_dict()
        opt_state['ema_helper'] = self.ema_helper.state_dict()
        torch.save(opt_state, opt_path)

    def load_network(self):
        load_path = self.opt['path']['resume_state']
        if load_path is not None:
            logger.info('Loading pretrained model for G [{:s}] ...'.format(load_path))
            gen_path = '{}_depth_gen.pth'.format(load_path)
            opt_path = '{}_depth_opt.pth'.format(load_path)
            network = self.netG
            if isinstance(self.netG, nn.DataParallel):
                network = network.module
            network.load_state_dict(torch.load(gen_path), strict=False)
            if self.opt['phase'] == 'train':
                opt = torch.load(opt_path)
                self.optG.load_state_dict(opt['optimizer'])
                self.begin_step = opt['iter']
                self.begin_epoch = opt['epoch']
                self.ema_helper.load_state_dict(opt['ema_helper'])