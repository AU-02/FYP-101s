import logging
from collections import OrderedDict

import torch
import torch.nn as nn
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
                self.shadow[name] = param.data.clone()

    def update(self, module):
        if isinstance(module, nn.DataParallel):
            module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
                self.shadow[name].data = (1. - self.mu) * param.data + self.mu * self.shadow[name].data

    def ema(self, module):
        if isinstance(module, nn.DataParallel):
            module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
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

class DDPM(BaseModel):
    def __init__(self, opt):
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

    # ✅ Updated to handle thermal + depth data
    def feed_data(self, data):
        self.data = self.set_device(data)

    # ✅ Updated to reflect depth estimation
    def optimize_parameters(self):
        self.optG.zero_grad()
        predicted_depth = self.netG(self.data['input'])
        l_pix = torch.mean(torch.abs(predicted_depth - self.data['target']))  # MAE for depth
        l_pix.backward()
        self.optG.step()
        self.ema_helper.update(self.netG)
        self.log_dict['l_pix'] = l_pix.item()

    # ✅ Updated for depth output
    def test(self, continous=False):
        self.netG.eval()
        with torch.no_grad():
            if isinstance(self.netG, nn.DataParallel):
                self.output = self.netG.module.predict(self.data['input'], continous)
            else:
                self.output = self.netG.predict(self.data['input'], continous)
        self.netG.train()

    # ✅ Updated for depth sample generation
    def sample(self, batch_size=1, continous=False):
        self.netG.eval()
        with torch.no_grad():
            if isinstance(self.netG, nn.DataParallel):
                self.output = self.netG.module.sample(batch_size, continous)
            else:
                self.output = self.netG.sample(batch_size, continous)
        self.netG.train()

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
            out_dict['Predicted'] = self.output.detach().float().cpu()
            out_dict['Target'] = self.data['target'].detach().float().cpu()
            out_dict['Input'] = self.data['input'].detach().float().cpu()
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
