import os
import torch
import torch.nn as nn

class BaseModel():
    def __init__(self, opt):
        self.opt = opt
        self.device = torch.device(
            'cuda' if opt['gpu_ids'] is not None else 'cpu')
        self.begin_step = 0
        self.begin_epoch = 0

    # ✅ Updated to handle thermal + depth data
    def feed_data(self, data):
        """
        Feed data into the model
        Expected input: {'input': thermal_input, 'target': depth_gt}
        """
        self.input = data['input'].to(self.device)
        self.target = data['target'].to(self.device)

    def optimize_parameters(self):
        pass

    # ✅ Updated to return predicted depth and ground truth
    def get_current_visuals(self):
        visuals = {
            'input': self.input.detach().cpu(),
            'predicted': self.output.detach().cpu(),
            'ground_truth': self.target.detach().cpu()
        }
        return visuals

    def get_current_losses(self):
        pass

    def print_network(self):
        pass

    def set_device(self, x):
        if isinstance(x, dict):
            for key, item in x.items():
                if item is not None:
                    x[key] = item.to(self.device)
        elif isinstance(x, list):
            for item in x:
                if item is not None:
                    item = item.to(self.device)
        else:
            x = x.to(self.device)
        return x

    def get_network_description(self, network):
        '''Get the string and total parameters of the network'''
        if isinstance(network, nn.DataParallel):
            network = network.module
        s = str(network)
        n = sum(map(lambda x: x.numel(), network.parameters()))
        return s, n
