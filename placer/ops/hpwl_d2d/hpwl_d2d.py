'''
Author: JeanneWillis hi@jeannewillis.cn
Date: 2025-06-17 13:33:09
LastEditors: JeanneWillis hi@jeannewillis.cn
LastEditTime: 2025-07-20 23:39:14
FilePath: /D2D-placer/install/hpwl_d2d/hpwl_d2d.py
Description:
'''
import torch
from torch.autograd import Function
from torch import nn
import numpy as np

import placer.ops.hpwl_d2d.hpwl_d2d_cpp as hpwl_d2d_cpp


class HPWLD2DFunction(Function):

    @staticmethod
    def forward(pin_pos, flat_netpin, netpin_start, pin2node_map, net_weights,
                cut_net_mask, tier, num_tiers, pos_terminal_legalized,
                terminal_size_x, terminal_size_y, terminal_spacing,
                num_terminals, net_names, terminal_names):

        func = hpwl_d2d_cpp.hpwl_d2d
        # hpwl_d2d_cpp is a CPU-only operator; move tensors off GPU if needed
        output = func(pin_pos.cpu().view(pin_pos.numel()),
                      flat_netpin.cpu(), netpin_start.cpu(),
                      pin2node_map.cpu(), net_weights.cpu(),
                      cut_net_mask.cpu(), tier.cpu(), num_tiers,
                      pos_terminal_legalized.cpu(),
                      terminal_size_x, terminal_size_y, terminal_spacing,
                      num_terminals, net_names, terminal_names)
        return output

class HPWLD2D(object):
    """ 
    @brief Compute half-perimeter wirelength. 
    Support two algoriths: net-by-net and atomic. 
    Different parameters are required for different algorithms. 
    """

    def __init__(self, flat_netpin, netpin_start, pin2node_map, net_weights,
                 terminal_size_x, terminal_size_y, terminal_spacing, net_names,
                 num_tiers):

        super(HPWLD2D, self).__init__()

        self.flat_netpin = flat_netpin
        self.netpin_start = netpin_start
        self.pin2node_map = pin2node_map
        self.net_weights = net_weights
        self.terminal_size_x = terminal_size_x
        self.terminal_size_y = terminal_size_y
        self.terminal_spacing = terminal_spacing
        self.net_names = net_names
        self.num_tiers = num_tiers

    def __call__(self,
                 pin_pos,
                 cut_net_mask,
                 tier,
                 pos_terminal_legalized=torch.empty(0),
                 num_terminals=0,
                 terminal_names=np.array([], dtype=np.bytes_)):

        return HPWLD2DFunction.forward(
            pin_pos, self.flat_netpin, self.netpin_start, self.pin2node_map,
            self.net_weights, cut_net_mask, tier, self.num_tiers,
            pos_terminal_legalized, self.terminal_size_x, self.terminal_size_y,
            self.terminal_spacing, num_terminals, self.net_names,
            terminal_names)


if __name__ == "__main__":
    pass
