'''
Author: JeanneWillis hi@jeannewillis.cn
Date: 2025-09-21 17:01:58
LastEditors: JeanneWillis hi@jeannewillis.cn
LastEditTime: 2025-10-14 23:42:41
FilePath: /D2D-placer/placer/ops/bin_based_fm/bin_based_fm.py
Description: fm refinement
'''

from torch.autograd import Function
import torch

import logging

logger = logging.getLogger(__name__)

import placer.ops.bin_based_fm.bin_based_fm_cpp as bin_based_fm_cpp


class BinBasedFMFunction(Function):

    @staticmethod
    def forward(tier, flat_netpin, netpin_start, pin2node_map, net_weights,
                num_movable_nodes, node_size_x, node_size_y, pin_offset_x,
                pin_offset_y, die_size_x, die_size_y, row_height,
                terminal_size_x, terminal_size_y, terminal_spacing, pin_pos,
                node_names, net_names, pos_2d, pos_terminal_legalized,
                case_name, num_terminals, terminal_names, top_die_max_util,
                bottom_die_max_util, num_nodes, num_bins_x, num_bins_y, xl, yl,
                xh, yh):
        func = bin_based_fm_cpp.bin_based_fm

        # node_size_x = node_size_x.to(dtype=torch.double).contiguous()
        # node_size_y = node_size_y.to(dtype=torch.double).contiguous()
        # pos_2d = pos_2d.to(dtype=torch.double).contiguous()

        output = func(tier, flat_netpin, netpin_start, pin2node_map,
                      net_weights, num_movable_nodes, node_size_x, node_size_y,
                      pin_offset_x, pin_offset_y, die_size_x, die_size_y,
                      row_height, terminal_size_x, terminal_size_y,
                      terminal_spacing, pin_pos, node_names, net_names, pos_2d,
                      pos_terminal_legalized, case_name, num_terminals,
                      terminal_names, top_die_max_util, bottom_die_max_util,
                      num_nodes, num_bins_x, num_bins_y, xl, yl, xh, yh)

        return output


class BinBasedFM(object):

    def __init__(self, flat_netpin, netpin_start, pin2node_map, net_weights,
                 num_movable_nodes, node_names, net_names, node_size_x,
                 node_size_y, pin_offset_x, pin_offset_y, die_size_x,
                 die_size_y, row_height, terminal_size_x, terminal_size_y,
                 terminal_spacing, case_name, top_die_max_util,
                 bottom_die_max_util, num_nodes, num_bins_x, num_bins_y, xl,
                 yl, xh, yh):
        super(BinBasedFM, self).__init__()

        self.flat_netpin = flat_netpin
        self.netpin_start = netpin_start
        self.pin2node_map = pin2node_map
        self.net_weights = net_weights
        self.num_movable_nodes = num_movable_nodes

        self.node_names = node_names
        self.net_names = net_names

        self.node_size_x = node_size_x
        self.node_size_y = node_size_y
        self.pin_offset_x = pin_offset_x
        self.pin_offset_y = pin_offset_y
        self.die_size_x = die_size_x
        self.die_size_y = die_size_y
        self.row_height = row_height
        self.terminal_size_x = terminal_size_x
        self.terminal_size_y = terminal_size_y
        self.terminal_spacing = terminal_spacing

        self.case_name = case_name

        self.top_die_max_util = top_die_max_util
        self.bottom_die_max_util = bottom_die_max_util
        self.num_nodes = num_nodes
        self.num_bins_x = num_bins_x
        self.num_bins_y = num_bins_y
        self.xl = xl
        self.yl = yl
        self.xh = xh
        self.yh = yh

    def __call__(self, tier, pin_pos, pos_2d, pos_terminal_legalized,
                 num_terminals, terminal_names):
        return BinBasedFMFunction.forward(
            tier, self.flat_netpin, self.netpin_start, self.pin2node_map,
            self.net_weights, self.num_movable_nodes, self.node_size_x,
            self.node_size_y, self.pin_offset_x, self.pin_offset_y,
            self.die_size_x, self.die_size_y, self.row_height,
            self.terminal_size_x, self.terminal_size_y, self.terminal_spacing,
            pin_pos, self.node_names, self.net_names, pos_2d,
            pos_terminal_legalized, self.case_name, num_terminals,
            terminal_names, self.top_die_max_util, self.bottom_die_max_util,
            self.num_nodes, self.num_bins_x, self.num_bins_y, self.xl, self.yl,
            self.xh, self.yh)


if __name__ == "__main__":
    pass
