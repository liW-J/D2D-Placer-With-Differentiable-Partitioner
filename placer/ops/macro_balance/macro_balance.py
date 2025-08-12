'''
Author: JeanneWillis hi@jeannewillis.cn
Date: 2025-07-21 01:12:38
LastEditors: JeanneWillis hi@jeannewillis.cn
LastEditTime: 2025-07-27 22:51:12
FilePath: /D2D-placer/placer/ops/macro_balance/macro_balance.py
Description: 
'''
from torch.autograd import Function
import torch

import logging

logger = logging.getLogger(__name__)

import ops.macro_balance.macro_balance_cpp as macro_balance_cpp


class MacroBalanceFunction(Function):

    @staticmethod
    def forward(tier, flat_netpin, netpin_start, pin2node_map, net_weights,
                num_movable_nodes, node_size_x, node_size_y, pin_offset_x,
                pin_offset_y, die_size_x, die_size_y, row_height,
                terminal_size_x, terminal_size_y, terminal_spacing, pin_pos,
                pos_terminal_legalized, num_terminals, node_names, net_names,
                terminal_names, pos_2d, case_name, node_orient,
                movable_macro_mask):
        func = macro_balance_cpp.macro_balance
        output = func(tier, flat_netpin, netpin_start, pin2node_map,
                      net_weights, num_movable_nodes, node_size_x, node_size_y,
                      pin_offset_x, pin_offset_y, die_size_x, die_size_y,
                      row_height, terminal_size_x, terminal_size_y,
                      terminal_spacing, pin_pos, pos_terminal_legalized,
                      num_terminals, node_names, net_names, terminal_names,
                      pos_2d, case_name, node_orient, movable_macro_mask)

        return output


class MacroBalance(object):

    def __init__(self, flat_netpin, netpin_start, pin2node_map, net_weights,
                 num_movable_nodes, node_names, net_names, node_size_x,
                 node_size_y, pin_offset_x, pin_offset_y, die_size_x,
                 die_size_y, row_height, terminal_size_x, terminal_size_y,
                 terminal_spacing, case_name):
        super(MacroBalance, self).__init__()

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

    def __call__(self,
                 tier,
                 node_orient,
                 movable_macro_mask,
                 pin_pos=torch.empty(0),
                 pos_terminal_legalized=torch.empty(0),
                 num_terminals=0,
                 pos_2d=torch.empty(0),
                 terminal_names=[]):
        return MacroBalanceFunction.forward(
            tier, self.flat_netpin, self.netpin_start, self.pin2node_map,
            self.net_weights, self.num_movable_nodes, self.node_size_x,
            self.node_size_y, self.pin_offset_x, self.pin_offset_y,
            self.die_size_x, self.die_size_y, self.row_height,
            self.terminal_size_x, self.terminal_size_y, self.terminal_spacing,
            pin_pos, pos_terminal_legalized, num_terminals, self.node_names,
            self.net_names, terminal_names, pos_2d, self.case_name,
            node_orient, movable_macro_mask)


if __name__ == "__main__":
    pass
