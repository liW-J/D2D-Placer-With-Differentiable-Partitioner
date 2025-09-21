'''
Author: JeanneWillis hi@jeannewillis.cn
Date: 2025-03-19 11:47:31
LastEditors: JeanneWillis hi@jeannewillis.cn
LastEditTime: 2025-09-20 22:51:51
FilePath: /D2D-placer/placer/ops/partition/partition.py
Description: partition flattened 2D placement to 2 Die
'''
from torch.autograd import Function
import torch

import logging

logger = logging.getLogger(__name__)

import ops.terminal_aux.terminal_aux_cpp as terminal_aux_cpp


class TerminalAuxFunction(Function):

    @staticmethod
    def forward(tier, flat_netpin, netpin_start, pin2node_map, net_weights,
                num_movable_nodes, node_size_x, node_size_y, pin_offset_x,
                pin_offset_y, die_size_x, die_size_y, row_height,
                terminal_size_x, terminal_size_y, terminal_spacing, pin_pos,
                node_names, net_names, pos_2d, case_name,
                terminal_legalize_flag, pos_terminal_legalized, num_terminals,
                terminal_names):
        func = terminal_aux_cpp.terminal_aux
        output = func(tier, flat_netpin, netpin_start, pin2node_map,
                      net_weights, num_movable_nodes, node_size_x, node_size_y,
                      pin_offset_x, pin_offset_y, die_size_x, die_size_y,
                      row_height, terminal_size_x, terminal_size_y,
                      terminal_spacing, pin_pos, node_names, net_names, pos_2d,
                      case_name, terminal_legalize_flag,
                      pos_terminal_legalized, num_terminals, terminal_names)

        return output


class TerminalAux(object):

    def __init__(self, flat_netpin, netpin_start, pin2node_map, net_weights,
                 num_movable_nodes, node_names, net_names, node_size_x,
                 node_size_y, pin_offset_x, pin_offset_y, die_size_x,
                 die_size_y, row_height, terminal_size_x, terminal_size_y,
                 terminal_spacing, case_name, terminal_legalize_flag):
        super(TerminalAux, self).__init__()

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
        self.terminal_legalize_flag = terminal_legalize_flag

    def __call__(self,
                 tier,
                 pin_pos=torch.empty(0),
                 pos_2d=torch.empty(0),
                 pos_terminal_legalized=torch.empty(0),
                 num_terminals=0,
                 terminal_names=[]):
        return TerminalAuxFunction.forward(
            tier,
            self.flat_netpin,
            self.netpin_start,
            self.pin2node_map,
            self.net_weights,
            self.num_movable_nodes,
            self.node_size_x,
            self.node_size_y,
            self.pin_offset_x,
            self.pin_offset_y,
            self.die_size_x,
            self.die_size_y,
            self.row_height,
            self.terminal_size_x,
            self.terminal_size_y,
            self.terminal_spacing,
            pin_pos,
            self.node_names,
            self.net_names,
            pos_2d,
            self.case_name,
            self.terminal_legalize_flag,
            pos_terminal_legalized,
            num_terminals,
            terminal_names,
        )


if __name__ == "__main__":
    pass
