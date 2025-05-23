'''
Author: JeanneWillis hi@jeannewillis.cn
Date: 2025-03-19 11:47:31
LastEditors: JeanneWillis hi@jeannewillis.cn
LastEditTime: 2025-05-20 20:01:55
FilePath: /D2D-placer/placer/ops/partition/partition.py
Description: partition flattened 2D placement to 2 Die
'''
from torch.autograd import Function
import torch

import logging

logger = logging.getLogger(__name__)

import ops.partition_aux.partition_aux_cpp as partition_aux_cpp


class PartitionAuxFunction(Function):

    @staticmethod
    def forward(tier, flat_netpin, netpin_start, pin2node_map, net_weights,
                num_movable_nodes, node_size_x, node_size_y, pin_offset_x,
                pin_offset_y, die_size_x, die_size_y, row_height, pos,
                terminal_instert_flag, terminal_legalize_flag,
                pos_tier_legalized_terminal, num_movable_nodes_top, node_names,
                net_names, pos_2d):
        func = partition_aux_cpp.partition_aux
        output = func(tier, flat_netpin, netpin_start, pin2node_map,
                      net_weights, num_movable_nodes, node_size_x, node_size_y,
                      pin_offset_x, pin_offset_y, die_size_x, die_size_y,
                      row_height, pos, terminal_instert_flag,
                      terminal_legalize_flag, pos_tier_legalized_terminal,
                      num_movable_nodes_top, node_names, net_names, pos_2d)

        return output


class PartitionAux(object):

    def __init__(self, flat_netpin, netpin_start, pin2node_map, net_weights,
                 num_movable_nodes, node_names, net_names,
                 terminal_instert_flag, terminal_legalize_flag):
        super(PartitionAux, self).__init__()

        self.flat_netpin = flat_netpin
        self.netpin_start = netpin_start
        self.pin2node_map = pin2node_map
        self.net_weights = net_weights
        self.num_movable_nodes = num_movable_nodes
        self.terminal_instert_flag = terminal_instert_flag
        self.terminal_legalize_flag = terminal_legalize_flag

        self.node_names = node_names
        self.net_names = net_names

    def __call__(self,
                 tier,
                 node_size_x,
                 node_size_y,
                 pin_offset_x,
                 pin_offset_y,
                 die_size_x,
                 die_size_y,
                 row_height,
                 pin_pos=torch.empty(0),
                 pos_tier_legalized_terminal=torch.empty(0),
                 num_movable_nodes_top=0,
                 pos_2d=torch.empty(0)):
        return PartitionAuxFunction.forward(
            tier, self.flat_netpin, self.netpin_start, self.pin2node_map,
            self.net_weights, self.num_movable_nodes, node_size_x, node_size_y,
            pin_offset_x, pin_offset_y, die_size_x, die_size_y, row_height,
            pin_pos, self.terminal_instert_flag, self.terminal_legalize_flag,
            pos_tier_legalized_terminal, num_movable_nodes_top, self.node_names,
            self.net_names, pos_2d)


if __name__ == "__main__":
    pass
