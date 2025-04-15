'''
Author: JeanneWillis hi@jeannewillis.cn
Date: 2025-03-19 11:47:31
LastEditors: JeanneWillis hi@jeannewillis.cn
LastEditTime: 2025-04-15 02:18:27
FilePath: /D2D-placer/placer/ops/partition/partition.py
Description: partition flattened 2D placement to 2 Die
'''
from torch.autograd import Function
import torch

import logging

logger = logging.getLogger(__name__)

import ops.partition.partition_cpp as partition_cpp


class PartitionFunction(Function):

    @staticmethod
    def forward(tier, flat_netpin, netpin_start, pin2node_map, net_weights,
                num_movable_nodes, node_size_x, node_size_y, pin_offset_x,
                pin_offset_y, die_size_x, die_size_y, row_height, pos,
                terminal_instert_flag):
        func = partition_cpp.partition
        output = func(tier, flat_netpin, netpin_start, pin2node_map,
                      net_weights, num_movable_nodes, node_size_x, node_size_y,
                      pin_offset_x, pin_offset_y, die_size_x, die_size_y,
                      row_height, pos, terminal_instert_flag)

        return output


class Partition(object):

    def __init__(self, flat_netpin, netpin_start, pin2node_map, net_weights,
                 num_movable_nodes, terminal_instert_flag):
        super(Partition, self).__init__()

        self.flat_netpin = flat_netpin
        self.netpin_start = netpin_start
        self.pin2node_map = pin2node_map
        self.net_weights = net_weights
        self.num_movable_nodes = num_movable_nodes
        self.terminal_instert_flag = terminal_instert_flag

    def __call__(self,
                 tier,
                 node_size_x,
                 node_size_y,
                 pin_offset_x,
                 pin_offset_y,
                 die_size_x,
                 die_size_y,
                 row_height,
                 pos=torch.empty(0)):
        return PartitionFunction.forward(
            tier, self.flat_netpin, self.netpin_start, self.pin2node_map,
            self.net_weights, self.num_movable_nodes, node_size_x, node_size_y,
            pin_offset_x, pin_offset_y, die_size_x, die_size_y, row_height,
            pos, self.terminal_instert_flag)


if __name__ == "__main__":
    pass
