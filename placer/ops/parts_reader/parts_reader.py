'''
Author: JeanneWillis hi@jeannewillis.cn
Date: 2025-03-19 11:47:31
LastEditors: JeanneWillis hi@jeannewillis.cn
LastEditTime: 2025-08-24 16:00:25
FilePath: /D2D-placer/placer/ops/partition/partition.py
Description: partition flattened 2D placement to 2 Die
'''
import torch
from torch.autograd import Function
from torch import nn

import logging

logger = logging.getLogger(__name__)

import ops.parts_reader.parts_reader_cpp as parts_reader_cpp


class PartsReaderFunction(Function):

    @staticmethod
    def forward(tier, flat_netpin, netpin_start, pin2node_map, net_weights,
                net_mask, num_movable_nodes, partitioner_path):
        func = parts_reader_cpp.parts_reader
        output = func(tier, flat_netpin, netpin_start, pin2node_map,
                      net_weights, net_mask, num_movable_nodes, partitioner_path)

        return output


class PartsReader(nn.Module):

    def __init__(self, flat_netpin, netpin_start, pin2node_map, net_weights,
                 net_mask, num_movable_nodes):

        super(PartsReader, self).__init__()
        assert net_mask is not None, "net_mask is a requried parameter"

        self.flat_netpin = flat_netpin
        self.netpin_start = netpin_start
        self.pin2node_map = pin2node_map
        self.net_weights = net_weights
        self.net_mask = net_mask
        self.num_movable_nodes = num_movable_nodes

    def __call__(self, tier, partitioner_path):
        return PartsReaderFunction.forward(tier, self.flat_netpin, self.netpin_start,
                                      self.pin2node_map, self.net_weights,
                                      self.net_mask, self.num_movable_nodes,
                                      partitioner_path)


if __name__ == "__main__":
    pass
