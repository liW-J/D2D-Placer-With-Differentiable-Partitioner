'''
Author: JeanneWillis hi@jeannewillis.cn
Date: 2025-03-19 11:47:31
LastEditors: JeanneWillis hi@jeannewillis.cn
LastEditTime: 2025-04-23 14:35:34
FilePath: /D2D-placer/placer/ops/partition/partition.py
Description: partition flattened 2D placement to 2 Die
'''
import torch
from torch.autograd import Function
from torch import nn

import logging

logger = logging.getLogger(__name__)

import ops.avg_cut.avg_cut_cpp as avg_cut_cpp


class AvgCutFunction(Function):

    @staticmethod
    def forward(flat_netpin, netpin_start, pin2node_map, net_weights,
                num_movable_nodes, pos):
        func = avg_cut_cpp.avg_cut
        output = func(flat_netpin, netpin_start, pin2node_map,
                      net_weights, num_movable_nodes, pos)
        return output


class AvgCut(nn.Module):

    def __init__(self, flat_netpin, netpin_start, pin2node_map, net_weights,
                 num_movable_nodes):

        super(AvgCut, self).__init__()

        self.flat_netpin = flat_netpin
        self.netpin_start = netpin_start
        self.pin2node_map = pin2node_map
        self.net_weights = net_weights
        self.num_movable_nodes = num_movable_nodes

    def __call__(self,
                 pos=torch.empty(0)):
        return AvgCutFunction.forward(self.flat_netpin, self.netpin_start,
                                      self.pin2node_map, self.net_weights,
                                      self.num_movable_nodes, pos)


if __name__ == "__main__":
    pass
