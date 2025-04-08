'''
Author: JeanneWillis hi@jeannewillis.cn
Date: 2025-03-19 11:47:31
LastEditors: JeanneWillis hi@jeannewillis.cn
LastEditTime: 2025-04-08 21:18:33
FilePath: /D2D-placer/placer/ops/partition/partition.py
Description: partition flattened 2D placement to 2 Die
'''
import torch
from torch.autograd import Function
from torch import nn

import logging

logger = logging.getLogger(__name__)

import ops.hmetis_partition.hmetis_partition_cpp as hmetis_partition_cpp


class HmetisPartitionFunction(Function):

    @staticmethod
    def forward(flat_netpin, netpin_start, pin2node_map, net_weights, net_mask):
        func = hmetis_partition_cpp.hmetis_partition
        output = func(flat_netpin, netpin_start, pin2node_map, net_weights,
                      net_mask)
        # breakpoint()
        return output


class HmetisPartition(nn.Module):

    def __init__(self,
                 flat_netpin=None,
                 netpin_start=None,
                 pin2node_map=None,
                 net_weights=None,
                 net_mask=None):

        super(HmetisPartition, self).__init__()
        assert net_mask is not None, "net_mask is a requried parameter"

        self.flat_netpin = flat_netpin
        self.netpin_start = netpin_start
        self.pin2node_map = pin2node_map
        self.net_weights = net_weights
        self.net_mask = net_mask

    def __call__(self):
        return HmetisPartitionFunction.forward(self.flat_netpin,
                                               self.netpin_start,
                                               self.pin2node_map,
                                               self.net_weights, self.net_mask)


if __name__ == "__main__":
    pass
