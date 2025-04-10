'''
Author: JeanneWillis hi@jeannewillis.cn
Date: 2025-03-19 11:47:31
LastEditors: JeanneWillis hi@jeannewillis.cn
LastEditTime: 2025-04-10 16:04:47
FilePath: /D2D-placer/placer/ops/partition/partition.py
Description: partition flattened 2D placement to 2 Die
'''
from torch.autograd import Function

import logging

logger = logging.getLogger(__name__)

import ops.partition.partition_cpp as partition_cpp


class PartitionFunction(Function):

    @staticmethod
    def forward(tier, flat_netpin, netpin_start, pin2node_map, net_weights, net_mask,
                num_movable_nodes):
        func = partition_cpp.partition
        output = func(tier, flat_netpin, netpin_start, pin2node_map, net_weights, net_mask,
                      num_movable_nodes)

        return output


class Partition(object):

    def __init__(self, flat_netpin, netpin_start, pin2node_map, net_weights,
                 net_mask, num_movable_nodes):
        super(Partition, self).__init__()

        self.flat_netpin = flat_netpin
        self.netpin_start = netpin_start
        self.pin2node_map = pin2node_map
        self.net_weights = net_weights
        self.net_mask = net_mask
        self.num_movable_nodes = num_movable_nodes

    def __call__(self, tier):
        return PartitionFunction.forward(tier, self.flat_netpin, self.netpin_start,
                                         self.pin2node_map, self.net_weights,
                                         self.net_mask, self.num_movable_nodes)

if __name__ == "__main__":
    pass
