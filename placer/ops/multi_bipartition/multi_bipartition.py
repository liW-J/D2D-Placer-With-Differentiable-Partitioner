'''
Author: JeanneWillis hi@jeannewillis.cn
Date: 2025-03-19 11:47:31
LastEditors: JeanneWillis hi@jeannewillis.cn
LastEditTime: 2025-05-27 01:59:06
FilePath: /D2D-placer/placer/ops/partition/partition.py
Description: partition flattened 2D placement to 2 Die
'''
import torch
from torch.autograd import Function
from torch import nn

import logging

logger = logging.getLogger(__name__)

import ops.multi_bipartition.multi_bipartition_cpp as multi_bipartition_cpp


class MultiBipartitionFunction(Function):

    @staticmethod
    def forward(pos, flat_netpin, netpin_start, pin2node_map, net_weights, net_mask, num_movable_nodes, flat_node2pin_map, flat_node2pin_start_map, pin2net_map):
        func = multi_bipartition_cpp.multi_bipartition
        output = func(pos, flat_netpin, netpin_start, pin2node_map, net_weights,
                      net_mask, num_movable_nodes, flat_node2pin_map, flat_node2pin_start_map, pin2net_map)
        # breakpoint()
        return output

class MultiBipartition(nn.Module):

    def __init__(self,
                 flat_netpin=None,
                 netpin_start=None,
                 pin2node_map=None,
                 net_weights=None,
                 net_mask=None,
                 num_movable_nodes=None,
                 flat_node2pin_map=None,
                 flat_node2pin_start_map=None,
                 pin2net_map=None):

        super(MultiBipartition, self).__init__()
        assert net_mask is not None, "net_mask is a requried parameter"

        self.flat_netpin = flat_netpin
        self.netpin_start = netpin_start
        self.pin2node_map = pin2node_map
        self.net_weights = net_weights
        self.net_mask = net_mask
        self.num_movable_nodes = num_movable_nodes
        self.flat_node2pin_map = flat_node2pin_map
        self.flat_node2pin_start_map = flat_node2pin_start_map
        self.pin2net_map = pin2net_map
    
    def __call__(self, pos):
        return MultiBipartitionFunction.forward(pos, self.flat_netpin,
                                               self.netpin_start,
                                               self.pin2node_map,
                                               self.net_weights, 
                                               self.net_mask,
                                               self.num_movable_nodes,
                                               self.flat_node2pin_map,
                                               self.flat_node2pin_start_map,
                                               self.pin2net_map)


if __name__ == "__main__":
    pass
