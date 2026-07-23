'''
Date: 2025-03-19 11:47:31
LastEditTime: 2025-07-23 15:40:09
FilePath: /D2D-placer/placer/ops/partition/partition.py
Description: partition flattened 2D placement to 2 Die
'''
from torch.autograd import Function
import torch

import logging
import os

logger = logging.getLogger(__name__)

import placer.ops.partition_aux.partition_aux_cpp as partition_aux_cpp


class PartitionAuxFunction(Function):

    @staticmethod
    def forward(tier, flat_netpin, netpin_start, pin2node_map, net_weights,
                num_movable_nodes, node_size_x, node_size_y, pin_offset_x,
                pin_offset_y, die_size_x, die_size_y, row_height,
                terminal_size_x, terminal_size_y, terminal_spacing, pin_pos,
                terminal_instert_flag, terminal_legalize_flag,
                pos_terminal_legalized, num_terminals, node_names, net_names,
                terminal_names, pos_2d, case_name, node_orient):
        target_device = tier.device
        func = partition_aux_cpp.partition_aux
        output = func(tier,
                      flat_netpin, netpin_start, pin2node_map,
                      net_weights, num_movable_nodes,
                      node_size_x, node_size_y,
                      pin_offset_x, pin_offset_y,
                      die_size_x, die_size_y, row_height,
                      terminal_size_x, terminal_size_y, terminal_spacing,
                      pin_pos, terminal_instert_flag, terminal_legalize_flag,
                      pos_terminal_legalized, num_terminals,
                      node_names, net_names, terminal_names,
                      pos_2d, case_name, node_orient)
        if torch.is_tensor(output) and target_device.type == "cuda":
            output = output.to(target_device, non_blocking=True)

        return output


class PartitionAux(object):

    def __init__(self, flat_netpin, netpin_start, pin2node_map, net_weights,
                 num_movable_nodes, node_names, net_names, node_size_x,
                 node_size_y, pin_offset_x, pin_offset_y, die_size_x,
                 die_size_y, row_height, terminal_size_x, terminal_size_y,
                 terminal_spacing, terminal_instert_flag,
                 terminal_legalize_flag, case_name, output_dir=None):
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
        self.output_dir = output_dir or f"./run_tmp/{self.case_name}/partition"
        self.flat_netpin_cpu = flat_netpin.detach().cpu().contiguous()
        self.netpin_start_cpu = netpin_start.detach().cpu().contiguous()
        self.pin2node_map_cpu = pin2node_map.detach().cpu().contiguous()
        self.net_weights_cpu = net_weights.detach().cpu().contiguous()
        self.node_size_x_cpu = node_size_x.detach().cpu().contiguous()
        self.node_size_y_cpu = node_size_y.detach().cpu().contiguous()
        self.pin_offset_x_cpu = pin_offset_x.detach().cpu().contiguous()
        self.pin_offset_y_cpu = pin_offset_y.detach().cpu().contiguous()

    def __call__(self,
                 tier,
                 node_orient,
                 pin_pos=torch.empty(0),
                 pos_terminal_legalized=torch.empty(0),
                 num_terminals=0,
                 pos_2d=torch.empty(0),
                 terminal_names=[]):
        os.makedirs(self.output_dir, exist_ok=True)
        output = PartitionAuxFunction.forward(
            tier.cpu().contiguous(),
            self.flat_netpin_cpu,
            self.netpin_start_cpu,
            self.pin2node_map_cpu,
            self.net_weights_cpu,
            self.num_movable_nodes,
            self.node_size_x_cpu,
            self.node_size_y_cpu,
            self.pin_offset_x_cpu,
            self.pin_offset_y_cpu,
            self.die_size_x, self.die_size_y, self.row_height,
            self.terminal_size_x, self.terminal_size_y, self.terminal_spacing,
            pin_pos.cpu().contiguous(), self.terminal_instert_flag,
            self.terminal_legalize_flag, pos_terminal_legalized.cpu().contiguous(),
            num_terminals, self.node_names, self.net_names, terminal_names,
            pos_2d.cpu().contiguous(), self.output_dir,
            node_orient)
        if torch.is_tensor(output) and tier.is_cuda:
            output = output.to(tier.device, non_blocking=True)
        return output


if __name__ == "__main__":
    pass
