'''
Date: 2025-09-21 17:01:58
LastEditTime: 2025-10-14 23:42:41
FilePath: /D2D-placer/placer/ops/bin_based_fm/bin_based_fm.py
Description: fm refinement
'''

from torch.autograd import Function
import torch

import logging

logger = logging.getLogger(__name__)

import placer.ops.bin_based_fm.bin_based_fm_cpp as bin_based_fm_cpp


class BinBasedFMFunction(Function):

    @staticmethod
    def forward(tier, flat_netpin, netpin_start, pin2node_map, net_weights,
                num_movable_nodes, node_size_x, node_size_y, pin_offset_x,
                pin_offset_y, die_size_x, die_size_y, row_height,
                terminal_size_x, terminal_size_y, terminal_spacing, pin_pos,
                node_names, net_names, pos_2d, pos_terminal_legalized,
                case_name, num_terminals, terminal_names, top_die_max_util,
                bottom_die_max_util, num_nodes, num_bins_x, num_bins_y, xl, yl,
                xh, yh):
        target_device = tier.device
        func = bin_based_fm_cpp.bin_based_fm
        output = func(tier,
                      flat_netpin, netpin_start, pin2node_map,
                      net_weights, num_movable_nodes,
                      node_size_x, node_size_y,
                      pin_offset_x, pin_offset_y,
                      die_size_x, die_size_y, row_height,
                      terminal_size_x, terminal_size_y, terminal_spacing,
                      pin_pos, node_names, net_names, pos_2d,
                      pos_terminal_legalized, case_name, num_terminals,
                      terminal_names, top_die_max_util, bottom_die_max_util,
                      num_nodes, num_bins_x, num_bins_y, xl, yl, xh, yh)
        if torch.is_tensor(output) and target_device.type == "cuda":
            output = output.to(target_device, non_blocking=True)

        return output


class BinBasedFM(object):

    def __init__(self, flat_netpin, netpin_start, pin2node_map, net_weights,
                 num_movable_nodes, node_names, net_names, node_size_x,
                 node_size_y, pin_offset_x, pin_offset_y, die_size_x,
                 die_size_y, row_height, terminal_size_x, terminal_size_y,
                 terminal_spacing, case_name, top_die_max_util,
                 bottom_die_max_util, num_nodes, num_bins_x, num_bins_y, xl,
                 yl, xh, yh):
        super(BinBasedFM, self).__init__()

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
        self.flat_netpin_cpu = flat_netpin.detach().cpu().contiguous()
        self.netpin_start_cpu = netpin_start.detach().cpu().contiguous()
        self.pin2node_map_cpu = pin2node_map.detach().cpu().contiguous()
        self.net_weights_cpu = net_weights.detach().cpu().contiguous()
        self.node_size_x_cpu = node_size_x.detach().cpu().contiguous()
        self.node_size_y_cpu = node_size_y.detach().cpu().contiguous()
        self.pin_offset_x_cpu = pin_offset_x.detach().cpu().contiguous()
        self.pin_offset_y_cpu = pin_offset_y.detach().cpu().contiguous()

        self.top_die_max_util = top_die_max_util
        self.bottom_die_max_util = bottom_die_max_util
        self.num_nodes = num_nodes
        self.num_bins_x = num_bins_x
        self.num_bins_y = num_bins_y
        self.xl = xl
        self.yl = yl
        self.xh = xh
        self.yh = yh

    def __call__(self, tier, pin_pos, pos_2d, pos_terminal_legalized,
                 num_terminals, terminal_names):
        if tier.numel() < self.num_movable_nodes:
            raise ValueError(
                "bin_based_fm tier length (%d) is smaller than num_movable_nodes (%d)"
                % (tier.numel(), self.num_movable_nodes))
        if self.num_movable_nodes > 0:
            tier_head = tier[:self.num_movable_nodes].detach()
            tier_min = int(tier_head.min().item())
            tier_max = int(tier_head.max().item())
            if tier_min < 0 or tier_max > 1:
                raise ValueError(
                    "bin_based_fm expects tier ids in [0, 1], got min=%d max=%d"
                    % (tier_min, tier_max))
        if pos_2d.numel() < 2 * self.num_nodes:
            raise ValueError(
                "bin_based_fm pos_2d length (%d) is smaller than 2*num_nodes (%d)"
                % (pos_2d.numel(), 2 * self.num_nodes))
        if pos_terminal_legalized.numel() < 2 * num_terminals:
            raise ValueError(
                "bin_based_fm terminal position length (%d) is smaller than 2*num_terminals (%d)"
                % (pos_terminal_legalized.numel(), 2 * num_terminals))
        if any(torch.is_tensor(t) and t.is_cuda for t in
               (tier, pin_pos, pos_2d, pos_terminal_legalized)):
            torch.cuda.synchronize()

        output = BinBasedFMFunction.forward(
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
            pin_pos.cpu().contiguous(), self.node_names, self.net_names,
            pos_2d.cpu().contiguous(), pos_terminal_legalized.cpu().contiguous(),
            self.case_name, num_terminals,
            terminal_names, self.top_die_max_util, self.bottom_die_max_util,
            self.num_nodes, self.num_bins_x, self.num_bins_y, self.xl, self.yl,
            self.xh, self.yh)
        if torch.is_tensor(output) and tier.is_cuda:
            output = output.to(tier.device, non_blocking=True)
        return output


if __name__ == "__main__":
    pass
