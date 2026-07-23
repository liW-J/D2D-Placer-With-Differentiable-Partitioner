'''
Date: 2025-03-19 11:47:31
LastEditTime: 2025-04-23 21:41:42
FilePath: /D2D-placer/placer/ops/partition/partition.py
Description: partition flattened 2D placement to 2 Die
'''
import torch
from torch.autograd import Function
from torch import nn

import logging

logger = logging.getLogger(__name__)

import placer.ops.avg_cut.avg_cut_cpp as avg_cut_cpp


class AvgCutFunction(Function):

    @staticmethod
    def forward(flat_netpin, netpin_start, pin2node_map, net_weights,
                num_movable_nodes, pos, node_size_x, node_size_y,
                netpin_start_host=None):
        if pos.is_cuda:
            return AvgCutFunction._forward_cuda(flat_netpin, netpin_start,
                                                pin2node_map,
                                                num_movable_nodes, pos,
                                                netpin_start_host)
        func = avg_cut_cpp.avg_cut
        output = func(flat_netpin, netpin_start, pin2node_map, net_weights,
                      num_movable_nodes, pos, node_size_x, node_size_y)
        return output

    @staticmethod
    def _forward_cuda(flat_netpin, netpin_start, pin2node_map, num_movable_nodes,
                      pos, netpin_start_host=None):
        # GPU implementation that mirrors avg_cut.cpp behavior.
        num_nets = netpin_start.numel() - 1
        pin_x = pos[:pos.numel() // 2]
        tier = torch.zeros(num_movable_nodes,
                           dtype=torch.int32,
                           device=pos.device)
        net_count = [0, 0]
        if netpin_start_host is None:
            netpin_start_host = netpin_start.detach().cpu().tolist()

        for net_id in range(num_nets):
            start = int(netpin_start_host[net_id])
            end = int(netpin_start_host[net_id + 1])
            if end <= start:
                continue

            x_vals = pin_x[start:end]
            cut_x = (torch.min(x_vals) + torch.max(x_vals)) / 2
            net_degree = end - start
            tier_assign = 0 if net_count[1] > net_count[0] else 1

            node_ids = pin2node_map[flat_netpin[start:end]].long()
            if net_degree > 3:
                left_mask = x_vals < cut_x
                tier[node_ids[left_mask]] = tier_assign
                tier[node_ids[~left_mask]] = 1 - tier_assign
                net_count[tier_assign] += int(left_mask.sum().item())
                net_count[1 - tier_assign] += int((~left_mask).sum().item())
            else:
                tier[node_ids] = tier_assign

        return tier


class AvgCut(nn.Module):

    def __init__(self, flat_netpin, netpin_start, pin2node_map, net_weights,
                 num_movable_nodes):

        super(AvgCut, self).__init__()

        self.flat_netpin = flat_netpin
        self.netpin_start = netpin_start
        self.pin2node_map = pin2node_map
        self.net_weights = net_weights
        self.flat_netpin_cpu = flat_netpin.detach().cpu().contiguous()
        self.netpin_start_cpu = netpin_start.detach().cpu().contiguous()
        self.netpin_start_host = self.netpin_start_cpu.tolist()
        self.pin2node_map_cpu = pin2node_map.detach().cpu().contiguous()
        self.net_weights_cpu = net_weights.detach().cpu().contiguous()
        self.num_movable_nodes = num_movable_nodes
        self._cuda_cache = {}

    def _get_cached_cuda(self, name, tensor_cpu, device):
        key = (name, device)
        value = self._cuda_cache.get(key)
        if value is None or value.device != device:
            value = tensor_cpu.to(device, non_blocking=True)
            self._cuda_cache[key] = value
        return value

    def __call__(self, pos, node_size_x, node_size_y):
        if pos.is_cuda:
            return AvgCutFunction.forward(
                                          self._get_cached_cuda(
                                              "flat_netpin",
                                              self.flat_netpin_cpu, pos.device),
                                          self._get_cached_cuda(
                                              "netpin_start",
                                              self.netpin_start_cpu, pos.device),
                                          self._get_cached_cuda(
                                              "pin2node_map",
                                              self.pin2node_map_cpu, pos.device),
                                          self._get_cached_cuda(
                                              "net_weights",
                                              self.net_weights_cpu, pos.device),
                                          self.num_movable_nodes, pos,
                                          node_size_x, node_size_y,
                                          self.netpin_start_host)
        return AvgCutFunction.forward(self.flat_netpin_cpu,
                                      self.netpin_start_cpu,
                                      self.pin2node_map_cpu,
                                      self.net_weights_cpu,
                                      self.num_movable_nodes,
                                      pos.contiguous(),
                                      node_size_x.contiguous(),
                                      node_size_y.contiguous(),
                                      self.netpin_start_host)


if __name__ == "__main__":
    pass
