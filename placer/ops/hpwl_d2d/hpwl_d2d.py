'''
Date: 2025-06-17 13:33:09
LastEditTime: 2025-07-20 23:39:14
FilePath: /D2D-placer/install/hpwl_d2d/hpwl_d2d.py
Description:
'''
import re

import torch
from torch.autograd import Function
from torch import nn
import numpy as np

import placer.ops.hpwl_d2d.hpwl_d2d_cpp as hpwl_d2d_cpp


class HPWLD2DFunction(Function):

    @staticmethod
    def _name_to_text(name):
        if isinstance(name, bytes):
            return name.decode("utf-8")
        if isinstance(name, np.bytes_):
            return name.tobytes().decode("utf-8")
        return str(name)

    @staticmethod
    def _terminal_aliases(name):
        text = HPWLD2DFunction._name_to_text(name)
        aliases = [text]
        match = re.match(r'^NET\d+_(.+)$', text)
        if match:
            aliases.append(match.group(1))
        return aliases

    @staticmethod
    def forward(pin_pos, flat_netpin, netpin_start, pin2node_map, net_weights,
                cut_net_mask, tier, num_tiers, pos_terminal_legalized,
                terminal_size_x, terminal_size_y, terminal_spacing,
                num_terminals, net_names, terminal_names,
                netpin_start_host=None):

        if pin_pos.is_cuda:
            return HPWLD2DFunction._forward_cuda(
                pin_pos, flat_netpin, netpin_start, pin2node_map, net_weights,
                tier, num_tiers, pos_terminal_legalized, terminal_size_x,
                terminal_size_y, terminal_spacing, num_terminals, net_names,
                terminal_names, netpin_start_host)

        has_non_movable_pins = (pin2node_map.numel() > 0 and
                                (int(pin2node_map.min().item()) < 0 or
                                 int(pin2node_map.max().item()) >=
                                 int(tier.numel())))
        if has_non_movable_pins:
            return HPWLD2DFunction._forward_cuda(
                pin_pos, flat_netpin, netpin_start, pin2node_map, net_weights,
                tier, num_tiers, pos_terminal_legalized, terminal_size_x,
                terminal_size_y, terminal_spacing, num_terminals, net_names,
                terminal_names, netpin_start_host)

        func = hpwl_d2d_cpp.hpwl_d2d
        output = func(pin_pos.view(pin_pos.numel()),
                      flat_netpin, netpin_start,
                      pin2node_map, net_weights,
                      cut_net_mask, tier, num_tiers,
                      pos_terminal_legalized,
                      terminal_size_x, terminal_size_y, terminal_spacing,
                      num_terminals, net_names, terminal_names)
        return output

    @staticmethod
    def _forward_cuda(pin_pos, flat_netpin, netpin_start, pin2node_map,
                      net_weights, tier, num_tiers, pos_terminal_legalized,
                      terminal_size_x, terminal_size_y, terminal_spacing,
                      num_terminals, net_names, terminal_names,
                      netpin_start_host=None):
        pin_pos = pin_pos.contiguous().view(-1)
        num_nets = netpin_start.numel() - 1
        num_pins = pin2node_map.numel()
        pin_x = pin_pos[:pin_pos.numel() // 2]
        pin_y = pin_pos[pin_pos.numel() // 2:]
        tier_long = tier.long()
        hpwl = torch.zeros(num_nets, dtype=pin_pos.dtype, device=pin_pos.device)
        if netpin_start_host is None:
            netpin_start_host = netpin_start.detach().cpu().tolist()
        terminal_id_by_net = {}
        for idx, name in enumerate(terminal_names):
            for alias in HPWLD2DFunction._terminal_aliases(name):
                terminal_id_by_net.setdefault(alias, idx)

        if pos_terminal_legalized.numel() > 0:
            term_x = pos_terminal_legalized[:pos_terminal_legalized.numel() // 2]
            term_y = pos_terminal_legalized[pos_terminal_legalized.numel() // 2:]
        else:
            term_x = None
            term_y = None

        for net_id in range(num_nets):
            start = int(netpin_start_host[net_id])
            end = int(netpin_start_host[net_id + 1])
            if end <= start:
                continue

            pins = flat_netpin[start:end].long()
            nodes = pin2node_map[pins].long()
            movable_mask = (nodes >= 0) & (nodes < tier_long.numel())
            if not bool(movable_mask.any().item()):
                continue
            pins = pins[movable_mask]
            nodes = nodes[movable_mask]

            node_tiers = tier_long[nodes]
            valid_tier_mask = (node_tiers >= 0) & (node_tiers < num_tiers)
            if not bool(valid_tier_mask.all().item()):
                bad_tiers = node_tiers[~valid_tier_mask].detach().cpu().tolist()
                raise RuntimeError(
                    "HPWL_D2D received tier ids outside [0, %d): %s" %
                    (num_tiers, bad_tiers[:8]))

            pin_index = node_tiers * num_pins + pins
            px = pin_x[pin_index]
            py = pin_y[pin_index]

            num_nodes_in_net = int(nodes.numel())
            node_count = [0] * num_tiers
            max_x = [None] * num_tiers
            min_x = [None] * num_tiers
            max_y = [None] * num_tiers
            min_y = [None] * num_tiers

            for t in range(num_tiers):
                mask = node_tiers == t
                cnt = int(mask.sum().item())
                node_count[t] = cnt
                if cnt > 0:
                    tpx = px[mask]
                    tpy = py[mask]
                    max_x[t] = torch.max(tpx)
                    min_x[t] = torch.min(tpx)
                    max_y[t] = torch.max(tpy)
                    min_y[t] = torch.min(tpy)

            is_cut = True
            single_t = -1
            for t, cnt in enumerate(node_count):
                if cnt == num_nodes_in_net:
                    is_cut = False
                    single_t = t
                    break

            if is_cut:
                net_name = HPWLD2DFunction._name_to_text(net_names[net_id])
                term_id = terminal_id_by_net.get(net_name, -1)
                if term_id >= 0 and term_id < num_terminals and term_x is not None:
                    tx = term_x[term_id] + (terminal_size_x + terminal_spacing) / 2
                    ty = term_y[term_id] + (terminal_size_y + terminal_spacing) / 2
                else:
                    valid_max_x = torch.stack([v for v in max_x if v is not None])
                    valid_min_x = torch.stack([v for v in min_x if v is not None])
                    valid_max_y = torch.stack([v for v in max_y if v is not None])
                    valid_min_y = torch.stack([v for v in min_y if v is not None])
                    tx = (torch.max(valid_min_x) + torch.min(valid_max_x)) / 2
                    ty = (torch.max(valid_min_y) + torch.min(valid_max_y)) / 2

                net_hpwl = pin_pos.new_zeros(())
                for t in range(num_tiers):
                    if max_x[t] is None:
                        continue
                    net_hpwl = net_hpwl + (
                        torch.maximum(max_x[t], tx) - torch.minimum(min_x[t], tx)
                        + torch.maximum(max_y[t], ty) - torch.minimum(min_y[t], ty))
                hpwl[net_id] = net_hpwl
            elif single_t >= 0:
                hpwl[net_id] = ((max_x[single_t] - min_x[single_t]) +
                                (max_y[single_t] - min_y[single_t]))

        if net_weights.numel() > 0:
            hpwl = hpwl * net_weights.to(hpwl.device)
        return hpwl.sum()

class HPWLD2D(object):
    """ 
    @brief Compute half-perimeter wirelength. 
    Support two algoriths: net-by-net and atomic. 
    Different parameters are required for different algorithms. 
    """

    def __init__(self, flat_netpin, netpin_start, pin2node_map, net_weights,
                 terminal_size_x, terminal_size_y, terminal_spacing, net_names,
                 num_tiers):

        super(HPWLD2D, self).__init__()

        self.flat_netpin = flat_netpin
        self.netpin_start = netpin_start
        self.pin2node_map = pin2node_map
        self.net_weights = net_weights
        self.flat_netpin_cpu = flat_netpin.detach().cpu().contiguous()
        self.netpin_start_cpu = netpin_start.detach().cpu().contiguous()
        self.netpin_start_host = self.netpin_start_cpu.tolist()
        self.pin2node_map_cpu = pin2node_map.detach().cpu().contiguous()
        self.net_weights_cpu = net_weights.detach().cpu().contiguous()
        self.terminal_size_x = terminal_size_x
        self.terminal_size_y = terminal_size_y
        self.terminal_spacing = terminal_spacing
        self.net_names = net_names
        self.net_name_set = {
            HPWLD2DFunction._name_to_text(name) for name in net_names
        }
        self.num_tiers = num_tiers
        self._cuda_cache = {}

    def _terminal_names_for_hpwl(self, terminal_names):
        if len(terminal_names) == 0:
            return terminal_names

        names = []
        changed = False
        for name in terminal_names:
            text = HPWLD2DFunction._name_to_text(name)
            mapped = text
            match = re.match(r'^NET\d+_(.+)$', text)
            if text not in self.net_name_set and match:
                candidate = match.group(1)
                if candidate in self.net_name_set:
                    mapped = candidate
            changed = changed or mapped != text
            names.append(mapped)

        if not changed:
            return terminal_names
        return np.array(names, dtype=np.bytes_)

    def _get_cached_cuda(self, name, tensor_cpu, device):
        key = (name, device)
        value = self._cuda_cache.get(key)
        if value is None or value.device != device:
            value = tensor_cpu.to(device, non_blocking=True)
            self._cuda_cache[key] = value
        return value

    def __call__(self,
                 pin_pos,
                 cut_net_mask,
                 tier,
                 pos_terminal_legalized=torch.empty(0),
                 num_terminals=0,
                 terminal_names=np.array([], dtype=np.bytes_)):
        terminal_names = self._terminal_names_for_hpwl(terminal_names)

        if pin_pos.is_cuda:
            return HPWLD2DFunction.forward(
                pin_pos,
                self._get_cached_cuda("flat_netpin", self.flat_netpin_cpu,
                                      pin_pos.device),
                self._get_cached_cuda("netpin_start", self.netpin_start_cpu,
                                      pin_pos.device),
                self._get_cached_cuda("pin2node_map", self.pin2node_map_cpu,
                                      pin_pos.device),
                self._get_cached_cuda("net_weights", self.net_weights_cpu,
                                      pin_pos.device),
                cut_net_mask.to(pin_pos.device),
                tier.to(pin_pos.device),
                self.num_tiers,
                pos_terminal_legalized.to(pin_pos.device),
                self.terminal_size_x,
                self.terminal_size_y,
                self.terminal_spacing,
                num_terminals,
                self.net_names,
                terminal_names,
                self.netpin_start_host)

        return HPWLD2DFunction.forward(
            pin_pos.contiguous().cpu(),
            self.flat_netpin_cpu,
            self.netpin_start_cpu,
            self.pin2node_map_cpu,
            self.net_weights_cpu,
            cut_net_mask.contiguous().cpu(),
            tier.contiguous().cpu(),
            self.num_tiers,
            pos_terminal_legalized.contiguous().cpu(),
            self.terminal_size_x,
            self.terminal_size_y,
            self.terminal_spacing,
            num_terminals,
            self.net_names,
            terminal_names,
            self.netpin_start_host)


if __name__ == "__main__":
    pass
