'''
Author: JeanneWillis hi@jeannewillis.cn
Date: 2025-11-14 16:06:43
LastEditors: JeanneWillis hi@jeannewillis.cn
LastEditTime: 2026-01-31 18:51:24
FilePath: /D2D-placer/install/placer/tools/differentiable_partitioner/partitioner.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
'''
Author: JeanneWillis hi@jeannewillis.cn
Date: 2025-11-14 16:03:37
LastEditors: JeanneWillis hi@jeannewillis.cn
LastEditTime: 2026-01-31 16:34:56
FilePath: /D2D-placer/placer/tools/differentiable_partitioner/partitioner.py
Description: Differentiable 3D Partitioner based on LogSumExp soft bounding box
'''
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import os

COORD_EPSILON = 1e-2


class LSEPartitioner(nn.Module):
    """
    Differentiable 3D Partitioner based on LogSumExp soft bounding box
    
    Features:
    1. Mapping trainable pre-activation variable t_i to soft assignment z_i = sigmoid(t_i)
    2. Using LSE soft bounding box to compute HPWL of top and bottom layers
    3. Computing gradient of total HPWL with respect to t, supporting end-to-end training
    4. Support differentiable cutsize loss based on Snake-3D, used to reduce cross-layer connections
    """

    def __init__(self,
                 num_cells,
                 flat_net2pin_map,
                 flat_net2pin_start_map,
                 pin2node_map,
                 pin_pos_x,
                 pin_pos_y,
                 node_x,
                 node_y,
                 node_size_x,
                 node_size_y,
                 alpha=1.0,
                 net_weights=None,
                 gumbel_tau=0.1):
        """
        initialize LSE partitioner
        
        Args:
            num_cells: number of cells (nodes)
            flat_net2pin_map: flat net to pin mapping, shape [num_pins] tensor
            flat_net2pin_start_map: start index of each net in flat_net2pin_map, shape [num_nets+1] tensor
            pin2node_map: pin to node mapping, shape [num_pins] tensor
            pin_pos_x: x coordinates of pins, shape [num_pins] tensor
            pin_pos_y: y coordinates of pins, shape [num_pins] tensor
            alpha: LSE smoothing parameter, larger means harder to be 0 or 1
            net_weights: optional net weights, shape [num_nets] tensor
            gumbel_tau: Gumbel Softmax temperature parameter, controlling the smoothness of softmax (default 0.1)
                        smaller tau means results closer to discrete distribution; larger tau means smoother distribution
        """
        super(LSEPartitioner, self).__init__()

        self.num_cells = num_cells
        self.num_nets = flat_net2pin_start_map.numel() - 1

        # register fixed data structures (no gradient)
        self.register_buffer('flat_net2pin_map',
                             flat_net2pin_map.detach().clone())
        self.register_buffer('flat_net2pin_start_map',
                             flat_net2pin_start_map.detach().clone())
        self.register_buffer('pin2node_map', pin2node_map.detach().clone())
        self.register_buffer('pin_pos_x', pin_pos_x.detach().clone())
        self.register_buffer('pin_pos_y', pin_pos_y.detach().clone())
        self.register_buffer('node_x', node_x.detach().clone())
        self.register_buffer('node_y', node_y.detach().clone())
        self.register_buffer('node_size_x', node_size_x.detach().clone())
        self.register_buffer('node_size_y', node_size_y.detach().clone())

        self.x_range = node_x.max() - node_x.min()
        self.y_range = node_y.max() - node_y.min()

        # trainable pre-activation variable t_i (one for each cell)
        # use small random initialization to avoid all z being 0.5 (symmetric point)
        self.t = nn.Parameter(torch.randn(num_cells) * 0.1)
        # self.t = torch.load("node_die_after_fm_wl.pt").float().to(
        #     pin_pos_x.device)
        # self.t[self.t > 0] = 10.0
        # self.t[self.t <= 0] = -10.0
        # self.t = nn.Parameter(self.t)

        # initial_values = torch.tensor([10.0, -10.0, -10.0, -10.0],
        #                               device=self.pin_pos_x.device)
        # initial_values = torch.tensor(
        #     [-10.0, 10.0, 10.0, 10.0, -10.0, 10.0, 10.0, 10.0],
        #     device=self.pin_pos_x.device)
        # self.t = nn.Parameter(initial_values)

        # LSE smoothing parameter
        self.alpha = alpha

        # Gumbel Softmax temperature parameter
        self.gumbel_tau = gumbel_tau
        self.gumbel_switch_iteration = 0

        # Current iteration counter (used to switch between sigmoid and gumbel_softmax)
        self.current_iteration = 0

        # net weights (if not provided, default to all 1)
        if net_weights is None:
            self.register_buffer('net_weights', torch.ones(self.num_nets))
        else:
            self.register_buffer('net_weights', net_weights.detach().clone())

    def get_z(self):
        """
        map pre-activation variable t to soft assignment z
        Returns:
            z: shape [num_cells] tensor, representing the probability of each cell being assigned to the top layer
        """
        # switch to gumbel_softmax_z after gumbel_switch_iteration iterations
        if self.current_iteration < self.gumbel_switch_iteration:
            return torch.sigmoid(self.t)
        else:
            return self.gumbel_softmax_z(self.t, tau=self.gumbel_tau)

    def gumbel_softmax_z(self, pi_logits, tau=0.1):
        """
        use Gumbel Softmax to convert logits to soft assignment probabilities   
        Gumbel Softmax is a differentiable sampling method for discrete variables.

        Args:
            pi_logits: shape [num_cells] tensor, logit values for each cell
            tau: temperature parameter, controlling the smoothness of the softmax
                 smaller tau means more discrete distribution; larger tau means more smooth distribution
        
        Returns:
            z: shape [num_cells] tensor, probability of each cell being assigned to the top layer
        """
        # convert single logit t to two logits: [t, 0]
        # the first logit corresponds to top layer, the second logit corresponds to bottom layer
        # use [t, 0] instead of [t, -t] to keep the semantic consistent with the original sigmoid
        # sigmoid(t) = exp(t) / (exp(t) + exp(0)) = exp(t) / (exp(t) + 1)
        logits = torch.stack(
            [pi_logits, torch.zeros_like(pi_logits)], dim=-1)  # [num_cells, 2]

        # generate Gumbel noise: G = -log(-log(U)), where U ~ Uniform(0,1)
        # use numerically stable implementation
        uniform = torch.rand_like(logits)
        # avoid log(0) and log(1), use clamp
        uniform = torch.clamp(uniform, min=1e-8, max=1.0 - 1e-8)
        gumbel_noise = -torch.log(-torch.log(uniform))

        # add Gumbel noise and divide by temperature parameter
        gumbel_logits = (logits + gumbel_noise) / tau  # [num_cells, 2]
        softmax_probs = torch.softmax(gumbel_logits, dim=-1)  # [num_cells, 2]

        # return the first element (probability of top layer)
        z = softmax_probs[..., 0]  # [num_cells]

        return z

    def lse_max(self, weighted_vals):
        """
        numerically stable LSE maximum calculation
        
        formula: max(x) ≈ (1/α) * log_sum_exp(α * x)
        
        numerically stable implementation: log_sum_exp(a_j) = m + log Σ_j exp(a_j - m)
        where m = max_j a_j
        
        Args:
            weighted_vals: weighted values, shape [..., ...] tensor
            
        Returns:
            soft maximum, shape same as weighted_vals (except the aggregated dimension)
        """
        # use torch.logsumexp to ensure numerical stability
        # logsumexp(α * x) = log(Σ exp(α * x))
        # then divide by α to get soft maximum
        if self.alpha <= 0:
            raise ValueError("alpha must be positive")

        # calculate log_sum_exp(α * weighted_vals)
        lse = torch.logsumexp(self.alpha * weighted_vals, dim=-1)

        # return (1/α) * log_sum_exp(α * weighted_vals)
        return lse / self.alpha

    def lse_min(self, weighted_vals):
        """
        numerically stable LSE minimum calculation
        formula: min(x) ≈ -(1/α) * log_sum_exp(-α * x)
        
        Args:
            weighted_vals: weighted values, shape [..., ...] tensor
            
        Returns:
            soft minimum, shape same as weighted_vals (except the aggregated dimension)
        """
        if self.alpha <= 0:
            raise ValueError("alpha must be positive")

        # calculate log_sum_exp(-α * weighted_vals)
        lse = torch.logsumexp(-self.alpha * weighted_vals, dim=-1)

        # return -(1/α) * log_sum_exp(-α * weighted_vals)
        return -lse / self.alpha

    def compute_cutsize(self, net_idx):
        """
        calculate differentiable cutsize of the specified net (based on Snake-3D formula)
        
        formula:
        - LSE-max(z_n) = (1/α) * log( Σ_{i∈n} exp( α * z_i ) )
        - LSE-min(z_n) = -(1/α) * log( Σ_{i∈n} exp(-α * z_i ) )
        - cutsize(n) = (1 - LSE-min(z_i for i∈n)) * LSE-max(z_i for i∈n)
        
        This formula penalizes cross-layer connections: when all nodes are in the same layer (z_i are close to 0 or 1),
        cutsize is close to 0; when nodes are distributed between two layers, cutsize increases.
        
        Args:
            net_idx: index of the net
            
        Returns:
            differentiable cutsize of the specified net, scalar tensor
        """
        # get all pin indices for this net
        start_idx = self.flat_net2pin_start_map[net_idx]
        end_idx = self.flat_net2pin_start_map[net_idx + 1]
        pin_indices = self.flat_net2pin_map[start_idx:end_idx]

        if pin_indices.numel() < 2:
            return torch.tensor(0.0, device=self.pin_pos_x.device)

        z = self.get_z()
        node_indices = self.pin2node_map[
            pin_indices]  # shape: [num_pins_in_net]
        z_net = z[node_indices]  # shape: [num_pins_in_net]

        # calculate LSE-max and LSE-min
        # LSE-max(z_n) = (1/α) * log_sum_exp(α * z_i)
        lse_max_z = self.lse_max(z_net)

        # LSE-min(z_n) = -(1/α) * log_sum_exp(-α * z_i)
        lse_min_z = self.lse_min(z_net)

        # cutsize(n) = (1 - LSE-min(z_i)) * LSE-max(z_i)
        #
        # mathematical analysis: why cutsize can be greater than 1?
        # 1. range of LSE-max and LSE-min:
        #    - when z_i is in [0,1] range, LSE-max ∈ [0, 1+log(n)/α], LSE-min ∈ [-log(n)/α, 1]
        #    - for larger α, LSE-max and LSE-min are both close to [0,1] range
        # 2. behavior of cutsize:
        #    - when all z_i≈0: LSE-min≈0, LSE-max≈0 → cutsize≈0
        #    - when all z_i≈1: LSE-min≈1, LSE-max≈1 → cutsize≈0
        #    - when z_i mixed distribution (some≈0, some≈1):
        #      * LSE-min close to 0 because some z_i≈0
        #      * LSE-max close to 1 because some z_i≈1
        #      * cutsize = (1 - close to 0) * close to 1 ≈ 1
        #    - when z_i uniformly distributed in middle (e.g. 0.3-0.7):
        #      * LSE-min close to 0.3-0.4
        #      * LSE-max close to 0.6-0.7
        #      * cutsize = (1 - 0.3-0.4) * 0.6-0.7 ≈ 0.4-0.5
        #    - when z_i distributed unevenly and with large span (e.g. 0.1 and 0.9 mixed):
        #      * LSE-min close to 0.1
        #      * LSE-max close to 0.9
        #      * cutsize = (1 - 0.1) * 0.9 = 0.9 * 0.9 = 0.81
        #    - when network has many pins and z_i distributed widely:
        #      * LSE-max can be greater than 1 (especially when α is small)
        #      * e.g.: n=10 pins, z_i uniformly distributed in [0,1], α=1.0
        #        LSE-max ≈ (1/1) * log(10 * exp(1*0.5)) ≈ log(10*1.65) ≈ 2.8
        #        LSE-min ≈ -(1/1) * log(10 * exp(-1*0.5)) ≈ -log(10*0.61) ≈ -1.8
        #        cutsize = (1 - (-1.8)) * 2.8 = 2.8 * 2.8 = 7.84
        #
        # conclusion: cutsize > 1 is normal, it represents the "soft penalty strength" of cross-layer connection
        # it is not a probability value, but a penalty term, the larger the value, the higher the degree of the network crossing two layers
        cutsize = (1.0 - lse_min_z) * lse_max_z

        return cutsize

    def compute_hpwl_batch(self, net_indices, layer='both'):
        """
        batch calculation of HPWL for multiple nets (vectorized version)
        
        Args:
            net_indices: network indices, tensor of shape [num_nets]
            layer: 'top', 'bottom', or 'both' (default 'both')
        
        Returns:
            if layer == 'both': tuple of (hpwl_top, hpwl_bottom), each shape [num_nets]
            else: hpwl values, tensor of shape [num_nets]
        """
        if net_indices.numel() == 0:
            empty = torch.tensor([], device=self.pin_pos_x.device)
            if layer == 'both':
                return empty, empty
            return empty

        num_nets = net_indices.numel()
        z = self.get_z()

        # get start and end indices for all nets
        start_indices = self.flat_net2pin_start_map[net_indices]  # [num_nets]
        end_indices = self.flat_net2pin_start_map[net_indices +
                                                  1]  # [num_nets]

        # calculate pin count for each net
        pin_counts = end_indices - start_indices  # [num_nets]
        valid_mask = pin_counts >= 2  # [num_nets]

        if not valid_mask.any():
            zeros = torch.zeros(num_nets, device=self.pin_pos_x.device)
            if layer == 'both':
                return zeros, zeros
            return zeros

        # only process valid nets
        valid_start_indices = start_indices[valid_mask]  # [num_valid_nets]
        valid_end_indices = end_indices[valid_mask]  # [num_valid_nets]
        valid_pin_counts = pin_counts[valid_mask]  # [num_valid_nets]
        valid_net_indices = net_indices[valid_mask]  # [num_valid_nets]
        num_valid_nets = valid_pin_counts.numel()

        # collect all valid net's pin indices
        all_pin_indices = []
        for i in range(num_valid_nets):
            start_idx = valid_start_indices[i].item()
            end_idx = valid_end_indices[i].item()
            all_pin_indices.append(self.flat_net2pin_map[start_idx:end_idx])

        all_pin_indices = torch.cat(all_pin_indices)  # [total_pins]

        # get all pin corresponding node indices and z values
        all_node_indices = self.pin2node_map[all_pin_indices]  # [total_pins]
        all_z_net = z[all_node_indices]  # [total_pins]

        # get position of each pin with random selection
        # randomly choose between original value or max_val - value for each pin
        all_x_net = self.pin_pos_x[all_pin_indices]  # [total_pins]
        all_y_net = self.pin_pos_y[all_pin_indices]  # [total_pins]
        all_x_net = all_x_net - all_x_net.min() + COORD_EPSILON
        all_y_net = all_y_net - all_y_net.min() + COORD_EPSILON
        all_x_net_rev = all_x_net.max() - all_x_net + COORD_EPSILON
        all_y_net_rev = all_y_net.max() - all_y_net + COORD_EPSILON

        # compute weighted values for top layer: z_node * x_pin and z_node * y_pin
        weighted_x_top_max = all_z_net * all_x_net  # [total_pins]
        weighted_y_top_max = all_z_net * all_y_net  # [total_pins]
        weighted_x_top_min = all_z_net * all_x_net_rev  # [total_pins]
        weighted_y_top_min = all_z_net * all_y_net_rev  # [total_pins]

        # compute weighted values for bottom layer: (1 - z_node) * x_pin and (1 - z_node) * y_pin
        z_bottom = 1.0 - all_z_net  # [total_pins]
        weighted_x_bottom_max = z_bottom * all_x_net_rev  # [total_pins]
        weighted_y_bottom_max = z_bottom * all_y_net_rev  # [total_pins]
        weighted_x_bottom_min = z_bottom * all_x_net  # [total_pins]
        weighted_y_bottom_min = z_bottom * all_y_net  # [total_pins]

        # vectorized calculation of HPWL for each net
        hpwl_top_per_net = torch.zeros(num_valid_nets,
                                       device=self.pin_pos_x.device)
        hpwl_bottom_per_net = torch.zeros(num_valid_nets,
                                          device=self.pin_pos_x.device)

        # calculate HPWL for each net using vectorized approach

        # precompute pin offsets for all nets
        pin_offsets = torch.cumsum(torch.cat([
            torch.tensor([0], device=valid_pin_counts.device),
            valid_pin_counts[:-1]
        ]),
                                   dim=0)  # [num_valid_nets]

        # vectorized HPWL calculation by grouping nets with same pin count
        # Strategy: group nets by pin count, then process each group in parallel
        unique_pin_counts, inverse_indices, counts = torch.unique(
            valid_pin_counts, return_inverse=True, return_counts=True)

        # process each group of nets with same pin count
        for group_idx, pin_count in enumerate(unique_pin_counts):
            # get indices of nets in this group
            group_mask = inverse_indices == group_idx
            group_net_indices = torch.where(group_mask)[
                0]  # [num_nets_in_group]
            num_nets_in_group = group_net_indices.numel()

            if num_nets_in_group == 0:
                continue

            pin_count_int = pin_count.item()

            # get pin offsets for this group
            group_pin_offsets = pin_offsets[
                group_net_indices]  # [num_nets_in_group]

            # create index tensor for batch slicing: [num_nets_in_group, pin_count]
            # each row corresponds to one net's pins
            batch_indices = (group_pin_offsets.unsqueeze(1) + torch.arange(
                pin_count_int, device=group_pin_offsets.device).unsqueeze(0))

            # batch extract data for all nets in this group
            # shape: [num_nets_in_group, pin_count]
            if layer in ['top', 'both']:
                wx_top_max_batch = weighted_x_top_max[batch_indices]
                wy_top_max_batch = weighted_y_top_max[batch_indices]
                wx_top_min_batch = weighted_x_top_min[batch_indices]
                wy_top_min_batch = weighted_y_top_min[batch_indices]

                # batch LSE computation: apply lse_max along pin dimension
                # lse_max uses logsumexp which supports batch dimension
                x_top_max_batch = self.lse_max(wx_top_max_batch)
                y_top_max_batch = self.lse_max(wy_top_max_batch)
                x_top_min_batch = self.lse_max(wx_top_min_batch)
                y_top_min_batch = self.lse_max(wy_top_min_batch)

                # batch max computation
                wx_top_max_max = wx_top_max_batch.max(dim=1)[0]
                wx_top_min_max = wx_top_min_batch.max(dim=1)[0]
                wy_top_max_max = wy_top_max_batch.max(dim=1)[0]
                wy_top_min_max = wy_top_min_batch.max(dim=1)[0]

                # vectorized HPWL calculation for this group
                hpwl_top_group = (
                    x_top_max_batch + y_top_max_batch + x_top_min_batch +
                    y_top_min_batch -
                    torch.maximum(wx_top_max_max, wx_top_min_max) -
                    torch.maximum(wy_top_max_max, wy_top_min_max))

                # assign results back
                hpwl_top_per_net[group_net_indices] = hpwl_top_group

            if layer in ['bottom', 'both']:
                wx_bottom_max_batch = weighted_x_bottom_max[batch_indices]
                wy_bottom_max_batch = weighted_y_bottom_max[batch_indices]
                wx_bottom_min_batch = weighted_x_bottom_min[batch_indices]
                wy_bottom_min_batch = weighted_y_bottom_min[batch_indices]

                # batch LSE computation
                x_bottom_max_batch = self.lse_max(wx_bottom_max_batch)
                y_bottom_max_batch = self.lse_max(wy_bottom_max_batch)
                x_bottom_min_batch = self.lse_max(wx_bottom_min_batch)
                y_bottom_min_batch = self.lse_max(wy_bottom_min_batch)

                # batch max computation
                wx_bottom_max_max = wx_bottom_max_batch.max(dim=1)[0]
                wx_bottom_min_max = wx_bottom_min_batch.max(dim=1)[0]
                wy_bottom_max_max = wy_bottom_max_batch.max(dim=1)[0]
                wy_bottom_min_max = wy_bottom_min_batch.max(dim=1)[0]

                # vectorized HPWL calculation for this group
                hpwl_bottom_group = (
                    x_bottom_max_batch + y_bottom_max_batch +
                    x_bottom_min_batch + y_bottom_min_batch -
                    torch.maximum(wx_bottom_max_max, wx_bottom_min_max) -
                    torch.maximum(wy_bottom_max_max, wy_bottom_min_max))

                # assign results back
                hpwl_bottom_per_net[group_net_indices] = hpwl_bottom_group

        # create complete result arrays (including invalid nets)
        if layer == 'both':
            result_top = torch.zeros(num_nets, device=self.pin_pos_x.device)
            result_bottom = torch.zeros(num_nets, device=self.pin_pos_x.device)
            result_top[valid_mask] = hpwl_top_per_net
            result_bottom[valid_mask] = hpwl_bottom_per_net
            return result_top, result_bottom
        elif layer == 'top':
            result = torch.zeros(num_nets, device=self.pin_pos_x.device)
            result[valid_mask] = hpwl_top_per_net
            return result
        else:  # layer == 'bottom'
            result = torch.zeros(num_nets, device=self.pin_pos_x.device)
            result[valid_mask] = hpwl_bottom_per_net
            return result

    def compute_cutsize_batch(self, net_indices):
        """
        batch calculation of differentiable cutsize for multiple nets (vectorized version)
        
        Args:
            net_indices: network indices, tensor of shape [num_nets]
        
        Returns:
            cutsize values, tensor of shape [num_nets]
        """
        if net_indices.numel() == 0:
            return torch.tensor([], device=self.pin_pos_x.device)

        num_nets = net_indices.numel()
        z = self.get_z()

        # get start and end indices for all nets
        start_indices = self.flat_net2pin_start_map[net_indices]  # [num_nets]
        end_indices = self.flat_net2pin_start_map[net_indices +
                                                  1]  # [num_nets]

        # calculate pin count for each net
        pin_counts = end_indices - start_indices  # [num_nets]

        # filter out nets with less than 2 pins (these nets have cutsize=0)
        valid_mask = pin_counts >= 2  # [num_nets]

        if not valid_mask.any():
            return torch.zeros(num_nets, device=self.pin_pos_x.device)

        # only process valid nets
        valid_start_indices = start_indices[valid_mask]  # [num_valid_nets]
        valid_end_indices = end_indices[valid_mask]  # [num_valid_nets]
        valid_pin_counts = pin_counts[valid_mask]  # [num_valid_nets]
        num_valid_nets = valid_pin_counts.numel()

        # collect all valid net's pin indices
        all_pin_indices = []
        for i in range(num_valid_nets):
            start_idx = valid_start_indices[i].item()
            end_idx = valid_end_indices[i].item()
            all_pin_indices.append(self.flat_net2pin_map[start_idx:end_idx])

        all_pin_indices = torch.cat(all_pin_indices)  # [total_pins]

        # get all pin corresponding node indices and z values
        all_node_indices = self.pin2node_map[all_pin_indices]  # [total_pins]
        all_z_net = z[all_node_indices]  # [total_pins]

        # vectorized calculation of LSE-max and LSE-min for each net
        # using grouped calculation: calculate logsumexp for each net separately
        lse_max_per_net = torch.zeros(num_valid_nets,
                                      device=self.pin_pos_x.device)
        lse_min_per_net = torch.zeros(num_valid_nets,
                                      device=self.pin_pos_x.device)

        # calculate LSE-max and LSE-min for each net
        pin_offset = 0
        for i in range(num_valid_nets):
            num_pins = valid_pin_counts[i].item()
            z_net = all_z_net[pin_offset:pin_offset + num_pins]

            # LSE-max: logsumexp(α * z) / α
            lse_max_per_net[i] = self.lse_max(z_net)

            # LSE-min: -logsumexp(-α * z) / α
            lse_min_per_net[i] = self.lse_min(z_net)

            pin_offset += num_pins

        # calculate cutsize: (1 - lse_min) * lse_max
        valid_cutsizes = (
            1.0 - lse_min_per_net) * lse_max_per_net  # [num_valid_nets]

        # create complete result array (including invalid nets)
        result = torch.zeros(num_nets, device=self.pin_pos_x.device)
        result[valid_mask] = valid_cutsizes

        return result

    def compute_terminal_positions(self, net_indices):
        """
        compute terminal positions at the center of optimal region for cut nets (vectorized version)
        
        Args:
            net_indices: network indices, tensor of shape [num_nets]
        
        Returns:
            terminal_positions: tensor of shape [num_nets, 2] (x, y coordinates)
                               terminal positions for non-cut nets are NaN
            cut_mask: boolean tensor of shape [num_nets], True indicates this net generates a terminal
        """
        if net_indices.numel() == 0:
            return torch.empty((0, 2),
                               device=self.pin_pos_x.device), torch.empty(
                                   0,
                                   dtype=torch.bool,
                                   device=self.pin_pos_x.device)

        num_nets = net_indices.numel()
        terminal_positions = torch.full((num_nets, 2),
                                        float('nan'),
                                        device=self.pin_pos_x.device)

        # get the probability of each cell being assigned to top layer
        z = self.get_z()  # [num_cells]

        # get start and end indices for all nets
        start_indices = self.flat_net2pin_start_map[net_indices]  # [num_nets]
        end_indices = self.flat_net2pin_start_map[net_indices +
                                                  1]  # [num_nets]

        # calculate pin count for each net
        pin_counts = end_indices - start_indices  # [num_nets]
        valid_mask = pin_counts >= 2  # [num_nets]

        if not valid_mask.any():
            cut_mask = torch.zeros(num_nets,
                                   dtype=torch.bool,
                                   device=self.pin_pos_x.device)
            return terminal_positions, cut_mask

        # only process valid nets
        valid_start_indices = start_indices[valid_mask]  # [num_valid_nets]
        valid_end_indices = end_indices[valid_mask]  # [num_valid_nets]
        valid_pin_counts = pin_counts[valid_mask]  # [num_valid_nets]
        valid_net_indices = net_indices[valid_mask]  # [num_valid_nets]
        num_valid_nets = valid_pin_counts.numel()

        # collect all valid net's pin indices
        all_pin_indices = []
        for i in range(num_valid_nets):
            start_idx = valid_start_indices[i].item()
            end_idx = valid_end_indices[i].item()
            all_pin_indices.append(self.flat_net2pin_map[start_idx:end_idx])

        all_pin_indices = torch.cat(all_pin_indices)  # [total_pins]

        # get all pin corresponding node indices and z values
        all_node_indices = self.pin2node_map[all_pin_indices]  # [total_pins]
        all_z_net = z[all_node_indices]  # [total_pins]

        # vectorized cut detection: determine whether each net is cut
        # a net is cut if its pins are not all in top (z > 0.5) or all in bottom (z < 0.5)
        cut_mask = torch.zeros(num_nets,
                               dtype=torch.bool,
                               device=self.pin_pos_x.device)

        # precompute pin offsets for all valid nets
        pin_offsets = torch.cumsum(torch.cat([
            torch.tensor([0], device=valid_pin_counts.device),
            valid_pin_counts[:-1]
        ]),
                                   dim=0)  # [num_valid_nets]

        # vectorized cut detection by grouping nets with same pin count
        unique_pin_counts, inverse_indices, counts = torch.unique(
            valid_pin_counts, return_inverse=True, return_counts=True)

        # process each group of nets with same pin count
        for group_idx, pin_count in enumerate(unique_pin_counts):
            # find all nets in this group
            group_mask = inverse_indices == group_idx
            group_net_indices_in_valid = group_mask.nonzero(
                as_tuple=True)[0]  # indices in valid_nets
            num_group_nets = group_net_indices_in_valid.numel()

            if num_group_nets == 0:
                continue

            # get pin offsets for this group
            group_pin_offsets = pin_offsets[
                group_net_indices_in_valid]  # [num_group_nets]

            # create batch indices for all pins in this group
            # shape: [num_group_nets, pin_count]
            batch_indices = group_pin_offsets.unsqueeze(1) + torch.arange(
                pin_count, device=group_pin_offsets.device).unsqueeze(0)

            # get z values for all pins in this group
            z_net_batch = all_z_net[
                batch_indices]  # [num_group_nets, pin_count]

            # determine if all in top (z > 0.5) or all in bottom (z < 0.5) for each net
            all_in_top = (z_net_batch > 0.5).all(dim=1)  # [num_group_nets]
            all_in_bottom = (z_net_batch < 0.5).all(dim=1)  # [num_group_nets]

            # if not all in top and not all in bottom, then it is cut
            group_cut_mask = ~(all_in_top | all_in_bottom)  # [num_group_nets]

            # map back to original net indices
            valid_net_positions = group_net_indices_in_valid[group_cut_mask]
            if valid_net_positions.numel() > 0:
                # find corresponding positions in original net_indices
                original_positions = valid_mask.nonzero(
                    as_tuple=True)[0][valid_net_positions]
                cut_mask[original_positions] = True

        if not cut_mask.any():
            return terminal_positions, cut_mask

        # for cut nets, compute their terminal positions (center of optimal region)
        # optimal region is defined as the center of the bounding box of all pin positions for this net
        cut_valid_mask = valid_mask & cut_mask  # [num_nets]
        cut_valid_positions = cut_valid_mask.nonzero(
            as_tuple=True)[0]  # positions in net_indices

        if cut_valid_positions.numel() == 0:
            return terminal_positions, cut_mask

        # get cut nets' start and end indices
        cut_start_indices = start_indices[cut_valid_mask]  # [num_cut_nets]
        cut_end_indices = end_indices[cut_valid_mask]  # [num_cut_nets]
        cut_pin_counts = pin_counts[cut_valid_mask]  # [num_cut_nets]
        num_cut_nets = cut_pin_counts.numel()

        # collect all cut net's pin indices
        all_cut_pin_indices = []
        for i in range(num_cut_nets):
            start_idx = cut_start_indices[i].item()
            end_idx = cut_end_indices[i].item()
            all_cut_pin_indices.append(
                self.flat_net2pin_map[start_idx:end_idx])

        all_cut_pin_indices = torch.cat(
            all_cut_pin_indices)  # [total_cut_pins]

        # get all cut pin positions
        all_cut_pin_x = self.pin_pos_x[all_cut_pin_indices]  # [total_cut_pins]
        all_cut_pin_y = self.pin_pos_y[all_cut_pin_indices]  # [total_cut_pins]

        # vectorized terminal position calculation by grouping nets with same pin count
        cut_pin_offsets = torch.cumsum(torch.cat([
            torch.tensor([0], device=cut_pin_counts.device),
            cut_pin_counts[:-1]
        ]),
                                       dim=0)  # [num_cut_nets]

        # group cut nets by pin count
        cut_unique_pin_counts, cut_inverse_indices, cut_counts = torch.unique(
            cut_pin_counts, return_inverse=True, return_counts=True)

        # process each group of cut nets with same pin count
        for group_idx, pin_count in enumerate(cut_unique_pin_counts):
            # find all cut nets in this group
            group_mask = cut_inverse_indices == group_idx
            group_net_indices_in_cut = group_mask.nonzero(
                as_tuple=True)[0]  # indices in cut_nets
            num_group_nets = group_net_indices_in_cut.numel()

            if num_group_nets == 0:
                continue

            # get pin offsets for this group
            group_pin_offsets = cut_pin_offsets[
                group_net_indices_in_cut]  # [num_group_nets]

            # create batch indices for all pins in this group
            batch_indices = group_pin_offsets.unsqueeze(1) + torch.arange(
                pin_count, device=group_pin_offsets.device).unsqueeze(0)

            # get pin positions for all pins in this group
            pin_x_batch = all_cut_pin_x[
                batch_indices]  # [num_group_nets, pin_count]
            pin_y_batch = all_cut_pin_y[
                batch_indices]  # [num_group_nets, pin_count]

            # compute center of bounding box for each net (vectorized)
            center_x_batch = (pin_x_batch.max(dim=1)[0] + pin_x_batch.min(
                dim=1)[0]) / 2.0  # [num_group_nets]
            center_y_batch = (pin_y_batch.max(dim=1)[0] + pin_y_batch.min(
                dim=1)[0]) / 2.0  # [num_group_nets]

            # map back to original net positions
            original_positions = cut_valid_positions[group_net_indices_in_cut]
            terminal_positions[original_positions, 0] = center_x_batch
            terminal_positions[original_positions, 1] = center_y_batch

        return terminal_positions, cut_mask

    def detect_terminal_overlaps(self,
                                 terminal_positions,
                                 cut_mask,
                                 overlap_threshold=1500):
        """
        detect whether terminals overlap
        
        Args:
            terminal_positions: tensor of shape [num_nets, 2] (x, y coordinates)
            cut_mask: boolean tensor of shape [num_nets], True indicates this net generates a terminal
            overlap_threshold: overlap threshold, terminals with distance less than this value are considered overlapping
        
        Returns:
            overlap_groups: list of lists, each sublist contains overlapping net indices
            overlap_mask: boolean tensor of shape [num_nets], True indicates this net's terminal overlaps with other nets' terminals
        """
        num_nets = terminal_positions.shape[0]
        overlap_mask = torch.zeros(num_nets,
                                   dtype=torch.bool,
                                   device=terminal_positions.device)
        overlap_groups = []

        # only consider nets that generate terminals
        valid_indices = torch.where(cut_mask)[0]  # [num_valid_nets]

        if valid_indices.numel() < 2:
            return overlap_groups, overlap_mask

        # get valid terminal positions
        valid_positions = terminal_positions[
            valid_indices]  # [num_valid_nets, 2]

        # compute distances between all terminals
        # use Euclidean distance
        positions_expanded_1 = valid_positions.unsqueeze(
            1)  # [num_valid_nets, 1, 2]
        positions_expanded_2 = valid_positions.unsqueeze(
            0)  # [1, num_valid_nets, 2]
        distances = torch.norm(positions_expanded_1 - positions_expanded_2,
                               dim=2)  # [num_valid_nets, num_valid_nets]

        # find overlapping terminals (distance less than threshold, and not itself)
        num_valid_nets = valid_indices.numel()
        overlap_matrix = (distances < overlap_threshold) & (
            distances > 0)  # [num_valid_nets, num_valid_nets]

        # use union-find or simple method to find overlap groups
        visited = torch.zeros(num_valid_nets,
                              dtype=torch.bool,
                              device=terminal_positions.device)

        for i in range(num_valid_nets):
            if visited[i]:
                continue

            # find all terminals overlapping with i
            overlaps_with_i = overlap_matrix[i] | overlap_matrix[:, i]
            if overlaps_with_i.any():
                # create an overlap group
                group_indices = torch.where(overlaps_with_i)[0]
                group_original_indices = valid_indices[group_indices].tolist()
                overlap_groups.append(group_original_indices)

                # mark as visited
                visited[group_indices] = True

                # update overlap_mask
                overlap_mask[valid_indices[group_indices]] = True

        return overlap_groups, overlap_mask

    def compute_cutsize_loss(self,
                             selected_nets=None,
                             cutsize_net_weights=None,
                             handle_terminal_overlap=False,
                             overlap_threshold=500,
                             overlap_weight_penalty=1.0):
        """
        calculate total cutsize loss (only for selected nets)
        
        L_cut = Σ_{n∈selected_nets} w(n) * cutsize(n)
        
        if handle_terminal_overlap=True, it will detect terminal overlaps and increase the weight of one of the overlapping terminals to eliminate cutsize
        
        Args:
            selected_nets: network indices to apply cutsize constraint, tensor or list
                           if None, calculate for all nets
            cutsize_net_weights: weights for each selected net, tensor or list
                                 if None, use self.net_weights corresponding to the selected nets
                                 if selected_nets is None, use self.net_weights
            handle_terminal_overlap: whether to handle terminal overlap, default True
            overlap_threshold: distance threshold for terminal overlap
            overlap_weight_penalty: multiplier to increase net weight when terminals overlap
        
        Returns:
            total_cutsize: total cutsize loss, scalar tensor
        """
        if selected_nets is None:
            # if not specified, calculate for all nets (may be slow)
            selected_nets = list(range(self.num_nets))

        if isinstance(selected_nets, list):
            selected_nets = torch.tensor(selected_nets,
                                         dtype=torch.long,
                                         device=self.pin_pos_x.device)

        # ensure selected_nets is in valid range
        if selected_nets.max() >= self.num_nets or selected_nets.min() < 0:
            raise ValueError(
                f"selected_nets index out of range [0, {self.num_nets-1}]")

        # get weights
        if cutsize_net_weights is None:
            # use weights from self.net_weights corresponding to the selected nets
            weights = self.net_weights[selected_nets].clone()
        else:
            # use user-provided weights
            if isinstance(cutsize_net_weights, list):
                cutsize_net_weights = torch.tensor(
                    cutsize_net_weights,
                    dtype=torch.float32,
                    device=self.pin_pos_x.device)
            if cutsize_net_weights.numel() != selected_nets.numel():
                raise ValueError(
                    f"cutsize_net_weights length ({cutsize_net_weights.numel()}) "
                    f"must match selected_nets length ({selected_nets.numel()})"
                )
            weights = cutsize_net_weights.clone()

        weights.fill_(1.0)

        # handle terminal overlap
        if handle_terminal_overlap:
            # compute terminal positions
            terminal_positions, cut_mask = self.compute_terminal_positions(
                selected_nets)
            overlap_groups, overlap_mask = self.detect_terminal_overlaps(
                terminal_positions,
                cut_mask,
                overlap_threshold=overlap_threshold)

            if overlap_groups:
                # concatenate all groups into a single tensor
                all_overlapping_indices = torch.cat([
                    torch.tensor(group,
                                 dtype=torch.long,
                                 device=selected_nets.device)
                    for group in overlap_groups if len(group) > 0
                ])
                # remove duplicates using torch.unique
                overlapping_net_indices = torch.unique(all_overlapping_indices)

                # expand selected_nets to [num_selected_nets, 1] and overlapping_net_indices to [1, num_overlapping_nets]
                # then compare to find matches
                selected_nets_expanded = selected_nets.unsqueeze(
                    1)  # [num_selected_nets, 1]
                overlapping_expanded = overlapping_net_indices.unsqueeze(
                    0)  # [1, num_overlapping_nets]

                # find matches: [num_selected_nets, num_overlapping_nets]
                matches = (selected_nets_expanded == overlapping_expanded)
                matching_positions = matches.any(dim=1)  # [num_selected_nets]
                weights[matching_positions] *= overlap_weight_penalty

        # vectorized calculation of cutsize for all selected nets
        cutsizes = self.compute_cutsize_batch(
            selected_nets)  # [num_selected_nets]

        total_cutsize = (weights * cutsizes).sum()

        return total_cutsize

    def compute_balance_loss(self):
        """
        calculate balance loss
        if the density of a bin exceeds half of the bin area, add relu penalty
        
        Returns:
            balance_loss: balance loss, scalar tensor
        """

        z = self.get_z()
        top_z = z
        bottom_z = 1 - z

        threshold_factor = 0.6

        def compute_density_map(partition_z, num_bin_x, num_bin_y):

            bin_size_x = self.x_range / num_bin_x
            bin_size_y = self.y_range / num_bin_y

            # calculate the area of each bin
            node_area_map = torch.zeros(num_bin_x, num_bin_y, device=z.device)
            density_map = torch.zeros(num_bin_x, num_bin_y, device=z.device)
            node_x_min = self.node_x.min()
            node_y_min = self.node_y.min()

            # compute bin indices for all nodes at once (vectorized)
            x_idx = ((self.node_x - node_x_min) / bin_size_x).long()
            y_idx = ((self.node_y - node_y_min) / bin_size_y).long()
            x_idx = torch.clamp(x_idx, 0, num_bin_x - 1)
            y_idx = torch.clamp(y_idx, 0, num_bin_y - 1)

            # compute node areas and weighted areas (vectorized)
            node_areas = self.node_size_x * self.node_size_y  # [num_nodes]
            weighted_areas = node_areas * partition_z  # [num_nodes]

            # use index_add_ to accumulate values (vectorized)
            # index_add_ requires 1D indices, so we flatten the 2D indices
            # Convert 2D indices (x_idx, y_idx) to 1D linear indices
            linear_indices = x_idx * num_bin_y + y_idx  # [num_nodes]

            # flatten density maps for index_add_
            density_map_flat = density_map.flatten()  # [num_bin_x * num_bin_y]
            node_area_map_flat = node_area_map.flatten(
            )  # [num_bin_x * num_bin_y]
            density_map_flat.index_add_(0, linear_indices, weighted_areas)
            node_area_map_flat.index_add_(0, linear_indices, node_areas)
            density_map = density_map_flat.view(num_bin_x, num_bin_y)
            node_area_map = node_area_map_flat.view(num_bin_x, num_bin_y)

            return density_map, node_area_map

        top_density_map, node_area_map = compute_density_map(top_z, 10, 10)
        bottom_density_map, _ = compute_density_map(bottom_z, 10, 10)

        balance_loss = torch.relu(top_density_map - node_area_map*threshold_factor).sum() + \
                       torch.relu(bottom_density_map - node_area_map*threshold_factor).sum()
        # balance_loss = torch.relu(top_density_map - node_area_map*0.329).sum() + \
        #                torch.relu(bottom_density_map - node_area_map*0.671).sum()

        return balance_loss

    def forward(self,
                lambda_wl=1.0,
                lambda_cut=0.0,
                lambda_balance=0.0,
                selected_nets=None,
                cutsize_net_weights=None,
                return_debug_info=False):
        """
        calculate total loss (HPWL + Cutsize + Balance)
        L_WL = Σ_e (HPWL_top_e + HPWL_bottom_e) * weight_e
        L_cut = Σ_{n∈selected_nets} w(n) * cutsize(n)
        L_balance = balance loss (penalizes density exceeding half bin area)
        L_total = λ_WL * L_WL + λ_cut * L_cut + λ_balance * L_balance
        
        Args:
            lambda_wl: weight of HPWL loss, default 1.0
            lambda_cut: weight of cutsize loss, default 0.0
            lambda_balance: weight of balance loss, default 0.0
            selected_nets: indices of nets to apply cutsize constraint, tensor or list
                           only used when lambda_cut > 0
            cutsize_net_weights: weights of each selected net, tensor or list
                                 if None, use self.net_weights corresponding to the selected nets
            return_debug_info: whether to return debug information, default False
        
        Returns:
            total_loss: total loss, scalar tensor
            if return_debug_info:
                debug_info: dict containing:
                    - 'L_WL': total HPWL loss
                    - 'L_cut': total cutsize loss
                    - 'L_balance': total balance loss
                    - 'L_total': total loss
                return (total_loss, debug_info) tuple
            else:
                return total_loss
        """
        # batch calculation of HPWL for all nets (vectorized, much faster)
        all_net_indices = torch.arange(self.num_nets,
                                       device=self.pin_pos_x.device)
        hpwl_top_all, hpwl_bottom_all = self.compute_hpwl_batch(
            all_net_indices, layer='both')

        # weighted accumulate: Σ_e (HPWL_top_e + HPWL_bottom_e) * weight_e
        total_hpwl = (self.net_weights *
                      (hpwl_top_all + hpwl_bottom_all)).sum()

        cutsize_loss = torch.tensor(0.0, device=self.pin_pos_x.device)
        if lambda_cut > 0:
            cutsize_loss = self.compute_cutsize_loss(
                selected_nets=selected_nets,
                cutsize_net_weights=cutsize_net_weights)

        balance_loss = torch.tensor(0.0, device=self.pin_pos_x.device)
        if lambda_balance > 0:
            balance_loss = self.compute_balance_loss()

        total_loss = (lambda_wl * total_hpwl + lambda_cut * cutsize_loss +
                      lambda_balance * balance_loss)

        # if not return debug information, return total loss
        if not return_debug_info:
            return total_loss

        # return total loss and debug information
        debug_info = {
            'L_WL': total_hpwl.item(),
            'L_cut': cutsize_loss.item() if lambda_cut > 0 else 0.0,
            'L_balance': balance_loss.item() if lambda_balance > 0 else 0.0,
            'L_total': total_loss.item()
        }
        return total_loss, debug_info

    def get_binary_assignment(self, threshold=0.5):
        """
        get binary assignment
        threshold for binary assignment, default 0.5
            
        Returns:
            binary_z: binary assignment, 1 for top, 0 for bottom
        """
        z = self.get_z()
        return (z > threshold).to(torch.int32)

    def get_assignment_stats(self):
        """
        get assignment statistics for debugging
        
        Returns:
            dict: contains z statistics
        """
        z = self.get_z()
        return {
            'z_mean': z.mean().item(),
            'z_std': z.std().item(),
            'z_min': z.min().item(),
            'z_max': z.max().item(),
            'top_cells': (z > 0.5).sum().item(),
            'bottom_cells': (z <= 0.5).sum().item()
        }


def visualize_z_single(x_coords,
                       y_coords,
                       z_values,
                       iteration,
                       save_path=None,
                       node_size_x=None,
                       node_size_y=None):
    """
    visualize z values over xy for a single iteration
    
    Args:
        x_coords: x coordinates, shape [num_cells] tensor or numpy array
        y_coords: y coordinates, shape [num_cells] tensor or numpy array
        z_values: z values, shape [num_cells] tensor or numpy array
        iteration: current iteration number
        save_path: save path (optional)
        node_size_x: x size of each node, shape [num_cells] tensor or numpy array (optional)
        node_size_y: y size of each node, shape [num_cells] tensor or numpy array (optional)
    """
    # convert to numpy array
    if isinstance(x_coords, torch.Tensor):
        x_coords = x_coords.detach().cpu().numpy()
    if isinstance(y_coords, torch.Tensor):
        y_coords = y_coords.detach().cpu().numpy()
    if isinstance(z_values, torch.Tensor):
        z_values = z_values.detach().cpu().numpy()
    if node_size_x is not None and isinstance(node_size_x, torch.Tensor):
        node_size_x = node_size_x.detach().cpu().numpy()
    if node_size_y is not None and isinstance(node_size_y, torch.Tensor):
        node_size_y = node_size_y.detach().cpu().numpy()

    # create single 3D plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # get colormap
    cmap = plt.cm.get_cmap('RdYlBu_r')

    # normalize z values for color mapping
    z_normalized = (z_values - z_values.min()) / (z_values.max() -
                                                  z_values.min() + 1e-8)

    # if node sizes are provided, draw rectangles; otherwise draw scatter points
    if node_size_x is not None and node_size_y is not None:
        # draw 2D rectangular planes for each node at their z height
        all_faces = []
        all_colors = []

        for i in range(len(x_coords)):
            x = x_coords[i]
            y = y_coords[i]
            z = z_values[i]
            size_x = node_size_x[i]
            size_y = node_size_y[i]

            # calculate rectangle corners (centered at x, y)
            x_min = x - size_x / 2
            x_max = x + size_x / 2
            y_min = y - size_y / 2
            y_max = y + size_y / 2

            # create a single 2D rectangular plane at z height (parallel to XY plane)
            rect_face = [[x_min, y_min, z], [x_max, y_min, z],
                         [x_max, y_max, z], [x_min, y_max, z]]

            all_faces.append(rect_face)

            # get color for this node based on z value
            color = cmap(z_normalized[i])
            all_colors.append(color)

        # create Poly3DCollection
        collection = Poly3DCollection(all_faces,
                                      facecolors=all_colors,
                                      edgecolors='k',
                                      linewidths=0.2,
                                      alpha=0.7)
        ax.add_collection3d(collection)

        # create a mappable for colorbar
        norm = Normalize(vmin=z_values.min(), vmax=z_values.max())
        scatter = ScalarMappable(norm=norm, cmap='RdYlBu_r')
        scatter.set_array([])
    else:
        # fallback to scatter plot if node sizes not provided
        scatter = ax.scatter(x_coords,
                             y_coords,
                             z_values,
                             c=z_values,
                             cmap='RdYlBu_r',
                             s=20,
                             alpha=0.6,
                             edgecolors='k',
                             linewidths=0.3)

    ax.set_xlabel('X Coordinate', fontsize=12)
    ax.set_ylabel('Y Coordinate', fontsize=12)
    ax.set_zlabel('Z Value (Soft Assignment)', fontsize=12)
    ax.set_title(f'Iteration {iteration} - Z Distribution over XY',
                 fontsize=14,
                 fontweight='bold')
    ax.set_zlim(0, 1)
    plt.colorbar(scatter, ax=ax, shrink=0.8, label='Z Value')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"visualization saved to: {save_path}")

    plt.close()


def main():
    """
    main function for differentiable partitioner
    """
    print("=" * 60)
    print("Differentiable 3D Partitioner")
    print("=" * 60)

    # load real circuit data
    print("\n1. Load real circuit data...")
    # num_cells = 8
    # num_nets = 1
    num_cells = 2735
    num_nets = 2644
    # num_cells = 44764
    # num_nets = 44360
    # load data from .pt files and detach from computation graph to avoid backward errors
    node_pos = torch.load("case2_hidden_2d_placement.pt").detach()
    # node_pos = torch.tensor([5, 10, 15, 20, 0, 0, 0, 0])
    # node_pos = torch.tensor(
    #     [5, 20, 25, 30, 5, 10, 15, 30, 0, 0, 0, 0, 10, 10, 10, 10])
    pin_pos = torch.load("case2_hidden_2d_pinpos.pt").detach()
    # pin_pos = torch.tensor([5, 10, 15, 20, 0, 0, 0, 0])
    # pin_pos = torch.tensor(
    #     [5, 20, 25, 30, 5, 10, 15, 30, 0, 0, 0, 0, 10, 10, 10, 10])
    flat_net2pin_map = torch.load("case2_hidden_flat_net2pin_map.pt").detach()
    # flat_net2pin_map = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7])
    # flat_net2pin_map = torch.tensor([0, 1, 2, 3])
    flat_net2pin_start_map = torch.load(
        "case2_hidden_flat_net2pin_start_map.pt").detach()
    # flat_net2pin_start_map = torch.tensor([0, 4, 8])
    # flat_net2pin_start_map = torch.tensor([0, 4])
    pin2node_map = torch.load("case2_hidden_pin2node_map.pt").detach()
    # pin2node_map = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7])
    # pin2node_map = torch.tensor([0, 1, 2, 3])

    node_x = node_pos[:num_cells]
    node_y = node_pos[node_pos.numel() // 2:node_pos.numel() // 2 + num_cells]
    pin_pos_x = pin_pos[:pin2node_map.numel()]
    pin_pos_y = pin_pos[pin2node_map.numel():]
    node_size_x = torch.load("case2h-node_size_x.pt").detach()
    node_size_y = torch.load("case2h-node_size_y.pt").detach()

    # verify data shape
    num_pins = pin2node_map.numel()

    print(f"   - Number of cells: {num_cells}")
    print(f"   - Number of nets: {num_nets}")
    print(f"   - Number of pins: {num_pins}")
    print(
        f"   - Coordinate range: x=[{pin_pos_x.min():.2f}, {pin_pos_x.max():.2f}], "
        f"y=[{pin_pos_y.min():.2f}, {pin_pos_y.max():.2f}]")

    # calculate number of pins per node for verification
    unique_nodes, node_pin_counts = torch.unique(pin2node_map,
                                                 return_counts=True)
    print(
        f"   - Average number of pins per node: {node_pin_counts.float().mean():.2f}"
    )
    print(
        f"   - Maximum number of pins per node: {node_pin_counts.max().item()}"
    )

    # initialize LSE partitioner
    print("\n2. Initialize LSE partitioner...")
    model = LSEPartitioner(
        num_cells=num_cells,
        flat_net2pin_map=flat_net2pin_map,
        flat_net2pin_start_map=flat_net2pin_start_map,
        pin2node_map=pin2node_map,
        pin_pos_x=pin_pos_x,
        pin_pos_y=pin_pos_y,
        node_x=node_x,
        node_y=node_y,
        node_size_x=node_size_x,
        node_size_y=node_size_y,
        alpha=1.0  # initial alpha value
    )
    print(f"   - Initial alpha: {model.alpha}")
    print(
        f"   - Number of trainable parameters: {sum(p.numel() for p in model.parameters())}"
    )

    # configure cutsize loss
    use_cutsize_loss = False
    lambda_cut_start = 10000.0
    lambda_cut_end = 10000.0
    selected_nets_for_cutsize = None  # None means apply cutsize constraint to all nets
    cutsize_net_weights = None  # None means use self.net_weights corresponding to the selected nets
    # can specify specific network subset, e.g.:
    # selected_nets_for_cutsize = torch.tensor([0, 1, 2, 10, 20, 50], dtype=torch.long)

    # configure balance loss
    use_balance_loss = True
    lambda_balance_start = 1.0
    lambda_balance_end = 1.0

    # print initial state
    print("\n3. Initial state:")
    use_debug_info = use_cutsize_loss or use_balance_loss
    if use_debug_info:
        initial_loss, debug_info = model(
            lambda_wl=1.0,
            lambda_cut=lambda_cut_start if use_cutsize_loss else 0.0,
            lambda_balance=lambda_balance_start if use_balance_loss else 0.0,
            selected_nets=selected_nets_for_cutsize,
            cutsize_net_weights=cutsize_net_weights,
            return_debug_info=True)
        print(f"   - Initial total loss: {initial_loss.item():.4f}")
        print(f"   - Initial HPWL: {debug_info['L_WL']:.4f}")
        if use_cutsize_loss:
            print(f"   - Initial cutsize: {debug_info['L_cut']:.4f}")
        if use_balance_loss:
            print(f"   - Initial balance: {debug_info['L_balance']:.4f}")
    else:
        initial_loss = model()
        print(f"   - Initial total HPWL: {initial_loss.item():.4f}")
    stats = model.get_assignment_stats()
    print(
        f"   - z statistics: mean={stats['z_mean']:.4f}, std={stats['z_std']:.4f}"
    )
    print(
        f"   - Number of top cells: {stats['top_cells']}, number of bottom cells: {stats['bottom_cells']}"
    )

    # set optimizer
    print("\n4. Set optimizer...")
    learning_rate = 0.1
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    print(f"   - Optimizer: Adam")
    print(f"   - Learning rate: {learning_rate}")

    # training parameters
    num_iterations = 5000
    alpha_start = 1.0
    alpha_end = 20.0
    alpha_schedule = np.linspace(alpha_start, alpha_end, num_iterations)

    # lambda_cut schedule (exponentially increase cutsize loss weight)
    if use_cutsize_loss:
        if lambda_cut_start <= 0:
            # if start value is 0 or negative, use a small value as start value, then exponential growth
            eps = 1e-6
            lambda_cut_schedule = np.logspace(
                np.log10(max(eps, lambda_cut_start + eps)),
                np.log10(lambda_cut_end), num_iterations)
            if lambda_cut_start == 0:
                lambda_cut_schedule[0] = 0.0
        else:
            lambda_cut_schedule = np.logspace(np.log10(lambda_cut_start),
                                              np.log10(lambda_cut_end),
                                              num_iterations)
    else:
        lambda_cut_schedule = None

    # lambda_balance schedule (linear schedule for balance loss weight)
    if use_balance_loss:
        lambda_balance_schedule = np.linspace(lambda_balance_start,
                                              lambda_balance_end,
                                              num_iterations)
    else:
        lambda_balance_schedule = None

    print(f"\n5. Start training ({num_iterations} iterations)...")
    print(f"   - Alpha schedule: {alpha_start} → {alpha_end}")
    if use_cutsize_loss:
        num_selected = num_nets if selected_nets_for_cutsize is None else len(
            selected_nets_for_cutsize)
        print(
            f"   - Cutsize loss enabled: λ_cut exponential schedule {lambda_cut_start} → {lambda_cut_end}, applied to {num_selected} nets"
        )
    else:
        print(f"   - Cutsize loss disabled")
    if use_balance_loss:
        print(
            f"   - Balance loss enabled: λ_balance linear schedule {lambda_balance_start} → {lambda_balance_end}"
        )
    else:
        print(f"   - Balance loss disabled")
    print("-" * 60)

    # create directory for saving visualizations
    save_dir = os.path.join(os.path.dirname(__file__), 'visualizations')
    os.makedirs(save_dir, exist_ok=True)

    # initialize lists to store training history
    history_loss = []
    history_hpwl = []
    history_cut = []
    history_balance = []
    history_iterations = []

    # training loop
    for iteration in range(num_iterations):
        # update current iteration (used to switch between sigmoid and gumbel_softmax)
        model.current_iteration = iteration
        model.alpha = alpha_schedule[iteration]

        # update lambda_cut (gradually increase cutsize loss weight)
        if use_cutsize_loss:
            lambda_cut = lambda_cut_schedule[iteration]
        else:
            lambda_cut = 0.0

        # update lambda_balance (gradually change balance loss weight)
        if use_balance_loss:
            lambda_balance = lambda_balance_schedule[iteration]
        else:
            lambda_balance = 0.0

        if use_debug_info:
            loss, debug_info = model(lambda_wl=1.0,
                                     lambda_cut=lambda_cut,
                                     lambda_balance=lambda_balance,
                                     selected_nets=selected_nets_for_cutsize,
                                     cutsize_net_weights=cutsize_net_weights,
                                     return_debug_info=True)
            # record training history
            history_iterations.append(iteration + 1)
            history_loss.append(loss.item())
            history_hpwl.append(debug_info['L_WL'])
            history_cut.append(
                debug_info.get('L_cut', 0.0) if use_cutsize_loss else 0.0)
            history_balance.append(
                debug_info.get('L_balance', 0.0) if use_balance_loss else 0.0)
        else:
            loss = model(lambda_balance=lambda_balance)
            # record training history (only loss and hpwl available)
            history_iterations.append(iteration + 1)
            history_loss.append(loss.item())
            history_hpwl.append(
                loss.item())  # when no debug_info, loss is HPWL
            history_cut.append(0.0)
            history_balance.append(0.0)

        # backward propagation
        optimizer.zero_grad()
        loss.backward()

        # calculate gradient statistics (for debugging)
        if model.t.grad is not None:
            t_grad_norm = model.t.grad.norm().item()
            # check if there are NaN or Inf gradients
            if torch.isnan(model.t.grad).any() or torch.isinf(
                    model.t.grad).any():
                print(
                    f"Warning: iteration {iteration+1} detected NaN/Inf gradients"
                )
        else:
            t_grad_norm = 0.0

        # update parameters
        optimizer.step()

        # print and visualize every 10 iterations
        if (iteration + 1) % 10 == 0 or iteration == 0:
            z = model.get_z()
            dz_dt_norm = (z * (1 - z)).norm().item()
            stats = model.get_assignment_stats()

            if use_debug_info:
                log_str = f"Iter {iteration+1:4d} | Loss: {loss.item():8.2f} | HPWL: {debug_info['L_WL']:8.2f}"
                if use_cutsize_loss:
                    log_str += f" | Cut: {debug_info['L_cut']:6.4f} | λ_cut: {lambda_cut:6.4f}"
                if use_balance_loss:
                    log_str += f" | Balance: {debug_info['L_balance']:6.4f} | λ_balance: {lambda_balance:6.4f}"
                log_str += f" | Alpha: {model.alpha:5.2f} | ||dt||: {t_grad_norm:6.4f} | Top: {stats['top_cells']:3d} | Bottom: {stats['bottom_cells']:3d}"
                print(log_str)
            else:
                print(f"Iter {iteration+1:4d} | "
                      f"Loss: {loss.item():8.2f} | "
                      f"Alpha: {model.alpha:5.2f} | "
                      f"||dt||: {t_grad_norm:6.4f} | "
                      f"||dz/dt||: {dz_dt_norm:6.4f} | "
                      f"Top: {stats['top_cells']:3d} | "
                      f"Bottom: {stats['bottom_cells']:3d}")

        if (iteration + 1) % 50 == 0 or iteration == 0:
            save_path = os.path.join(
                save_dir, f'z_evolution_iter_{iteration+1:04d}.png')
            z = model.get_z()
            visualize_z_single(node_x,
                               node_y,
                               z,
                               iteration + 1,
                               save_path,
                               node_size_x=node_size_x,
                               node_size_y=node_size_y)

            fig, axes = plt.subplots(2, 2, figsize=(14, 10))

            # Plot Loss
            axes[0, 0].plot(history_iterations,
                            history_loss,
                            'b-',
                            linewidth=2,
                            label='Total Loss')
            axes[0, 0].set_xlabel('Iteration', fontsize=12)
            axes[0, 0].set_ylabel('Loss', fontsize=12)
            axes[0, 0].set_title('Training Loss',
                                 fontsize=14,
                                 fontweight='bold')
            axes[0, 0].grid(True, alpha=0.3)
            axes[0, 0].legend(fontsize=10)

            # Plot HPWL
            axes[0, 1].plot(history_iterations,
                            history_hpwl,
                            'g-',
                            linewidth=2,
                            label='HPWL')
            axes[0, 1].set_xlabel('Iteration', fontsize=12)
            axes[0, 1].set_ylabel('HPWL', fontsize=12)
            axes[0, 1].set_title('Half-Perimeter Wire Length',
                                 fontsize=14,
                                 fontweight='bold')
            axes[0, 1].grid(True, alpha=0.3)
            axes[0, 1].legend(fontsize=10)

            # Plot Cutsize
            if use_cutsize_loss and any(v > 0 for v in history_cut):
                axes[1, 0].plot(history_iterations,
                                history_cut,
                                'r-',
                                linewidth=2,
                                label='Cutsize Loss')
                axes[1, 0].set_xlabel('Iteration', fontsize=12)
                axes[1, 0].set_ylabel('Cutsize Loss', fontsize=12)
                axes[1, 0].set_title('Cutsize Loss',
                                     fontsize=14,
                                     fontweight='bold')
                axes[1, 0].grid(True, alpha=0.3)
                axes[1, 0].legend(fontsize=10)
            else:
                axes[1, 0].text(0.5,
                                0.5,
                                'Cutsize Loss\nNot Enabled',
                                ha='center',
                                va='center',
                                fontsize=12,
                                transform=axes[1, 0].transAxes)
                axes[1, 0].set_title('Cutsize Loss',
                                     fontsize=14,
                                     fontweight='bold')

            # Plot Balance
            if use_balance_loss and any(v > 0 for v in history_balance):
                axes[1, 1].plot(history_iterations,
                                history_balance,
                                'm-',
                                linewidth=2,
                                label='Balance Loss')
                axes[1, 1].set_xlabel('Iteration', fontsize=12)
                axes[1, 1].set_ylabel('Balance Loss', fontsize=12)
                axes[1, 1].set_title('Balance Loss',
                                     fontsize=14,
                                     fontweight='bold')
                axes[1, 1].grid(True, alpha=0.3)
                axes[1, 1].legend(fontsize=10)
            else:
                axes[1, 1].text(0.5,
                                0.5,
                                'Balance Loss\nNot Enabled',
                                ha='center',
                                va='center',
                                fontsize=12,
                                transform=axes[1, 1].transAxes)
                axes[1, 1].set_title('Balance Loss',
                                     fontsize=14,
                                     fontweight='bold')

            plt.tight_layout()
            curve_save_path = os.path.join(save_dir, 'training_curves.png')
            plt.savefig(curve_save_path, dpi=150, bbox_inches='tight')
            print(f"   Training curves saved to: {curve_save_path}")
            plt.close()

    print("-" * 60)

    # final results
    print("\n6. Training completed, final results:")
    if use_debug_info:
        final_loss, final_debug_info = model(
            lambda_wl=1.0,
            lambda_cut=lambda_cut_end if use_cutsize_loss else 0.0,
            lambda_balance=lambda_balance_end if use_balance_loss else 0.0,
            selected_nets=selected_nets_for_cutsize,
            cutsize_net_weights=cutsize_net_weights,
            return_debug_info=True)
        print(f"   - Final total loss: {final_loss.item():.4f}")
        print(f"   - Final HPWL: {final_debug_info['L_WL']:.4f}")
        if use_cutsize_loss:
            print(f"   - Final cutsize: {final_debug_info['L_cut']:.4f}")
        if use_balance_loss:
            print(f"   - Final balance: {final_debug_info['L_balance']:.4f}")
    else:
        final_loss = model(
            lambda_balance=lambda_balance_end if use_balance_loss else 0.0)
        print(f"   - Final total HPWL: {final_loss.item():.4f}")

    stats = model.get_assignment_stats()
    print(
        f"   - z statistics: mean={stats['z_mean']:.4f}, std={stats['z_std']:.4f}, "
        f"min={stats['z_min']:.4f}, max={stats['z_max']:.4f}")
    print(
        f"   - Number of top cells: {stats['top_cells']}, number of bottom cells: {stats['bottom_cells']}"
    )

    # get binary assignment
    binary_z = model.get_binary_assignment()
    print(f"\n7. Binary assignment (threshold=0.5):")
    print(f"   - Number of top cells: {binary_z.sum().item()}")
    print(f"   - Number of bottom cells: {(1 - binary_z).sum().item()}")

    z = model.get_z()
    print(f"\n8. Example soft assignment values (first 10 cells):")
    for i in range(min(10, num_cells)):
        print(
            f"   Cell {i:3d} (x={node_x[i].item():.2f}, y={node_y[i].item():.2f}): z={z[i].item():.4f} → {'Top' if z[i] > 0.5 else 'Bottom'}"
        )

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)
    torch.save(binary_z, "case2_hidden_2d_binary_assignment.pt")


if __name__ == "__main__":
    main()
