'''
Author: JeanneWillis hi@jeannewillis.cn
Date: 2025-11-14 16:03:37
LastEditors: JeanneWillis hi@jeannewillis.cn
LastEditTime: 2025-11-23 18:28:32
FilePath: /D2D-placer/placer/tools/differentiable_partitioner/partitioner.py
Description: Differentiable 3D Partitioner based on LogSumExp soft bounding box
'''
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os


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
                 alpha=1.0,
                 net_weights=None):
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

        # trainable pre-activation variable t_i (one for each cell)
        # use small random initialization to avoid all z being 0.5 (symmetric point)
        self.t = nn.Parameter(torch.randn(num_cells) * 0.1)

        # LSE smoothing parameter
        self.alpha = alpha

        # net weights (if not provided, default to all 1)
        if net_weights is None:
            self.register_buffer('net_weights', torch.ones(self.num_nets))
        else:
            self.register_buffer('net_weights', net_weights.detach().clone())

    def get_z(self):
        """
        map pre-activation variable t to soft assignment z = sigmoid(t)
        
        Returns:
            z: shape [num_cells] tensor, representing the probability of each cell being assigned to the top layer
        """
        return torch.sigmoid(self.t)

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

    def compute_hpwl_top(self, net_idx):
        """
        compute soft HPWL of the specified net on the top layer
        using LSE soft bounding box:
        - x_max_top = (1/α) * log_sum_exp(α * z_node * x_pin)
        - x_min_top = -(1/α) * log_sum_exp(-α * z_node * x_pin)
        - y_max_top = (1/α) * log_sum_exp(α * z_node * y_pin)
        - y_min_top = -(1/α) * log_sum_exp(-α * z_node * y_pin)
        - HPWL_top = (x_max_top - x_min_top) + (y_max_top - y_min_top)        
        
        Args:
            net_idx: index of the net
            
        Returns:
            soft HPWL of the specified net on the top layer
        """
        # obtain all pin indices of the specified net
        start_idx = self.flat_net2pin_start_map[net_idx]
        end_idx = self.flat_net2pin_start_map[net_idx + 1]
        pin_indices = self.flat_net2pin_map[start_idx:end_idx]

        if pin_indices.numel() < 2:
            return torch.tensor(0.0, device=self.pin_pos_x.device)

        # get soft assignment z (defined by node)
        z = self.get_z()

        # get node indices for each pin
        node_indices = self.pin2node_map[
            pin_indices]  # shape: [num_pins_in_net]

        # get z value for each pin corresponding to the node
        z_net = z[node_indices]  # shape: [num_pins_in_net]

        # get position of each pin
        x_net = self.pin_pos_x[pin_indices]  # shape: [num_pins_in_net]
        y_net = self.pin_pos_y[pin_indices]  # shape: [num_pins_in_net]

        # compute weighted values: z_node * x_pin and z_node * y_pin
        weighted_x = z_net * x_net  # shape: [num_pins_in_net]
        weighted_y = z_net * y_net  # shape: [num_pins_in_net]

        # use LSE to compute soft bounding box
        x_max_top = self.lse_max(weighted_x)
        x_min_top = self.lse_min(weighted_x)
        y_max_top = self.lse_max(weighted_y)
        y_min_top = self.lse_min(weighted_y)

        # caculate HPWL
        hpwl_x = x_max_top - x_min_top
        hpwl_y = y_max_top - y_min_top
        hpwl_top = hpwl_x + hpwl_y

        return hpwl_top

    def compute_hpwl_bottom(self, net_idx):
        """
        calculate soft HPWL of the specified net on the bottom layer
        using the same LSE formula as the top layer, but using (1 - z_node) instead of z_node
        
        Args:
            net_idx: index of the net
            
        Returns:
            soft HPWL of the specified net on the bottom layer
        """
        # obtain all pin indices of the specified net
        start_idx = self.flat_net2pin_start_map[net_idx]
        end_idx = self.flat_net2pin_start_map[net_idx + 1]
        pin_indices = self.flat_net2pin_map[start_idx:end_idx]

        if pin_indices.numel() < 2:
            return torch.tensor(0.0, device=self.pin_pos_x.device)

        # get soft assignment z (defined by node)
        z = self.get_z()

        # get node indices for each pin
        node_indices = self.pin2node_map[
            pin_indices]  # shape: [num_pins_in_net]

        # use (1 - z_node) to represent bottom assignment
        z_bottom = 1.0 - z[node_indices]  # shape: [num_pins_in_net]

        # get position of each pin
        x_net = self.pin_pos_x.max() - self.pin_pos_x[
            pin_indices]  # shape: [num_pins_in_net]
        y_net = self.pin_pos_y.max() - self.pin_pos_y[
            pin_indices]  # shape: [num_pins_in_net]

        # compute weighted values: (1 - z_node) * x_pin and (1 - z_node) * y_pin
        weighted_x = z_bottom * x_net
        weighted_y = z_bottom * y_net

        # use LSE to compute soft bounding box
        x_max_bottom = self.lse_max(weighted_x)
        x_min_bottom = self.lse_min(weighted_x)
        y_max_bottom = self.lse_max(weighted_y)
        y_min_bottom = self.lse_min(weighted_y)

        # caculate HPWL
        hpwl_x = x_max_bottom - x_min_bottom
        hpwl_y = y_max_bottom - y_min_bottom
        hpwl_bottom = hpwl_x + hpwl_y

        return hpwl_bottom

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
        # 获取该网的所有pin索引
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

        # filter out nets with less than 2 pins (these nets have HPWL=0)
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

        # get position of each pin
        all_x_net = self.pin_pos_x[all_pin_indices]  # [total_pins]
        all_y_net = self.pin_pos_y[all_pin_indices]  # [total_pins]

        # compute weighted values for top layer: z_node * x_pin and z_node * y_pin
        weighted_x_top = all_z_net * all_x_net  # [total_pins]
        weighted_y_top = all_z_net * all_y_net  # [total_pins]

        # compute weighted values for bottom layer: (1 - z_node) * x_pin and (1 - z_node) * y_pin
        z_bottom = 1.0 - all_z_net  # [total_pins]
        x_max_val = self.pin_pos_x.max()
        y_max_val = self.pin_pos_y.max()
        all_x_net_bottom = x_max_val - all_x_net  # [total_pins]
        all_y_net_bottom = y_max_val - all_y_net  # [total_pins]
        weighted_x_bottom = z_bottom * all_x_net_bottom  # [total_pins]
        weighted_y_bottom = z_bottom * all_y_net_bottom  # [total_pins]

        # vectorized calculation of HPWL for each net
        hpwl_top_per_net = torch.zeros(num_valid_nets,
                                       device=self.pin_pos_x.device)
        hpwl_bottom_per_net = torch.zeros(num_valid_nets,
                                          device=self.pin_pos_x.device)

        # calculate HPWL for each net
        pin_offset = 0
        for i in range(num_valid_nets):
            num_pins = valid_pin_counts[i].item()

            # top layer
            if layer in ['top', 'both']:
                wx_top = weighted_x_top[pin_offset:pin_offset + num_pins]
                wy_top = weighted_y_top[pin_offset:pin_offset + num_pins]
                x_max_top = self.lse_max(wx_top)
                x_min_top = self.lse_min(wx_top)
                y_max_top = self.lse_max(wy_top)
                y_min_top = self.lse_min(wy_top)
                hpwl_top_per_net[i] = (x_max_top - x_min_top) + (y_max_top -
                                                                 y_min_top)

            # bottom layer
            if layer in ['bottom', 'both']:
                wx_bottom = weighted_x_bottom[pin_offset:pin_offset + num_pins]
                wy_bottom = weighted_y_bottom[pin_offset:pin_offset + num_pins]
                x_max_bottom = self.lse_max(wx_bottom)
                x_min_bottom = self.lse_min(wx_bottom)
                y_max_bottom = self.lse_max(wy_bottom)
                y_min_bottom = self.lse_min(wy_bottom)
                hpwl_bottom_per_net[i] = (x_max_bottom - x_min_bottom) + (
                    y_max_bottom - y_min_bottom)

            pin_offset += num_pins

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

    def compute_cutsize_loss(self,
                             selected_nets=None,
                             cutsize_net_weights=None):
        """
        calculate total cutsize loss (only for selected nets)
        
        L_cut = Σ_{n∈selected_nets} w(n) * cutsize(n)
        
        Args:
            selected_nets: network indices to apply cutsize constraint, tensor or list
                           if None, calculate for all nets
            cutsize_net_weights: weights for each selected net, tensor or list
                                 if None, use self.net_weights corresponding to the selected nets
                                 if selected_nets is None, use self.net_weights
        
        Returns:
            total_cutsize: 总cutsize损失，标量tensor
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
            raise ValueError(f"selected_nets索引超出范围 [0, {self.num_nets-1}]")

        total_cutsize = torch.tensor(0.0, device=self.pin_pos_x.device)

        # get weights
        if cutsize_net_weights is None:
            # use weights from self.net_weights corresponding to the selected nets
            weights = self.net_weights[selected_nets]
        else:
            # use user-provided weights
            if isinstance(cutsize_net_weights, list):
                cutsize_net_weights = torch.tensor(
                    cutsize_net_weights,
                    dtype=torch.float32,
                    device=self.pin_pos_x.device)
            if cutsize_net_weights.numel() != selected_nets.numel():
                raise ValueError(
                    f"cutsize_net_weights长度({cutsize_net_weights.numel()}) "
                    f"必须与selected_nets长度({selected_nets.numel()})匹配")
            weights = cutsize_net_weights

        # vectorized calculation of cutsize for all selected nets
        cutsizes = self.compute_cutsize_batch(
            selected_nets)  # [num_selected_nets]

        total_cutsize = (weights * cutsizes).sum()

        return total_cutsize

    def forward(self,
                entropy_weight=0.0,
                lambda_wl=1.0,
                lambda_cut=0.0,
                selected_nets=None,
                cutsize_net_weights=None,
                return_debug_info=False):
        """
        calculate total loss (HPWL + Cutsize)
        L_WL = Σ_e (HPWL_top_e + HPWL_bottom_e) * weight_e
        L_cut = Σ_{n∈selected_nets} w(n) * cutsize(n)
        L_total = λ_WL * L_WL + λ_cut * L_cut
        
        Args:
            entropy_weight: entropy regularization weight, used to encourage z near 0 or 1 (default 0.0, no regularization)
                            suggested value: 0.01-0.1, gradually increase during training
            lambda_wl: weight of HPWL loss, default 1.0
            lambda_cut: weight of cutsize loss, default 0.0
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
                    - 'L_entropy': entropy regularization loss
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

        # add entropy regularization term, encourage z near 0 or 1
        # entropy H(z) = -z*log(z) - (1-z)*log(1-z)
        # when z is near 0 or 1, entropy is minimized; when z=0.5, entropy is maximized
        entropy_loss = torch.tensor(0.0, device=self.pin_pos_x.device)
        if entropy_weight > 0:
            z = self.get_z()
            # numerically stable: avoid log(0)
            eps = 1e-8
            entropy = -(z * torch.log(z + eps) +
                        (1 - z) * torch.log(1 - z + eps))
            entropy_loss = entropy.mean()

        cutsize_loss = torch.tensor(0.0, device=self.pin_pos_x.device)
        if lambda_cut > 0:
            cutsize_loss = self.compute_cutsize_loss(
                selected_nets=selected_nets,
                cutsize_net_weights=cutsize_net_weights)

        total_loss = (lambda_wl * total_hpwl + lambda_cut * cutsize_loss +
                      entropy_weight * entropy_loss)

        # if not return debug information, return total loss
        if not return_debug_info:
            return total_loss

        # return total loss and debug information
        debug_info = {
            'L_WL': total_hpwl.item(),
            'L_cut': cutsize_loss.item() if lambda_cut > 0 else 0.0,
            'L_entropy': entropy_loss.item() if entropy_weight > 0 else 0.0,
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
                       save_path=None):
    """
    visualize z values over xy for a single iteration
    
    Args:
        x_coords: x coordinates, shape [num_cells] tensor or numpy array
        y_coords: y coordinates, shape [num_cells] tensor or numpy array
        z_values: z values, shape [num_cells] tensor or numpy array
        iteration: current iteration number
        save_path: save path (optional)
    """
    # convert to numpy array
    if isinstance(x_coords, torch.Tensor):
        x_coords = x_coords.detach().cpu().numpy()
    if isinstance(y_coords, torch.Tensor):
        y_coords = y_coords.detach().cpu().numpy()
    if isinstance(z_values, torch.Tensor):
        z_values = z_values.detach().cpu().numpy()

    # create single 3D plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # set colors based on z values (blue=bottom, red=top)
    colors = z_values

    # plot 3D scatter plot
    scatter = ax.scatter(x_coords,
                         y_coords,
                         z_values,
                         c=colors,
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

    # set z axis range
    ax.set_zlim(0, 1)

    # add color bar
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
    num_cells = 2735
    num_nets = 2644
    node_pos = torch.load("case2_hidden_2d_placement.pt")
    pin_pos = torch.load("case2_hidden_2d_pinpos.pt")
    flat_net2pin_map = torch.load("case2_hidden_flat_net2pin_map.pt")
    flat_net2pin_start_map = torch.load(
        "case2_hidden_flat_net2pin_start_map.pt")
    pin2node_map = torch.load("case2_hidden_pin2node_map.pt")
    # breakpoint()

    node_x = node_pos[:num_cells]
    node_y = node_pos[node_pos.numel() // 2:node_pos.numel() // 2 + num_cells]
    pin_pos_x = pin_pos[:pin2node_map.numel()]
    pin_pos_y = pin_pos[pin2node_map.numel():]
    # breakpoint()  # 已注释掉，避免程序暂停

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
        alpha=1.0  # initial alpha value
    )
    print(f"   - Initial alpha: {model.alpha}")
    print(
        f"   - Number of trainable parameters: {sum(p.numel() for p in model.parameters())}"
    )

    # configure cutsize loss
    use_cutsize_loss = True
    lambda_cut_start = 1.0
    lambda_cut_end = 1.0
    selected_nets_for_cutsize = None  # None means apply cutsize constraint to all nets
    cutsize_net_weights = None  # None means use self.net_weights corresponding to the selected nets
    # can specify specific network subset, e.g.:
    # selected_nets_for_cutsize = torch.tensor([0, 1, 2, 10, 20, 50], dtype=torch.long)

    # print initial state
    print("\n3. Initial state:")
    if use_cutsize_loss:
        initial_loss, debug_info = model(
            entropy_weight=0.0,
            lambda_wl=1.0,
            lambda_cut=lambda_cut_start,
            selected_nets=selected_nets_for_cutsize,
            cutsize_net_weights=cutsize_net_weights,
            return_debug_info=True)
        print(f"   - Initial total loss: {initial_loss.item():.4f}")
        print(f"   - Initial HPWL: {debug_info['L_WL']:.4f}")
        print(f"   - Initial cutsize: {debug_info['L_cut']:.4f}")
    else:
        initial_loss = model(entropy_weight=0.0)
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
    num_iterations = 500
    alpha_start = 1.0
    alpha_end = 20.0
    alpha_schedule = np.linspace(alpha_start, alpha_end, num_iterations)

    # entropy regularization weight schedule (optional, to encourage z near 0 or 1)
    # no entropy regularization at the beginning, gradually increase later
    entropy_start = 0.0
    entropy_end = 0.05  # can be adjusted according to needs
    entropy_schedule = np.linspace(entropy_start, entropy_end, num_iterations)

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

    print(f"\n5. Start training ({num_iterations} iterations)...")
    print(f"   - Alpha schedule: {alpha_start} → {alpha_end}")
    print(
        f"   - Entropy regularization weight schedule: {entropy_start} → {entropy_end}"
    )
    if use_cutsize_loss:
        num_selected = num_nets if selected_nets_for_cutsize is None else len(
            selected_nets_for_cutsize)
        print(
            f"   - Cutsize loss enabled: λ_cut exponential schedule {lambda_cut_start} → {lambda_cut_end}, applied to {num_selected} nets"
        )
    else:
        print(f"   - Cutsize loss disabled")
    print("-" * 60)

    # create directory for saving visualizations
    save_dir = os.path.join(os.path.dirname(__file__), 'visualizations')
    os.makedirs(save_dir, exist_ok=True)

    # training loop
    for iteration in range(num_iterations):
        # update alpha (gradually increase, making soft assignment harder)
        model.alpha = alpha_schedule[iteration]
        entropy_weight = entropy_schedule[iteration]

        # update lambda_cut (gradually increase cutsize loss weight)
        if use_cutsize_loss:
            lambda_cut = lambda_cut_schedule[iteration]
        else:
            lambda_cut = 0.0

        # forward propagation (with entropy regularization and optional cutsize loss)
        if use_cutsize_loss:
            loss, debug_info = model(entropy_weight=entropy_weight,
                                     lambda_wl=1.0,
                                     lambda_cut=lambda_cut,
                                     selected_nets=selected_nets_for_cutsize,
                                     cutsize_net_weights=cutsize_net_weights,
                                     return_debug_info=True)
        else:
            loss = model(entropy_weight=entropy_weight)

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

            if use_cutsize_loss:
                print(f"Iter {iteration+1:4d} | "
                      f"Loss: {loss.item():8.2f} | "
                      f"HPWL: {debug_info['L_WL']:8.2f} | "
                      f"Cut: {debug_info['L_cut']:6.4f} | "
                      f"λ_cut: {lambda_cut:6.4f} | "
                      f"Alpha: {model.alpha:5.2f} | "
                      f"||dt||: {t_grad_norm:6.4f} | "
                      f"Top: {stats['top_cells']:3d} | "
                      f"Bottom: {stats['bottom_cells']:3d}")
            else:
                print(f"Iter {iteration+1:4d} | "
                      f"Loss: {loss.item():8.2f} | "
                      f"Alpha: {model.alpha:5.2f} | "
                      f"||dt||: {t_grad_norm:6.4f} | "
                      f"||dz/dt||: {dz_dt_norm:6.4f} | "
                      f"Top: {stats['top_cells']:3d} | "
                      f"Bottom: {stats['bottom_cells']:3d}")

            save_path = os.path.join(
                save_dir, f'z_evolution_iter_{iteration+1:04d}.png')
            visualize_z_single(node_x, node_y, z, iteration + 1, save_path)

    print("-" * 60)

    # final results
    print("\n6. Training completed, final results:")
    if use_cutsize_loss:
        final_loss, final_debug_info = model(
            entropy_weight=entropy_end,
            lambda_wl=1.0,
            lambda_cut=lambda_cut_end,
            selected_nets=selected_nets_for_cutsize,
            cutsize_net_weights=cutsize_net_weights,
            return_debug_info=True)
        print(f"   - Final total loss: {final_loss.item():.4f}")
        print(f"   - Final HPWL: {final_debug_info['L_WL']:.4f}")
        print(f"   - Final cutsize: {final_debug_info['L_cut']:.4f}")
    else:
        final_loss = model(entropy_weight=entropy_end)
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
