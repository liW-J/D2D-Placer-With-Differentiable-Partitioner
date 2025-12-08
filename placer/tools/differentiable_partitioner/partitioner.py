'''
Author: JeanneWillis hi@jeannewillis.cn
Date: 2025-11-14 16:03:37
LastEditors: JeanneWillis hi@jeannewillis.cn
LastEditTime: 2025-11-17 22:54:10
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
        self.t = nn.Parameter(torch.randn(num_cells) * 0.01)

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
        x_net = self.pin_pos_x[pin_indices]  # shape: [num_pins_in_net]
        y_net = self.pin_pos_y[pin_indices]  # shape: [num_pins_in_net]

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

    def forward(self, entropy_weight=0.0):
        """
        calculate total HPWL loss
        L_WL = Σ_e (HPWL_top_e + HPWL_bottom_e) * weight_e
        
        Args:
            entropy_weight: entropy regularization weight, used to encourage z near 0 or 1 (default 0.0, no regularization)
                            suggested value: 0.01-0.1, gradually increase during training
        
        Returns:
            total_hpwl: total HPWL loss, scalar tensor
        """
        total_hpwl = torch.tensor(0.0, device=self.pin_pos_x.device)

        # iterate over all nets, accumulate HPWL
        for net_idx in range(self.num_nets):
            hpwl_top = self.compute_hpwl_top(net_idx)
            hpwl_bottom = self.compute_hpwl_bottom(net_idx)

            # weighted accumulate
            total_hpwl += self.net_weights[net_idx] * (hpwl_top + hpwl_bottom)

        # add entropy regularization term, encourage z near 0 or 1
        # entropy H(z) = -z*log(z) - (1-z)*log(1-z)
        # when z is near 0 or 1, entropy is minimized; when z=0.5, entropy is maximized
        if entropy_weight > 0:
            z = self.get_z()
            # numerically stable: avoid log(0)
            eps = 1e-8
            entropy = -(z * torch.log(z + eps) +
                        (1 - z) * torch.log(1 - z + eps))
            entropy_loss = entropy.mean()
            total_hpwl = total_hpwl + entropy_weight * entropy_loss

        return total_hpwl

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

    node_x = node_pos[:num_cells]
    node_y = node_pos[node_pos.numel() // 2:node_pos.numel() // 2 + num_cells]
    pin_pos_x = pin_pos[:pin2node_map.numel()]
    pin_pos_y = pin_pos[pin2node_map.numel():]

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

    # print initial state
    print("\n3. Initial state:")
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

    print(f"\n5. Start training ({num_iterations} iterations)...")
    print(f"   - Alpha schedule: {alpha_start} → {alpha_end}")
    print(
        f"   - Entropy regularization weight schedule: {entropy_start} → {entropy_end}"
    )
    print("-" * 60)

    # create directory for saving visualizations
    save_dir = os.path.join(os.path.dirname(__file__), 'visualizations')
    os.makedirs(save_dir, exist_ok=True)

    # training loop
    for iteration in range(num_iterations):
        # update alpha (gradually increase, making soft assignment harder)
        model.alpha = alpha_schedule[iteration]
        entropy_weight = entropy_schedule[iteration]

        # forward propagation (with entropy regularization)
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
                print(f"警告: 迭代 {iteration+1} 检测到NaN/Inf梯度")
        else:
            t_grad_norm = 0.0

        # update parameters
        optimizer.step()

        # print and visualize every 100 iterations
        if (iteration + 1) % 10 == 0 or iteration == 0:
            z = model.get_z()
            dz_dt_norm = (z * (1 - z)).norm().item()  # sigmoid导数的范数
            stats = model.get_assignment_stats()

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

    # 显示一些示例z值
    z = model.get_z()
    print(f"\n8. Example soft assignment values (first 10 cells):")
    for i in range(min(10, num_cells)):
        print(
            f"   Cell {i:3d}: z={z[i].item():.4f} → {'Top' if z[i] > 0.5 else 'Bottom'}"
        )

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)
    torch.save(binary_z, "case2_hidden_2d_binary_assignment.pt")


if __name__ == "__main__":
    main()
