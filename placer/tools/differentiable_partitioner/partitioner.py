'''
Author: JeanneWillis hi@jeannewillis.cn
Date: 2025-11-14 16:03:37
LastEditors: JeanneWillis hi@jeannewillis.cn
LastEditTime: 2025-11-15 20:43:29
FilePath: /D2D-placer/placer/tools/differentiable_partitioner/partitioner.py
Description: 基于LSE软边界框的可微分3D分割器
'''
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os


class LSEPartitioner(nn.Module):
    """
    基于LogSumExp软边界框的可微分3D分割器
    
    该模块实现：
    1. 将可训练的预激活变量t_i映射到软分配z_i = sigmoid(t_i)
    2. 使用LSE软边界框计算顶层和底层的HPWL
    3. 计算总HPWL对t的梯度，支持端到端训练
    
    支持真实的电路结构：
    - 每个net包含多个pins
    - 每个pin属于一个node（cell）
    - 每个node可能有多个pins
    - 使用pin的位置计算HPWL
    - 通过pin2node_map将梯度映射到node级别
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
        初始化LSE分割器
        
        Args:
            num_cells: 单元（node）数量
            flat_net2pin_map: 扁平化的net到pin映射，形状为[num_pins]的tensor
            flat_net2pin_start_map: 每个net在flat_net2pin_map中的起始位置，形状为[num_nets+1]的tensor
            pin2node_map: pin到node的映射，形状为[num_pins]的tensor
            pin_pos_x: pin的x坐标，形状为[num_pins]的tensor
            pin_pos_y: pin的y坐标，形状为[num_pins]的tensor
            alpha: LSE平滑参数，越大越接近硬max/min
            net_weights: 可选的网权重，形状为[num_nets]的tensor
        """
        super(LSEPartitioner, self).__init__()

        self.num_cells = num_cells
        self.num_nets = flat_net2pin_start_map.numel() - 1

        # 注册固定的数据结构（不需要梯度）
        self.register_buffer('flat_net2pin_map',
                             flat_net2pin_map.detach().clone())
        self.register_buffer('flat_net2pin_start_map',
                             flat_net2pin_start_map.detach().clone())
        self.register_buffer('pin2node_map', pin2node_map.detach().clone())
        self.register_buffer('pin_pos_x', pin_pos_x.detach().clone())
        self.register_buffer('pin_pos_y', pin_pos_y.detach().clone())

        # 可训练的预激活变量t_i（每个cell一个）
        # 使用小的随机初始化，避免所有z都等于0.5（对称点）
        self.t = nn.Parameter(torch.randn(num_cells) * 0.01)

        # LSE平滑参数
        self.alpha = alpha

        # 网权重（如果未提供，默认为全1）
        if net_weights is None:
            self.register_buffer('net_weights', torch.ones(self.num_nets))
        else:
            self.register_buffer('net_weights', net_weights.detach().clone())

    def get_z(self):
        """
        将预激活变量t映射到软分配z = sigmoid(t)
        
        Returns:
            z: 形状为[num_cells]的tensor，表示每个cell分配到顶层的概率
        """
        return torch.sigmoid(self.t)

    def lse_max(self, weighted_vals):
        """
        数值稳定的LSE最大值计算
        
        公式: max(x) ≈ (1/α) * log_sum_exp(α * x)
        
        数值稳定实现: log_sum_exp(a_j) = m + log Σ_j exp(a_j - m)
        其中 m = max_j a_j
        
        Args:
            weighted_vals: 加权值，形状为[...]的tensor
            
        Returns:
            软最大值，形状与weighted_vals相同（除了被聚合的维度）
        """
        # 使用torch.logsumexp确保数值稳定性
        # logsumexp(α * x) = log(Σ exp(α * x))
        # 然后除以α得到软最大值
        if self.alpha <= 0:
            raise ValueError("alpha must be positive")

        # 计算 log_sum_exp(α * weighted_vals)
        lse = torch.logsumexp(self.alpha * weighted_vals, dim=-1)

        # 返回 (1/α) * log_sum_exp(α * weighted_vals)
        return lse / self.alpha

    def lse_min(self, weighted_vals):
        """
        数值稳定的LSE最小值计算
        
        公式: min(x) ≈ -(1/α) * log_sum_exp(-α * x)
        
        Args:
            weighted_vals: 加权值，形状为[...]的tensor
            
        Returns:
            软最小值
        """
        if self.alpha <= 0:
            raise ValueError("alpha must be positive")

        # 计算 log_sum_exp(-α * weighted_vals)
        lse = torch.logsumexp(-self.alpha * weighted_vals, dim=-1)

        # 返回 -(1/α) * log_sum_exp(-α * weighted_vals)
        return -lse / self.alpha

    def compute_hpwl_top(self, net_idx):
        """
        计算指定网在顶层的软HPWL
        
        使用LSE软边界框：
        - x_max_top = (1/α) * log_sum_exp(α * z_node * x_pin)
        - x_min_top = -(1/α) * log_sum_exp(-α * z_node * x_pin)
        - y_max_top = (1/α) * log_sum_exp(α * z_node * y_pin)
        - y_min_top = -(1/α) * log_sum_exp(-α * z_node * y_pin)
        - HPWL_top = (x_max_top - x_min_top) + (y_max_top - y_min_top)
        
        注意：z是按node定义的，但位置是按pin定义的
        
        Args:
            net_idx: 网的索引
            
        Returns:
            该网在顶层的软HPWL
        """
        # 获取该网的所有pin索引
        start_idx = self.flat_net2pin_start_map[net_idx]
        end_idx = self.flat_net2pin_start_map[net_idx + 1]
        pin_indices = self.flat_net2pin_map[start_idx:end_idx]

        if pin_indices.numel() < 2:
            return torch.tensor(0.0, device=self.pin_pos_x.device)

        # 获取软分配z（按node定义）
        z = self.get_z()

        # 获取每个pin对应的node索引
        node_indices = self.pin2node_map[pin_indices]  # 形状: [num_pins_in_net]

        # 获取每个pin对应的node的z值
        z_net = z[node_indices]  # 形状: [num_pins_in_net]

        # 获取每个pin的位置
        x_net = self.pin_pos_x[pin_indices]  # 形状: [num_pins_in_net]
        y_net = self.pin_pos_y[pin_indices]  # 形状: [num_pins_in_net]

        # 计算加权值: z_node * x_pin 和 z_node * y_pin
        weighted_x = z_net * x_net  # 形状: [num_pins_in_net]
        weighted_y = z_net * y_net  # 形状: [num_pins_in_net]

        # 使用LSE计算软边界框
        x_max_top = self.lse_max(weighted_x)
        x_min_top = self.lse_min(weighted_x)
        y_max_top = self.lse_max(weighted_y)
        y_min_top = self.lse_min(weighted_y)

        # 计算HPWL
        hpwl_x = x_max_top - x_min_top
        hpwl_y = y_max_top - y_min_top
        hpwl_top = hpwl_x + hpwl_y

        return hpwl_top

    def compute_hpwl_bottom(self, net_idx):
        """
        计算指定网在底层的软HPWL
        
        使用与顶层相同的LSE公式，但使用(1 - z_node)代替z_node
        
        Args:
            net_idx: 网的索引
            
        Returns:
            该网在底层的软HPWL
        """
        # 获取该网的所有pin索引
        start_idx = self.flat_net2pin_start_map[net_idx]
        end_idx = self.flat_net2pin_start_map[net_idx + 1]
        pin_indices = self.flat_net2pin_map[start_idx:end_idx]

        if pin_indices.numel() < 2:
            return torch.tensor(0.0, device=self.pin_pos_x.device)

        # 获取软分配z（按node定义）
        z = self.get_z()

        # 获取每个pin对应的node索引
        node_indices = self.pin2node_map[pin_indices]  # 形状: [num_pins_in_net]

        # 使用(1 - z_node)表示底层分配
        z_bottom = 1.0 - z[node_indices]  # 形状: [num_pins_in_net]

        # 获取每个pin的位置
        x_net = self.pin_pos_x[pin_indices]  # 形状: [num_pins_in_net]
        y_net = self.pin_pos_y[pin_indices]  # 形状: [num_pins_in_net]

        # 计算加权值: (1 - z_node) * x_pin 和 (1 - z_node) * y_pin
        weighted_x = z_bottom * x_net
        weighted_y = z_bottom * y_net

        # 使用LSE计算软边界框
        x_max_bottom = self.lse_max(weighted_x)
        x_min_bottom = self.lse_min(weighted_x)
        y_max_bottom = self.lse_max(weighted_y)
        y_min_bottom = self.lse_min(weighted_y)

        # 计算HPWL
        hpwl_x = x_max_bottom - x_min_bottom
        hpwl_y = y_max_bottom - y_min_bottom
        hpwl_bottom = hpwl_x + hpwl_y

        return hpwl_bottom

    def forward(self, entropy_weight=0.0):
        """
        前向传播：计算总HPWL损失
        
        L_WL = Σ_e (HPWL_top_e + HPWL_bottom_e) * weight_e
        
        Args:
            entropy_weight: 熵正则化权重，用于鼓励z接近0或1（默认0.0，不添加正则化）
                           建议值：0.01-0.1，随着训练逐渐增加
        
        Returns:
            total_hpwl: 总HPWL损失，标量tensor
        """
        total_hpwl = torch.tensor(0.0, device=self.pin_pos_x.device)

        # 遍历所有网，累加HPWL
        for net_idx in range(self.num_nets):
            hpwl_top = self.compute_hpwl_top(net_idx)
            hpwl_bottom = self.compute_hpwl_bottom(net_idx)

            # 加权累加
            total_hpwl += self.net_weights[net_idx] * (hpwl_top + hpwl_bottom)

        # 添加熵正则化项，鼓励z接近0或1
        # 熵 H(z) = -z*log(z) - (1-z)*log(1-z)
        # 当z接近0或1时，熵最小；当z=0.5时，熵最大
        if entropy_weight > 0:
            z = self.get_z()
            # 数值稳定：避免log(0)
            eps = 1e-8
            entropy = -(z * torch.log(z + eps) +
                        (1 - z) * torch.log(1 - z + eps))
            entropy_loss = entropy.mean()
            total_hpwl = total_hpwl + entropy_weight * entropy_loss

        return total_hpwl

    def get_binary_assignment(self, threshold=0.5):
        """
        将软分配z转换为硬二进制分配
        
        Args:
            threshold: 阈值，默认0.5
            
        Returns:
            binary_z: 二进制分配，1表示顶层，0表示底层
        """
        z = self.get_z()
        return (z > threshold).to(torch.int32)

    def get_assignment_stats(self):
        """
        获取分配统计信息，用于调试
        
        Returns:
            dict: 包含z的统计信息
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


def create_dummy_data(num_cells=100, num_nets=50, seed=42):
    """
    创建用于测试的虚拟数据
    
    Args:
        num_cells: 单元数量
        num_nets: 网数量
        seed: 随机种子
        
    Returns:
        x_coords: x坐标tensor
        y_coords: y坐标tensor
        netlist: 网表
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    # 生成随机2D坐标（假设在[0, 1000]范围内）
    x_coords = torch.rand(num_cells) * 1000.0
    y_coords = torch.rand(num_cells) * 1000.0

    # 生成随机网表
    # 每个网包含2-10个随机选择的pin
    netlist = []
    for _ in range(num_nets):
        num_pins = np.random.randint(2, 11)
        pins = np.random.choice(num_cells, size=num_pins,
                                replace=False).tolist()
        netlist.append(pins)

    return x_coords, y_coords, netlist


def visualize_z_single(x_coords,
                       y_coords,
                       z_values,
                       iteration,
                       save_path=None):
    """
    可视化单个迭代的z值相对于xy的分布
    
    Args:
        x_coords: x坐标，形状为[num_cells]的tensor或numpy数组
        y_coords: y坐标，形状为[num_cells]的tensor或numpy数组
        z_values: z值，形状为[num_cells]的tensor或numpy数组
        iteration: 当前迭代次数
        save_path: 保存路径（可选）
    """
    # 转换为numpy数组
    if isinstance(x_coords, torch.Tensor):
        x_coords = x_coords.detach().cpu().numpy()
    if isinstance(y_coords, torch.Tensor):
        y_coords = y_coords.detach().cpu().numpy()
    if isinstance(z_values, torch.Tensor):
        z_values = z_values.detach().cpu().numpy()

    # 创建单个3D图
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # 根据z值设置颜色（蓝色=底层，红色=顶层）
    colors = z_values

    # 绘制3D散点图
    scatter = ax.scatter(x_coords,
                         y_coords,
                         z_values,
                         c=colors,
                         cmap='RdYlBu_r',
                         s=20,
                         alpha=0.6,
                         edgecolors='k',
                         linewidths=0.3)

    # 使用英文标签以避免字体问题，同时保持清晰
    ax.set_xlabel('X Coordinate', fontsize=12)
    ax.set_ylabel('Y Coordinate', fontsize=12)
    ax.set_zlabel('Z Value (Soft Assignment)', fontsize=12)
    ax.set_title(f'Iteration {iteration} - Z Distribution over XY',
                 fontsize=14,
                 fontweight='bold')

    # 设置z轴范围
    ax.set_zlim(0, 1)

    # 添加颜色条
    plt.colorbar(scatter, ax=ax, shrink=0.8, label='Z Value')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"   可视化已保存到: {save_path}")

    plt.close()  # 关闭图形以释放内存，不显示窗口


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
