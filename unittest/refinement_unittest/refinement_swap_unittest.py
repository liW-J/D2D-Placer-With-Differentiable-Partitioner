'''
Author: JeanneWillis hi@jeannewillis.cn
Date: 2025-10-18 18:22:43
LastEditors: JeanneWillis hi@jeannewillis.cn
LastEditTime: 2025-10-19 23:40:01
FilePath: /D2D-placer/unittest/refinement_swap_unittest.py
Description: 
'''

import os
import sys

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))))

import configure
import time
from placer.d2d_placer import D2Dplacer
from placer.d2d_placer import init_log
import matplotlib.pyplot as plt
import torch


def d2d_placer_init(d2d_placer):
    d2d_placer.tier = torch.load(
        'results/case2/2025-10-19_01-10-22/tier-refinement.pt')
    d2d_placer.params.flatten_2d.global_place_flag = False
    d2d_placer.params.flatten_2d.random_center_init_flag = False
    d2d_placer.params.flatten_2d.legalize_flag = False
    d2d_placer.params.flatten_2d.detailed_place_flag = False
    d2d_placer.params.flatten_2d.ntuplace_flag = False

    d2d_placer.params.terminal.global_place_flag = False
    d2d_placer.params.terminal.random_center_init_flag = False
    d2d_placer.params.terminal.legalize_flag = False
    d2d_placer.params.terminal.detailed_place_flag = False
    d2d_placer.params.terminal.ntuplace_flag = False

    d2d_placer.dreamplace.dp_2d.place(d2d_placer.params.flatten_2d,
                                      d2d_placer.timer)

    d2d_placer.dreamplace.dp_terminal.init_basic_place(
        d2d_placer.params.terminal, d2d_placer.timer)
    d2d_placer.dreamplace.dp_terminal.place(d2d_placer.params.terminal,
                                            d2d_placer.timer)

    d2d_placer.num_terminal_NIs = 132
    d2d_placer.cut_net_mask = torch.load("./cut_net_mask.pt")

    d2d_placer.die_by_die_place(global_place_flag=False,
                                legalize_flag=False,
                                detailed_place_flag=False,
                                random_center_init_flag=False,
                                ntuplace_flag=False)

    d2d_placer.hpwl_d2d()

    d2d_placer.params.flatten_2d.global_place_flag = True
    d2d_placer.params.flatten_2d.random_center_init_flag = True
    d2d_placer.params.flatten_2d.legalize_flag = False
    d2d_placer.params.flatten_2d.detailed_place_flag = False
    d2d_placer.params.flatten_2d.ntuplace_flag = True

    d2d_placer.params.terminal.global_place_flag = True
    d2d_placer.params.terminal.random_center_init_flag = True
    d2d_placer.params.terminal.legalize_flag = False
    d2d_placer.params.terminal.detailed_place_flag = False
    d2d_placer.params.terminal.ntuplace_flag = True

    return d2d_placer


def refinement_test(d2d_placer):

    fig, ax1 = plt.subplots(figsize=(12, 8))
    plt.title('swap same pos but different tiers',
              fontsize=16,
              fontweight='bold',
              pad=20)

    ax1.set_xlabel('number of swaps (k)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('HPWL Improvement Ratio',
                   fontsize=14,
                   fontweight='bold',
                   color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.grid(True, alpha=0.3, linestyle='--')

    # 创建第二个y轴
    ax2 = ax1.twinx()
    ax2.set_ylabel('Max Distance',
                   fontsize=14,
                   fontweight='bold',
                   color='red',
                   rotation=270,
                   va='bottom')
    ax2.yaxis.set_label_position('right')  # 将标签显示在右侧
    ax2.tick_params(axis='y', labelcolor='red')

    k = [i for i in range(0, 720, 2)]
    k_tmp = []
    max_distance_tmp = []
    hpwl_gain_ratio_tmp = []
    hpwl = d2d_placer.hpwl_d2d()
    tier_init = d2d_placer.tier.clone()
    pos_init = d2d_placer.dreamplace.dp_2d.pos.clone()

    def swap_pos(u_index, v_index):
        pos_2d_x = d2d_placer.dreamplace.dp_2d.pos[:d2d_placer.dreamplace.dp_2d
                                                   .placedb.num_nodes].clone()
        pos_2d_y = d2d_placer.dreamplace.dp_2d.pos[d2d_placer.dreamplace.dp_2d.
                                                   placedb.num_nodes:].clone()

        # pos_2d_x[u_index], pos_2d_x[v_index] = pos_2d_x[v_index].clone(
        # ), pos_2d_x[u_index].clone()
        # pos_2d_y[u_index], pos_2d_y[v_index] = pos_2d_y[v_index].clone(
        # ), pos_2d_y[u_index].clone()

        pos_2d_tmp = torch.cat([pos_2d_x, pos_2d_y])

        d2d_placer.tier[u_index], d2d_placer.tier[v_index] = d2d_placer.tier[
            v_index].clone(), d2d_placer.tier[u_index].clone()
        return pos_2d_tmp

    # d2d_placer.params.terminal.global_place_flag = False

    def find_closest_nodes_across_tiers(tier0_indices,
                                        tier1_indices,
                                        max_distance=200.0):
        """
        找到不同层中位置最相近的节点对
        """
        pairs = []
        used_tier0 = set()
        used_tier1 = set()

        pos_2d_x = d2d_placer.dreamplace.dp_2d.pos[:d2d_placer.dreamplace.dp_2d
                                                   .placedb.num_nodes].clone()
        pos_2d_y = d2d_placer.dreamplace.dp_2d.pos[d2d_placer.dreamplace.dp_2d.
                                                   placedb.num_nodes:].clone()

        for t0_idx in tier0_indices:
            if t0_idx in used_tier0:
                continue

            min_distance = float('inf')
            best_t1_idx = None

            for t1_idx in tier1_indices:
                if t1_idx in used_tier1:
                    continue

                distance = torch.sqrt(
                    (pos_2d_x[t0_idx] - pos_2d_x[t1_idx])**2 +
                    (pos_2d_y[t0_idx] - pos_2d_y[t1_idx])**2).item()

                if distance < min_distance and distance <= max_distance:
                    min_distance = distance
                    best_t1_idx = t1_idx

            if best_t1_idx is not None:
                pairs.append((t0_idx, best_t1_idx, min_distance))
                used_tier0.add(t0_idx)
                used_tier1.add(best_t1_idx)

        pairs.sort(key=lambda x: x[2])
        return pairs

    idx_t0 = [int(x) for x in torch.where(tier_init == 0)[0].tolist()]
    idx_t1 = [int(x) for x in torch.where(tier_init == 1)[0].tolist()]
    closest_pairs = find_closest_nodes_across_tiers(idx_t0, idx_t1)

    for i in k:
        d2d_placer.tier = tier_init.clone()
        d2d_placer.dreamplace.dp_2d.pos = pos_init.clone()
        d2d_placer.hpwl_d2d()
        swapped_nodes = set()

        num_swaps = min(i, len(closest_pairs))
        if num_swaps < i:
            break

        swap_count = 0
        pair_index = 0
        max_distance_this_k = 0.0

        while swap_count < num_swaps and pair_index < len(closest_pairs):
            u_index, v_index, distance = closest_pairs[pair_index]

            if (u_index not in swapped_nodes) and (v_index
                                                   not in swapped_nodes):

                d2d_placer.dreamplace.dp_2d.pos = swap_pos(u_index, v_index)
                swapped_nodes.add(u_index)
                swapped_nodes.add(v_index)
                swap_count += 1

                # 更新当前k的最大距离
                max_distance_this_k = max(max_distance_this_k, distance)

            pair_index += 1

        d2d_placer.hpwl_d2d()

        d2d_placer.refinement()
        hpwl_tmp = d2d_placer.die_by_die_place(global_place_flag=True,
                                               legalize_flag=False,
                                               detailed_place_flag=False,
                                               random_center_init_flag=True,
                                               ntuplace_flag=True,
                                               logger=d2d_logger)

        hpwl_gain_ratio_tmp.append((hpwl - hpwl_tmp) / hpwl)
        max_distance_tmp.append(max_distance_this_k)
        k_tmp.append(i)

        # 清除之前的图例
        ax1.clear()
        ax2.clear()

        # 重新设置坐标轴
        ax1.set_xlabel('number of swaps (k)', fontsize=14, fontweight='bold')
        ax1.set_ylabel('HPWL Improvement Ratio',
                       fontsize=14,
                       fontweight='bold',
                       color='blue')
        ax1.tick_params(axis='y', labelcolor='blue')
        ax1.grid(True, alpha=0.3, linestyle='--')

        ax2.set_ylabel('Max Distance',
                       fontsize=14,
                       fontweight='bold',
                       color='red',
                       rotation=270,
                       va='bottom')
        ax2.tick_params(axis='y', labelcolor='red')
        ax2.yaxis.set_label_position('right')  # 将标签显示在右侧
        # 绘制HPWL Improvement Ratio
        ax1.plot(k_tmp,
                 hpwl_gain_ratio_tmp,
                 'b-o',
                 linewidth=2,
                 markersize=4,
                 label='HPWL Improvement Ratio')

        # 绘制Max Distance
        ax2.plot(k_tmp,
                 max_distance_tmp,
                 'r-s',
                 linewidth=2,
                 markersize=4,
                 label='Max Distance')
        for i in range(0, len(k_tmp), 10):
            plt.annotate(f'{hpwl_gain_ratio_tmp[i]:.4f}',
                         (k_tmp[i], hpwl_gain_ratio_tmp[i]),
                         textcoords="offset points",
                         xytext=(0, 10),
                         ha='center',
                         fontsize=8,
                         bbox=dict(boxstyle="round,pad=0.3",
                                   facecolor="yellow",
                                   alpha=0.7))

        ax1.set_xlim(0, max(k_tmp) * 1.05)
        ax1.set_ylim(
            min(hpwl_gain_ratio_tmp) -
            abs(max(hpwl_gain_ratio_tmp) - min(hpwl_gain_ratio_tmp)) * 0.05,
            max(hpwl_gain_ratio_tmp) +
            abs(max(hpwl_gain_ratio_tmp) - min(hpwl_gain_ratio_tmp)) * 0.05)

        max_gain = max(hpwl_gain_ratio_tmp)
        max_gain_k = k_tmp[hpwl_gain_ratio_tmp.index(max_gain)]
        ax1.axhline(y=max_gain, color='blue', linestyle=':', alpha=0.7)
        ax1.axvline(x=max_gain_k, color='blue', linestyle=':', alpha=0.7)

        # 添加图例
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2,
                   labels1 + labels2,
                   loc='upper right',
                   fontsize=12)

        plt.tight_layout()
        plt.savefig('refinement_test_ratio.png', dpi=300, bbox_inches='tight')


if __name__ == "__main__":
    """
    @brief main function to invoke the entire placement flow.
    """

    d2d_placer = D2Dplacer(sys.argv[1])
    d2d_logger = init_log(d2d_placer.params.result_dir_root)
    d2d_placer.init_spec()

    d2d_placer = d2d_placer_init(d2d_placer)

    d2d_placer.refinement()
    d2d_placer.hpwl_d2d()

    # refinement_test(d2d_placer)
    # d2d_placer.output()
    # d2d_placer.analyze_results()
