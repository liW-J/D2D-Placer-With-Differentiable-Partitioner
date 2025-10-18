'''
Author: JeanneWillis hi@jeannewillis.cn
Date: 2025-10-18 18:22:43
LastEditors: JeanneWillis hi@jeannewillis.cn
LastEditTime: 2025-10-18 19:12:22
FilePath: /D2D-placer/unittest/refinement_swap_unittest.py
Description: 
'''

import sys
import time
from placer.d2d_placer import D2Dplacer
from placer.d2d_placer import init_log
import matplotlib.pyplot as plt
import torch


def refinement_test(d2d_placer):

    plt.figure(figsize=(12, 8))
    plt.title('numberOfSwaps vs HPWL Improvement Ratio',
              fontsize=16,
              fontweight='bold',
              pad=20)
    plt.xlabel('number of swaps (k)', fontsize=14, fontweight='bold')
    plt.ylabel('HPWL Improvement Ratio', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.legend(fontsize=12)

    k = [i for i in range(1, 1300, 5)]
    k_tmp = []
    hpwl_gain_ratio_tmp = []
    hpwl = d2d_placer.hpwl_d2d()
    tier_init = d2d_placer.tier.clone()

    pos_2d_x = d2d_placer.dreamplace.dp_2d.pos[:d2d_placer.dreamplace.dp_2d.
                                               placedb.num_nodes].clone()
    pos_2d_y = d2d_placer.dreamplace.dp_2d.pos[d2d_placer.dreamplace.dp_2d.
                                               placedb.num_nodes:].clone()

    def swap_pos(u_index, v_index):

        pos_2d_x[u_index], pos_2d_x[v_index] = pos_2d_x[v_index].clone(
        ), pos_2d_x[u_index].clone()
        pos_2d_y[u_index], pos_2d_y[v_index] = pos_2d_y[v_index].clone(
        ), pos_2d_y[u_index].clone()

        pos_2d_tmp = torch.cat([pos_2d_x, pos_2d_y])

        d2d_placer.tier[u_index], d2d_placer.tier[v_index] = d2d_placer.tier[
            v_index].clone(), d2d_placer.tier[u_index].clone()
        return pos_2d_tmp

    # d2d_placer.params.terminal.global_place_flag = False

    for i in k:
        d2d_placer.tier = tier_init.clone()

        swapped_nodes = set()
        idx_t0 = [int(x) for x in torch.where(tier_init == 0)[0].tolist()]
        idx_t1 = [int(x) for x in torch.where(tier_init == 1)[0].tolist()]
        num_swaps = min(i, len(idx_t0), len(idx_t1))

        for j in range(num_swaps):
            u_index = idx_t0[j]
            v_index = idx_t1[j]
            if (u_index in swapped_nodes) or (v_index in swapped_nodes):
                continue
            d2d_placer.dreamplace.dp_2d.pos = swap_pos(u_index, v_index)
            swapped_nodes.add(u_index)
            swapped_nodes.add(v_index)

        d2d_placer.cut_net_mask = d2d_placer.op_wrapper.d2d_op_collections.terminal_insert_op(
            d2d_placer.tier, d2d_placer.dreamplace.dp_2d.pos,
            d2d_placer.node_orient)
        d2d_placer.num_terminal_NIs = int(d2d_placer.cut_net_mask.sum().item())

        d2d_placer.refinement()
        hpwl_tmp = d2d_placer.die_by_die_place(global_place_flag=True,
                                               legalize_flag=False,
                                               detailed_place_flag=False,
                                               random_center_init_flag=True,
                                               ntuplace_flag=True,
                                               logger=d2d_logger)

        hpwl_gain_ratio_tmp.append((hpwl - hpwl_tmp) / hpwl)
        k_tmp.append(i)
        plt.plot(k_tmp,
                 hpwl_gain_ratio_tmp,
                 'b-o',
                 linewidth=2,
                 markersize=4,
                 label='HPWL Improvement Ratio')
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

        plt.xlim(0, max(k_tmp) * 1.05)
        plt.ylim(
            min(hpwl_gain_ratio_tmp) -
            abs(max(hpwl_gain_ratio_tmp) - min(hpwl_gain_ratio_tmp)) * 0.05,
            max(hpwl_gain_ratio_tmp) +
            abs(max(hpwl_gain_ratio_tmp) - min(hpwl_gain_ratio_tmp)) * 0.05)

        max_gain = max(hpwl_gain_ratio_tmp)
        max_gain_k = k_tmp[hpwl_gain_ratio_tmp.index(max_gain)]
        plt.axhline(y=max_gain,
                    color='red',
                    linestyle=':',
                    alpha=0.7,
                    label=f'max improvement ratio: {max_gain:.4f}')
        plt.axvline(x=max_gain_k,
                    color='red',
                    linestyle=':',
                    alpha=0.7,
                    label=f'max improvement number of swaps: {max_gain_k}')

        plt.tight_layout()
        plt.savefig('refinement_test_ratio.png', dpi=300, bbox_inches='tight')


def d2d_placer_params_set(d2d_placer):
    d2d_placer.params.result_dir_root = "refinement_test"
    return d2d_placer


if __name__ == "__main__":
    """
    @brief main function to invoke the entire placement flow.
    """

    d2d_placer = D2Dplacer(sys.argv[1])
    d2d_logger = init_log(d2d_placer.params.result_dir_root)

    d2d_placer.init_spec()

    d2d_placer.partition(d2d_logger)
