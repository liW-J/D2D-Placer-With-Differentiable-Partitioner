/*
 * @Author: JeanneWillis hi@jeannewillis.cn
 * @Date: 2025-05-26 14:47:09
 * @LastEditors: JeanneWillis hi@jeannewillis.cn
 * @LastEditTime: 2025-11-01 04:01:36
 * @FilePath: /D2D-placer/placer/ops/utils_3d/src/partition.h
 * @Description:
 */

#ifndef UTILS_3D_PARTITIONER_H
#define UTILS_3D_PARTITIONER_H

// dreamplace
#include "utility/src/torch.h"
#include "utility/src/utils.h"
// 3d-placer parser
#include "include/common.h"
#include "placer/placer.h"

PLACER_BEGIN_NAMESPACE

struct NetStatus {
  int is_cut_net;
  int tier_id;

  NetStatus(int is_cut_net = 0, int tier_id = 0)
      : is_cut_net(is_cut_net), tier_id(tier_id) {}
};

struct Partitioner {

  static inline NetStatus check_net_cut(int *tier, const int *flat_netpin,
                                        const int *netpin_start,
                                        const int *pin2node_map,
                                        int num_movable_nodes, int num_nets,
                                        int num_tiers, int net_id) {

    int is_cut_net = 0;
    int num_nodes_in_net = 0;
    int tier_id = -1;
    vector<int> node_count(num_tiers, 0);

    // count nodes in each tier for check if net is cut
    for (int tier_id = 0; tier_id < num_tiers; ++tier_id) {
      for (int pin_id = netpin_start[net_id]; pin_id < netpin_start[net_id + 1];
           ++pin_id) {
        int node_id = pin2node_map[flat_netpin[pin_id]];
        if (node_id < 0 || node_id >= num_movable_nodes) {
          continue;
        }
        if (tier[node_id] == tier_id) {
          int index_node = num_movable_nodes * tier_id + node_id;
          node_count[tier_id]++;
          num_nodes_in_net++;
        }
      }
    }

    // if all nodes are in the same tier, not cut
    if (std::find(node_count.begin(), node_count.end(), num_nodes_in_net) ==
        node_count.end()) {
      is_cut_net = 1;
    } else {
      tier_id =
          std::find(node_count.begin(), node_count.end(), num_nodes_in_net) -
          node_count.begin();
    }
    return NetStatus(is_cut_net, tier_id);
  }

  static inline std::vector<int>
  countRelatedNodesInCutNets(const int *tier, const int *flat_netpin,
                             const int *netpin_start, const int *pin2node_map,
                             int num_movable_nodes, int num_nets, int num_tiers,
                             int *cut_net_mask) {

    std::vector<int> nodes_per_tier(num_tiers, 0);

    for (int net_id = 0; net_id < num_nets; ++net_id) {
      if (cut_net_mask[net_id]) {
        for (int pin_id = netpin_start[net_id];
             pin_id < netpin_start[net_id + 1]; ++pin_id) {
          int node_id = pin2node_map[flat_netpin[pin_id]];
          if (node_id < 0 || node_id >= num_movable_nodes) {
            continue;
          }

          int node_tier = tier[node_id];
          if (node_tier < 0 || node_tier >= num_tiers) {
            continue;
          }
          nodes_per_tier[node_tier]++;
        }
      }
    }
    for (int i = 0; i < num_tiers; ++i) {
      LOG(INFO, "nodes_per_tier[%d] = %d", i, nodes_per_tier[i]);
    }

    return nodes_per_tier;
  }

  static inline int
  getCutNetMask(int *cut_net_mask, int num_nets, int num_tiers, const int *tier,
                const int *flat_netpin, const int *netpin_start,
                const int *pin2node_map, int num_movable_nodes) {
    int num_terminals = 0;
    for (int net_id = 0; net_id < num_nets; ++net_id) {
      int num_nodes_in_net = 0;
      vector<int> node_count(num_tiers, 0);

      // count nodes in each tier for check if net is cut
      for (int tier_id = 0; tier_id < num_tiers; ++tier_id) {
        for (int pin_id = netpin_start[net_id];
             pin_id < netpin_start[net_id + 1]; ++pin_id) {
          int node_id = pin2node_map[flat_netpin[pin_id]];
          if (node_id < 0 || node_id >= num_movable_nodes) {
            continue;
          }
          if (tier[node_id] == tier_id) {
            int index_node = num_movable_nodes * tier_id + node_id;
            node_count[tier_id]++;
            num_nodes_in_net++;
          }
        }
      }

      // if all nodes are in the same tier, not cut
      if (std::find(node_count.begin(), node_count.end(), num_nodes_in_net) ==
          node_count.end()) {
        cut_net_mask[net_id] = 1;
        num_terminals++;
      }else{
        cut_net_mask[net_id] = 0;
      }
    }
    return num_terminals;
  }

  template <typename T>
  static inline int computeHPWLD2D(
      const T *pin_x, const T *pin_y, const int *flat_netpin,
      const int *netpin_start, const int *pin2node_map, const int *cut_net_mask,
      int num_nets, int num_pins, int num_movable_nodes, const int *tier,
      int num_tiers, const T *terminal_x, const T *terminal_y,
      int terminal_size_x,
      int terminal_size_y, int terminal_spacing, int num_terminals,
      const std::vector<std::string> &net_names,
      const std::vector<std::string> &terminal_name, int num_threads) {
    int hpwl = 0;
    // #pragma omp parallel for num_threads(num_threads)
    for (int net_id = 0; net_id < num_nets; ++net_id) {

      if (cut_net_mask[net_id]) {

        int cur_terminal_id = -1;
        for (int terminal_id = 0; terminal_id < num_terminals; ++terminal_id) {
          if (net_names[net_id] == terminal_name[terminal_id]) {
            cur_terminal_id = terminal_id;
            break;
          }
        }

        std::vector<T> max_x(num_tiers, -std::numeric_limits<T>::max());
        std::vector<T> min_x(num_tiers, std::numeric_limits<T>::max());
        std::vector<T> max_y(num_tiers, -std::numeric_limits<T>::max());
        std::vector<T> min_y(num_tiers, std::numeric_limits<T>::max());
        T terminal_x_center, terminal_y_center;

        for (int tier_id = 0; tier_id < num_tiers; ++tier_id) {
          for (int pin_id = netpin_start[net_id];
               pin_id < netpin_start[net_id + 1]; pin_id++) {
            int node_id = pin2node_map[flat_netpin[pin_id]];
            if (node_id < 0 || node_id >= num_movable_nodes) {
              continue;
            }
            int index_pin = num_pins * tier_id + flat_netpin[pin_id];
            if (tier[node_id] == tier_id) {

              max_x[tier_id] = std::max(max_x[tier_id], pin_x[index_pin]);
              min_x[tier_id] = std::min(min_x[tier_id], pin_x[index_pin]);
              max_y[tier_id] = std::max(max_y[tier_id], pin_y[index_pin]);
              min_y[tier_id] = std::min(min_y[tier_id], pin_y[index_pin]);
            }
          }
        }

        if (cur_terminal_id == -1) {

          // get inster bonding
          auto inner_min_x_it = max_element(min_x.begin(), min_x.end());
          auto inner_max_x_it = min_element(max_x.begin(), max_x.end());
          auto inner_min_y_it = max_element(min_y.begin(), min_y.end());
          auto inner_max_y_it = min_element(max_y.begin(), max_y.end());
          T inner_min_x = *inner_min_x_it;
          T inner_max_x = *inner_max_x_it;
          T inner_min_y = *inner_min_y_it;
          T inner_max_y = *inner_max_y_it;
          terminal_x_center = (inner_min_x + inner_max_x) / 2;
          terminal_y_center = (inner_min_y + inner_max_y) / 2;

          // LOG(WARN, "net_id: %d, cur_terminal_id: %d", net_id,
          // cur_terminal_id);
        } else {
          terminal_x_center = terminal_x[cur_terminal_id] +
                              (terminal_size_x + terminal_spacing) / 2;
          terminal_y_center = terminal_y[cur_terminal_id] +
                              (terminal_size_y + terminal_spacing) / 2;
        }

        for (int tier_id = 0; tier_id < num_tiers; ++tier_id) {
          // LOG(WARN, "tier_id: %d, terminal_x_center: %f, terminal_y_center:
          // %f", tier_id, terminal_x_center, terminal_y_center);
          max_x[tier_id] = std::max(max_x[tier_id], terminal_x_center);
          min_x[tier_id] = std::min(min_x[tier_id], terminal_x_center);
          max_y[tier_id] = std::max(max_y[tier_id], terminal_y_center);
          min_y[tier_id] = std::min(min_y[tier_id], terminal_y_center);
          hpwl +=
              max_x[tier_id] - min_x[tier_id] + max_y[tier_id] - min_y[tier_id];
          // LOG(INFO, "net_id: %d, tier_id: %d, hpwl: %f", net_id, tier_id,
          // hpwl[net_id]);
        }
      } else {
        T max_x = -std::numeric_limits<T>::max();
        T min_x = std::numeric_limits<T>::max();
        T max_y = -std::numeric_limits<T>::max();
        T min_y = std::numeric_limits<T>::max();
        int tier_id = tier[pin2node_map[flat_netpin[netpin_start[net_id]]]];

        for (int pin_id = netpin_start[net_id];
             pin_id < netpin_start[net_id + 1]; pin_id++) {
          int index_pin = num_pins * tier_id + flat_netpin[pin_id];
          // LOG(WARN, "pin_x: %f, pin_y: %f", pin_x[index_pin],
          // pin_y[index_pin]);
          min_x = std::min(min_x, pin_x[index_pin]);
          max_x = std::max(max_x, pin_x[index_pin]);
          min_y = std::min(min_y, pin_y[index_pin]);
          max_y = std::max(max_y, pin_y[index_pin]);
        }
        hpwl += max_x - min_x + max_y - min_y;
        // LOG(INFO, "net_id: %d, tier_id: %d, hpwl: %f", net_id, tier_id,
        // hpwl[net_id]);
      }
    }

    return hpwl;
  }
};

PLACER_END_NAMESPACE

#endif
