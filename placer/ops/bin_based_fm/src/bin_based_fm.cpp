/*
 * @Author: JeanneWillis hi@jeannewillis.cn
 * @Date: 2025-09-21 17:01:58
 * @LastEditors: JeanneWillis hi@jeannewillis.cn
 * @LastEditTime: 2025-10-07 21:56:08
 * @FilePath: /D2D-placer/placer/ops/bin_based_fm/src/bin_based_fm.cpp
 * @Description: fm refinement
 */

#include <pybind11/pybind11.h>
#include <queue>
#include <unordered_set>
// dreamplace
#include "utility/src/torch.h"
#include "utility/src/utils.h"

// 3d-placer parser
#include "include/common.h"
#include "placer/placer.h"

#include "utils_3d/src/partitioner.h"

PLACER_BEGIN_NAMESPACE

struct Item {
  int gain;
  int node_id;
};

template <typename T>
void binBasedFMLauncher(
    int *tier, const int *flat_netpin, const int *netpin_start,
    const int *pin2node_map, int num_nets, int num_tiers, int num_movable_nodes,
    int num_pins, const T *node_size_x, const T *node_size_y,
    const T *pin_offset_x, const T *pin_offset_y, float die_size_x,
    float die_size_y, pybind11::list row_height, int terminal_size_x,
    int terminal_size_y, int terminal_spacing, int *cut_net_mask,
    const T *pin_x, const T *pin_y, const std::vector<std::string> &node_names,
    const std::vector<std::string> &net_names, const T *pos_2d_x,
    const T *pos_2d_y, const T *terminal_x, const T *terminal_y,
    std::string case_name, int num_terminals,
    const std::vector<std::string> &terminal_names, float top_die_max_util,
    float bottom_die_max_util, int num_threads) {

  int num_terminals_tmp = Partitioner::getCutNetMask(
      cut_net_mask, num_nets, num_tiers, tier, flat_netpin, netpin_start,
      pin2node_map, num_movable_nodes);

  // build node -> nets adjacency
  std::vector<std::vector<int>> node_to_nets(num_movable_nodes);
  for (int net_id = 0; net_id < num_nets; ++net_id) {
    for (int pin_id = netpin_start[net_id]; pin_id < netpin_start[net_id + 1];
         ++pin_id) {
      int node_id = pin2node_map[flat_netpin[pin_id]];
      if (node_id >= 0 && node_id < num_movable_nodes) {
        node_to_nets[node_id].push_back(net_id);
      }
    }
  }

  // FM database
  std::vector<std::vector<int>> net_to_nodes(num_nets);
  for (int net_id = 0; net_id < num_nets; ++net_id) {
    for (int pin_id = netpin_start[net_id]; pin_id < netpin_start[net_id + 1];
         ++pin_id) {
      int node_id = pin2node_map[flat_netpin[pin_id]];
      if (node_id >= 0 && node_id < num_movable_nodes) {
        net_to_nodes[net_id].push_back(node_id);
      }
    }
  }

  std::vector<std::vector<int>> net_count(num_nets,
                                          std::vector<int>(num_tiers, 0));
  for (int net_id = 0; net_id < num_nets; ++net_id) {
    for (int node_id : net_to_nodes[net_id]) {
      int t = tier[node_id];
      net_count[net_id][t]++;
    }
    // sync cut_net_mask
    bool is_cut = false;
    int total = 0;
    for (int t = 0; t < num_tiers; ++t)
      total += net_count[net_id][t];
    for (int t = 0; t < num_tiers; ++t) {
      if (net_count[net_id][t] == total) {
        is_cut = false;
        break;
      }
      if (net_count[net_id][t] > 0 && net_count[net_id][t] < total) {
        is_cut = true;
      }
    }
    cut_net_mask[net_id] = is_cut ? 1 : 0;
  }

  auto compute_single_net_hpwl = [&](int net_id, const int *tier_ptr) -> int {
    int hpwl_net = 0;
    // check if the net is cut: if cut_net_mask is prepared, it can be directly
    // used; but this lambda supports recalculating the HPWL of the net assuming
    // the tier state here we directly recalculate the HPWL of the net assuming
    // the tier state first, we collect the bbox of each tier
    std::vector<T> max_x(num_tiers, -std::numeric_limits<T>::max());
    std::vector<T> min_x(num_tiers, std::numeric_limits<T>::max());
    std::vector<T> max_y(num_tiers, -std::numeric_limits<T>::max());
    std::vector<T> min_y(num_tiers, std::numeric_limits<T>::max());

    // collect the tiers involved in the net
    std::vector<int> node_count(num_tiers, 0);
    int num_nodes_in_net = 0;
    for (int pin_id = netpin_start[net_id]; pin_id < netpin_start[net_id + 1];
         ++pin_id) {
      int node_id = pin2node_map[flat_netpin[pin_id]];
      int t = tier_ptr[node_id];
      int index_pin = num_pins * t + flat_netpin[pin_id];
      max_x[t] = std::max(max_x[t], pin_x[index_pin]);
      min_x[t] = std::min(min_x[t], pin_x[index_pin]);
      max_y[t] = std::max(max_y[t], pin_y[index_pin]);
      min_y[t] = std::min(min_y[t], pin_y[index_pin]);
      node_count[t]++;
      num_nodes_in_net++;
    }

    // check if the net is bound to a terminal
    int cur_terminal_id = -1;
    for (int terminal_id = 0; terminal_id < num_terminals; ++terminal_id) {
      if (net_names[net_id] == terminal_names[terminal_id]) {
        cur_terminal_id = terminal_id;
        break;
      }
    }

    // check if the net is cut
    bool is_cut = true;
    for (int t = 0; t < num_tiers; ++t) {
      if (node_count[t] == num_nodes_in_net) {
        is_cut = false;
        break;
      }
    }

    if (is_cut) {
      T terminal_x_center, terminal_y_center;
      if (cur_terminal_id == -1) {
        // inner cross-tier bonding point
        auto inner_min_x_it = std::max_element(min_x.begin(), min_x.end());
        auto inner_max_x_it = std::min_element(max_x.begin(), max_x.end());
        auto inner_min_y_it = std::max_element(min_y.begin(), min_y.end());
        auto inner_max_y_it = std::min_element(max_y.begin(), max_y.end());
        T inner_min_x = *inner_min_x_it;
        T inner_max_x = *inner_max_x_it;
        T inner_min_y = *inner_min_y_it;
        T inner_max_y = *inner_max_y_it;
        terminal_x_center = (inner_min_x + inner_max_x) / 2;
        terminal_y_center = (inner_min_y + inner_max_y) / 2;
      } else {
        terminal_x_center = terminal_x[cur_terminal_id] +
                            (terminal_size_x + terminal_spacing) / 2;
        terminal_y_center = terminal_y[cur_terminal_id] +
                            (terminal_size_y + terminal_spacing) / 2;
      }
      for (int t = 0; t < num_tiers; ++t) {
        max_x[t] = std::max(max_x[t], terminal_x_center);
        min_x[t] = std::min(min_x[t], terminal_x_center);
        max_y[t] = std::max(max_y[t], terminal_y_center);
        min_y[t] = std::min(min_y[t], terminal_y_center);
        hpwl_net += static_cast<int>(max_x[t] - min_x[t] + max_y[t] - min_y[t]);
      }
    } else {
      int t_single = -1;
      for (int t = 0; t < num_tiers; ++t)
        if (node_count[t] == num_nodes_in_net) {
          t_single = t;
          break;
        }
      if (t_single >= 0) {
        hpwl_net += static_cast<int>(max_x[t_single] - min_x[t_single] +
                                     max_y[t_single] - min_y[t_single]);
      }
    }
    return hpwl_net;
  };

  // 缓存每条 net 的 HPWL 贡献，并用其求和校准全局 hpwl
  std::vector<int> net_hpwl(num_nets, 0);
  int hpwl_sum = 0;
  for (int net_id = 0; net_id < num_nets; ++net_id) {
    net_hpwl[net_id] = compute_single_net_hpwl(net_id, tier);
    hpwl_sum += net_hpwl[net_id];
  }

  auto compute_node_gain = [&](int node_id) -> int {
    // 仅对受影响的 nets 做增量 HPWL 计算
    int hpwl_delta = 0;
    int old_tier = tier[node_id];
    tier[node_id] = 1 - old_tier;
    for (int net_id : node_to_nets[node_id]) {
      int hpwl_before = net_hpwl[net_id];
      int hpwl_after = compute_single_net_hpwl(net_id, tier);
      hpwl_delta += (hpwl_before - hpwl_after);
    }
    tier[node_id] = old_tier;

    // terminal 相关保持原逻辑
    std::vector<int> tier_tmp(tier, tier + num_movable_nodes);
    std::vector<int> cut_net_mask_tmp(num_nets, 0);
    tier_tmp[node_id] = 1 - tier_tmp[node_id];
    int num_terminals_new = Partitioner::getCutNetMask(
        cut_net_mask_tmp.data(), num_nets, num_tiers, tier_tmp.data(),
        flat_netpin, netpin_start, pin2node_map, num_movable_nodes);

    int terminal_gain = num_terminals_tmp - num_terminals_new;
    int g =
        (hpwl_delta + 1000 * terminal_gain) * (num_nets - num_terminals_new);
    return g;
  };

  // helper: rebuild net_count and cut mark based on current tier
  auto rebuild_counts_and_cut = [&]() {
    for (int net_id = 0; net_id < num_nets; ++net_id) {
      std::fill(net_count[net_id].begin(), net_count[net_id].end(), 0);
      for (int node_id : net_to_nodes[net_id]) {
        int t = tier[node_id];
        if (t >= 0 && t < num_tiers)
          net_count[net_id][t]++;
      }
      int total = 0;
      for (int t = 0; t < num_tiers; ++t)
        total += net_count[net_id][t];
      bool is_cut = true;
      for (int t = 0; t < num_tiers; ++t) {
        if (net_count[net_id][t] == total) {
          is_cut = false;
          break;
        }
      }
      cut_net_mask[net_id] = is_cut ? 1 : 0;
    }
  };
  const double die_area =
      static_cast<double>(die_size_x) * static_cast<double>(die_size_y);
  std::vector<double> tier_area(num_tiers, 0.0);

  bool pass_flag = true;
  while (pass_flag) {

    for (int n = 0; n < num_movable_nodes; ++n) {
      int t = tier[n];
      assert(t >= 0 && t < num_tiers);
      double node_area = static_cast<double>(node_size_x[n]) *
                         static_cast<double>(node_size_y[n]);
      tier_area[t] += node_area;
    }
    std::vector<int> gain(num_movable_nodes, 0);

    auto cmp = [](const Item &a, const Item &b) { return a.gain < b.gain; };
    std::priority_queue<Item, std::vector<Item>, decltype(cmp)> pq(cmp);
    for (int node_id = 0; node_id < num_movable_nodes; ++node_id) {
      gain[node_id] = compute_node_gain(node_id);
      pq.push({gain[node_id], node_id});
    }

    // clear locked each pass
    std::vector<char> locked(num_movable_nodes, 0);
    std::vector<int> move_order;
    move_order.reserve(num_movable_nodes);
    std::vector<int> move_from;
    move_from.reserve(num_movable_nodes);
    std::vector<int> move_gain;
    move_gain.reserve(num_movable_nodes);

    int cumulative_gain = 0;
    int best_prefix_gain = 0;
    int best_prefix_idx = -1;

    // switch until all nodes are locked
    for (int step = 0; step < num_movable_nodes; ++step) {
      int pick = -1;
      int pick_gain = std::numeric_limits<int>::min();
      while (!pq.empty()) {
        Item it = pq.top();
        pq.pop();
        int node_id = it.node_id;
        if (locked[node_id] || gain[node_id] != it.gain) {
          continue;
        }
        // hard constraint check: whether the area exceeds the limit
        int from_t_chk = tier[node_id];
        int to_t_chk = 1 - from_t_chk;
        double area_u = static_cast<double>(node_size_x[node_id]) *
                        static_cast<double>(node_size_y[node_id]);
        double to_after = tier_area[to_t_chk] + area_u;
        float max_util = to_t_chk == 0 ? top_die_max_util : bottom_die_max_util;
        if (to_after > max_util * die_area) {
          continue;
        }
        pick = node_id;
        pick_gain = it.gain;
        break;
      }
      if (pick == -1)
        break;

      int node_id = pick;
      int from_t = tier[node_id];
      int to_t = 1 - from_t;

      // execute move and lock
      locked[node_id] = 1;
      tier[node_id] = to_t;
      // sync area count
      double area_u = static_cast<double>(node_size_x[node_id]) *
                      static_cast<double>(node_size_y[node_id]);
      tier_area[from_t] -= area_u;
      tier_area[to_t] += area_u;
      move_order.push_back(node_id);
      move_from.push_back(from_t);
      move_gain.push_back(pick_gain);
      cumulative_gain += pick_gain;
      if (cumulative_gain > best_prefix_gain) {
        best_prefix_gain = cumulative_gain;
        best_prefix_idx = static_cast<int>(move_order.size()) - 1;
      }

      // update net count and cut mark of the node, and collect affected
      std::unordered_set<int> affected;
      for (int net_id : node_to_nets[node_id]) {
        // collect all nodes on the net
        for (int v : net_to_nodes[net_id]) {
          if (!locked[v] && v != node_id)
            affected.insert(v);
        }
        // update net count of the node
        net_count[net_id][from_t]--;
        net_count[net_id][to_t]++;
        // recompute cut mark
        int total = net_count[net_id][0] + net_count[net_id][1];
        bool now_cut =
            !(net_count[net_id][0] == total || net_count[net_id][1] == total);
        cut_net_mask[net_id] = now_cut ? 1 : 0;
      }

      // update hpwl_sum and net_hpwl by the node
      int hpwl_delta_apply = 0;
      for (int net_id : node_to_nets[node_id]) {
        int before = net_hpwl[net_id];
        int after = compute_single_net_hpwl(net_id, tier);
        net_hpwl[net_id] = after;
        hpwl_delta_apply += (after - before);
      }
      hpwl_sum += hpwl_delta_apply;

      num_terminals_tmp = Partitioner::getCutNetMask(
          cut_net_mask, num_nets, num_tiers, tier, flat_netpin, netpin_start,
          pin2node_map, num_movable_nodes);

      // recompute gain based on HPWL for affected neighbors, and update
      // priority queue
      for (int v : affected) {
        int old = gain[v];
        gain[v] = compute_node_gain(v);
        if (gain[v] != old) {
          pq.push({gain[v], v});
        }
      }
    }

    // rollback to best prefix
    for (int i = static_cast<int>(move_order.size()) - 1; i > best_prefix_idx;
         --i) {
      int rollback_id = move_order[i];
      int from_t = move_from[i];
      int to_t = 1 - from_t;
      tier[rollback_id] = from_t;
      // rollback area count
      double area_u = static_cast<double>(node_size_x[rollback_id]) *
                      static_cast<double>(node_size_y[rollback_id]);
      tier_area[from_t] += area_u;
      tier_area[to_t] -= area_u;

      for (int net_id : node_to_nets[rollback_id]) {
        net_count[net_id][from_t]++;
        net_count[net_id][to_t]--;
        int total = net_count[net_id][0] + net_count[net_id][1];
        bool now_cut =
            !(net_count[net_id][0] == total || net_count[net_id][1] == total);
        cut_net_mask[net_id] = now_cut ? 1 : 0;
      }
    }

    // update net_count/cut and hpwl with new tier
    rebuild_counts_and_cut();

    num_terminals_tmp = Partitioner::getCutNetMask(
        cut_net_mask, num_nets, num_tiers, tier, flat_netpin, netpin_start,
        pin2node_map, num_movable_nodes);
    LOG(INFO, "num_terminals_tmp: %d", num_terminals_tmp);

    hpwl_sum = 0;
    for (int net_id = 0; net_id < num_nets; ++net_id) {
      net_hpwl[net_id] = compute_single_net_hpwl(net_id, tier);
      hpwl_sum += net_hpwl[net_id];
    }
    LOG(INFO, "hpwl: %d", hpwl_sum);

    pass_flag = best_prefix_gain > 0 ? true : false;
  }
}

at::Tensor bin_based_fm_forward(
    at::Tensor tier, at::Tensor flat_netpin, at::Tensor netpin_start,
    at::Tensor pin2node_map, at::Tensor net_weights, int num_movable_nodes,
    at::Tensor node_size_x, at::Tensor node_size_y, at::Tensor pin_offset_x,
    at::Tensor pin_offset_y, float die_size_x, float die_size_y,
    pybind11::list row_height, int terminal_size_x, int terminal_size_y,
    int terminal_spacing, at::Tensor pin_pos,
    const std::vector<std::string> &node_names,
    const std::vector<std::string> &net_names, at::Tensor pos_2d,
    at::Tensor pos_terminal_legalized, std::string case_name, int num_terminals,
    const std::vector<std::string> &terminal_names, float top_die_max_util,
    float bottom_die_max_util) {
  CHECK_FLAT_CPU(flat_netpin);
  CHECK_CONTIGUOUS(flat_netpin);
  CHECK_FLAT_CPU(netpin_start);
  CHECK_CONTIGUOUS(netpin_start);
  CHECK_FLAT_CPU(pin2node_map);
  CHECK_CONTIGUOUS(pin2node_map);
  CHECK_FLAT_CPU(net_weights);
  CHECK_CONTIGUOUS(net_weights);

  CHECK_CONTIGUOUS(pin_offset_x);
  CHECK_CONTIGUOUS(pin_offset_y);
  CHECK_CONTIGUOUS(node_size_x);
  CHECK_CONTIGUOUS(node_size_y);

  CHECK_CONTIGUOUS(tier);

  CHECK_EVEN(pin_pos);
  CHECK_CONTIGUOUS(pin_pos);
  CHECK_FLAT_CPU(pos_2d);
  CHECK_EVEN(pos_2d);
  CHECK_CONTIGUOUS(pos_2d);

  int num_nets = netpin_start.numel() - 1;
  int num_pins = pin2node_map.numel();

  // TODO: get num_tiers from param.json
  int num_tiers = 2;
  at::Tensor cut_net_mask = at::zeros(num_nets, tier.options());

  DREAMPLACE_DISPATCH_FLOATING_TYPES(node_size_x, "binBasedFMLauncher", [&] {
    binBasedFMLauncher<scalar_t>(
        DREAMPLACE_TENSOR_DATA_PTR(tier, int),
        DREAMPLACE_TENSOR_DATA_PTR(flat_netpin, int),
        DREAMPLACE_TENSOR_DATA_PTR(netpin_start, int),
        DREAMPLACE_TENSOR_DATA_PTR(pin2node_map, int), num_nets, num_tiers,
        num_movable_nodes, num_pins,
        DREAMPLACE_TENSOR_DATA_PTR(node_size_x, scalar_t),
        DREAMPLACE_TENSOR_DATA_PTR(node_size_y, scalar_t),
        DREAMPLACE_TENSOR_DATA_PTR(pin_offset_x, scalar_t),
        DREAMPLACE_TENSOR_DATA_PTR(pin_offset_y, scalar_t), die_size_x,
        die_size_y, row_height, terminal_size_x, terminal_size_y,
        terminal_spacing, DREAMPLACE_TENSOR_DATA_PTR(cut_net_mask, int),
        DREAMPLACE_TENSOR_DATA_PTR(pin_pos, scalar_t),
        DREAMPLACE_TENSOR_DATA_PTR(pin_pos, scalar_t) + pin_pos.numel() / 2,
        node_names, net_names, DREAMPLACE_TENSOR_DATA_PTR(pos_2d, scalar_t),
        DREAMPLACE_TENSOR_DATA_PTR(pos_2d, scalar_t) + pos_2d.numel() / 2,
        DREAMPLACE_TENSOR_DATA_PTR(pos_terminal_legalized, scalar_t),
        DREAMPLACE_TENSOR_DATA_PTR(pos_terminal_legalized, scalar_t) +
            pos_terminal_legalized.numel() / 2,
        case_name, num_terminals, terminal_names, top_die_max_util,
        bottom_die_max_util, at::get_num_threads());
  });

  return tier;
}

PLACER_END_NAMESPACE

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("bin_based_fm", &PLACER_NAMESPACE::bin_based_fm_forward,
        "bin_based_fm_forward");
}
