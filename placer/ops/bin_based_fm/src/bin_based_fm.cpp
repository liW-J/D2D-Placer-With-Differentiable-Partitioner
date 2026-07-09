/*
 * @Author: JeanneWillis hi@jeannewillis.cn
 * @Date: 2025-09-21 17:01:58
 * @LastEditors: JeanneWillis hi@jeannewillis.cn
 * @LastEditTime: 2025-11-09 16:07:40
 * @FilePath: /D2D-placer/placer/ops/bin_based_fm/src/bin_based_fm.cpp
 * @Description: fm refinement
 */

#include <pybind11/pybind11.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <climits>
#include <iostream>
#include <map>
#include <queue>
#include <string>
#include <stdexcept>
// dreamplace
#include "utility/src/torch.h"
#include "utility/src/utils.h"

// 3d-placer parser
#include "include/common.h"
#include "placer/placer.h"

#include "bin_based_fm/src/density_map.h"
#include "utils_3d/src/partitioner.h"

PLACER_BEGIN_NAMESPACE

struct Item {
  long long gain;
  int node_id;
};
struct ItemComparator {
  bool operator()(const Item& a, const Item& b) const {
    return a.gain < b.gain;
  }
};

template <typename T>
void binBasedFMLauncher(
    int* tier, const int* flat_netpin, const int* netpin_start, const int* pin2node_map,
    int num_nets, int num_tiers, int num_movable_nodes, int num_pins, const T* node_size_x,
    const T* node_size_y, const T* pin_offset_x, const T* pin_offset_y, float die_size_x,
    float die_size_y, pybind11::list row_height, int terminal_size_x, int terminal_size_y,
    int terminal_spacing, int* cut_net_mask, const T* pin_x, const T* pin_y,
    const std::vector<std::string>& node_names, const std::vector<std::string>& net_names,
    const T* pos_2d_x, const T* pos_2d_y, const T* terminal_x, const T* terminal_y,
    std::string case_name, int num_terminals, const std::vector<std::string>& terminal_names,
    float top_die_max_util, float bottom_die_max_util, int num_threads, int num_nodes,
    const int num_bins_x, const int num_bins_y, const T xl, const T yl, const T xh, const T yh) {

  if (num_tiers <= 0 || num_bins_x <= 0 || num_bins_y <= 0 || die_size_x <= 0 ||
      die_size_y <= 0) {
    throw std::runtime_error("bin_based_fm received invalid dimensions");
  }
  for (int node_id = 0; node_id < num_movable_nodes; ++node_id) {
    int t = tier[node_id];
    if (t < 0 || t >= num_tiers) {
      throw std::runtime_error("bin_based_fm received tier id outside valid range");
    }
  }

  auto movable_node_for_pin = [&](int flat_pin_id) -> int {
    if (flat_pin_id < 0 || flat_pin_id >= num_pins) {
      return -1;
    }
    int node_id = pin2node_map[flat_pin_id];
    if (node_id < 0 || node_id >= num_movable_nodes) {
      return -1;
    }
    return node_id;
  };

  auto tier_of_node = [&](int node_id, const int* tier_ptr) -> int {
    if (node_id < 0 || node_id >= num_movable_nodes) {
      return -1;
    }
    int t = tier_ptr[node_id];
    if (t < 0 || t >= num_tiers) {
      return -1;
    }
    return t;
  };

  int num_terminals_tmp = 0;

  int num_bins = num_bins_x * num_bins_y;
  DREAMPLACE_NAMESPACE::AtomicAdd<double> atomic_add_op;
  std::vector<double> buf_map_tier(num_bins * num_tiers, 0);
  computeDensityMapLauncher(pos_2d_x, pos_2d_y, node_size_x, node_size_y, num_movable_nodes,
                            num_bins_x, num_bins_y, xl, yl, xh, yh, num_threads, atomic_add_op,
                            buf_map_tier.data(), tier, num_tiers);

  std::vector<double> buf_map_terminal(num_bins, 0);
  std::vector<T> terminal_node_size_x(num_terminals, static_cast<T>(terminal_size_x));
  std::vector<T> terminal_node_size_y(num_terminals, static_cast<T>(terminal_size_y));
  std::vector<int> terminal_tier(num_terminals, 0);
  computeDensityMapLauncher(terminal_x, terminal_y, terminal_node_size_x.data(),
                            terminal_node_size_y.data(), num_terminals, num_bins_x, num_bins_y, xl,
                            yl, xh, yh, num_threads, atomic_add_op, buf_map_terminal.data(),
                            terminal_tier.data(), 1);

  // build node -> nets adjacency
  std::vector<std::vector<int>> node_to_nets(num_movable_nodes);
  for (int net_id = 0; net_id < num_nets; ++net_id) {
    for (int pin_id = netpin_start[net_id]; pin_id < netpin_start[net_id + 1]; ++pin_id) {
      int node_id = movable_node_for_pin(flat_netpin[pin_id]);
      if (node_id >= 0) {
        node_to_nets[node_id].push_back(net_id);
      }
    }
  }

  std::vector<std::vector<int>> net_to_nodes(num_nets);
  for (int net_id = 0; net_id < num_nets; ++net_id) {
    for (int pin_id = netpin_start[net_id]; pin_id < netpin_start[net_id + 1]; ++pin_id) {
      int node_id = movable_node_for_pin(flat_netpin[pin_id]);
      if (node_id >= 0) {
        net_to_nodes[net_id].push_back(node_id);
      }
    }
  }
  for (auto& nets : node_to_nets) {
    std::sort(nets.begin(), nets.end());
    nets.erase(std::unique(nets.begin(), nets.end()), nets.end());
  }
  for (auto& nodes : net_to_nodes) {
    std::sort(nodes.begin(), nodes.end());
    nodes.erase(std::unique(nodes.begin(), nodes.end()), nodes.end());
  }

  // Track per-net tier counts so cut status can be updated incrementally.
  std::vector<std::array<int, 2>> net_tier_count(num_nets);
  auto recompute_cut_state = [&]() -> int {
    int cut_count = 0;
    for (int net_id = 0; net_id < num_nets; ++net_id) {
      net_tier_count[net_id] = {0, 0};
      for (int node_id : net_to_nodes[net_id]) {
        int t = tier_of_node(node_id, tier);
        if (t >= 0) {
          net_tier_count[net_id][t]++;
        }
      }
      int is_cut = (net_tier_count[net_id][0] > 0 && net_tier_count[net_id][1] > 0) ? 1 : 0;
      cut_net_mask[net_id] = is_cut;
      cut_count += is_cut;
    }
    return cut_count;
  };
  num_terminals_tmp = recompute_cut_state();

  auto is_cut_after_move = [&](int net_id, int from_t, int to_t) -> int {
    int from_count = net_tier_count[net_id][from_t] - 1;
    int to_count = net_tier_count[net_id][to_t] + 1;
    int tier0_count = (from_t == 0) ? from_count : to_count;
    int tier1_count = (from_t == 1) ? from_count : to_count;
    return (tier0_count > 0 && tier1_count > 0) ? 1 : 0;
  };

  // Cache terminal binding lookup once to avoid string matching in hot loops.
  std::vector<int> net_to_terminal_id(num_nets, -1);
  {
    std::map<std::string, int> terminal_name_to_id;
    for (int terminal_id = 0; terminal_id < num_terminals; ++terminal_id) {
      terminal_name_to_id[terminal_names[terminal_id]] = terminal_id;
    }
    for (int net_id = 0; net_id < num_nets; ++net_id) {
      auto it = terminal_name_to_id.find(net_names[net_id]);
      if (it != terminal_name_to_id.end()) {
        net_to_terminal_id[net_id] = it->second;
      }
    }
  }

  // Cache node area for repeated utilization checks/updates.
  std::vector<double> node_area_cache(num_movable_nodes, 0.0);
  for (int node_id = 0; node_id < num_movable_nodes; ++node_id) {
    node_area_cache[node_id] =
        static_cast<double>(node_size_x[node_id]) * static_cast<double>(node_size_y[node_id]);
  }

  auto get_terminal_center = [&](int net_id, const int* tier_ptr) -> std::pair<T, T> {
    std::vector<T> max_x(num_tiers, -std::numeric_limits<T>::max());
    std::vector<T> min_x(num_tiers, std::numeric_limits<T>::max());
    std::vector<T> max_y(num_tiers, -std::numeric_limits<T>::max());
    std::vector<T> min_y(num_tiers, std::numeric_limits<T>::max());
    T terminal_x_center, terminal_y_center;

    for (int pin_id = netpin_start[net_id]; pin_id < netpin_start[net_id + 1]; ++pin_id) {
      int flat_pin_id = flat_netpin[pin_id];
      int node_id = movable_node_for_pin(flat_pin_id);
      int t = tier_of_node(node_id, tier_ptr);
      if (t < 0) {
        continue;
      }
      int index_pin = num_pins * t + flat_pin_id;
      max_x[t] = std::max(max_x[t], pin_x[index_pin]);
      min_x[t] = std::min(min_x[t], pin_x[index_pin]);
      max_y[t] = std::max(max_y[t], pin_y[index_pin]);
      min_y[t] = std::min(min_y[t], pin_y[index_pin]);
    }

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
    return std::make_pair(terminal_x_center, terminal_y_center);
  };

  auto get_gain = [&](int net_id, const int* tier_ptr) -> std::pair<T, T> {
    std::vector<T> max_x(num_tiers, -std::numeric_limits<T>::max());
    std::vector<T> min_x(num_tiers, std::numeric_limits<T>::max());
    std::vector<T> max_y(num_tiers, -std::numeric_limits<T>::max());
    std::vector<T> min_y(num_tiers, std::numeric_limits<T>::max());
    T terminal_x_center, terminal_y_center;

    for (int pin_id = netpin_start[net_id]; pin_id < netpin_start[net_id + 1]; ++pin_id) {
      int flat_pin_id = flat_netpin[pin_id];
      int node_id = movable_node_for_pin(flat_pin_id);
      int t = tier_of_node(node_id, tier_ptr);
      if (t < 0) {
        continue;
      }
      int index_pin = num_pins * t + flat_pin_id;
      max_x[t] = std::max(max_x[t], pin_x[index_pin]);
      min_x[t] = std::min(min_x[t], pin_x[index_pin]);
      max_y[t] = std::max(max_y[t], pin_y[index_pin]);
      min_y[t] = std::min(min_y[t], pin_y[index_pin]);
    }

    auto inner_min_x_it = std::max_element(min_x.begin(), min_x.end());
    auto inner_max_x_it = std::min_element(max_x.begin(), max_x.end());
    auto inner_min_y_it = std::max_element(min_y.begin(), min_y.end());
    auto inner_max_y_it = std::min_element(max_y.begin(), max_y.end());
    T inner_min_x = *inner_min_x_it;
    T inner_max_x = *inner_max_x_it;
    T inner_min_y = *inner_min_y_it;
    T inner_max_y = *inner_max_y_it;
    T x_gain = inner_min_x - inner_max_x;
    T y_gain = inner_min_y - inner_max_y;
    // LOG(INFO, "x_gain: %f, y_gain: %f", x_gain, y_gain);
    return std::make_pair(x_gain, y_gain);
  };

  auto compute_single_net_hpwl = [&](int net_id, const int* tier_ptr,
                                     const int* cut_net_mask_ptr) -> long long {
    long long hpwl_net = 0;
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
    for (int pin_id = netpin_start[net_id]; pin_id < netpin_start[net_id + 1]; ++pin_id) {
      int flat_pin_id = flat_netpin[pin_id];
      int node_id = movable_node_for_pin(flat_pin_id);
      int t = tier_of_node(node_id, tier_ptr);
      if (t < 0) {
        continue;
      }
      int index_pin = num_pins * t + flat_pin_id;
      max_x[t] = std::max(max_x[t], pin_x[index_pin]);
      min_x[t] = std::min(min_x[t], pin_x[index_pin]);
      max_y[t] = std::max(max_y[t], pin_y[index_pin]);
      min_y[t] = std::min(min_y[t], pin_y[index_pin]);
      node_count[t]++;
      num_nodes_in_net++;
    }
    if (num_nodes_in_net == 0) {
      return 0;
    }

    // check if the net is bound to a terminal
    int cur_terminal_id = net_to_terminal_id[net_id];

    if (cut_net_mask_ptr[net_id]) {
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
        terminal_x_center = terminal_x[cur_terminal_id] + (terminal_size_x + terminal_spacing) / 2;
        terminal_y_center = terminal_y[cur_terminal_id] + (terminal_size_y + terminal_spacing) / 2;
      }
      for (int t = 0; t < num_tiers; ++t) {
        if (node_count[t] == 0) {
          continue;
        }
        max_x[t] = std::max(max_x[t], terminal_x_center);
        min_x[t] = std::min(min_x[t], terminal_x_center);
        max_y[t] = std::max(max_y[t], terminal_y_center);
        min_y[t] = std::min(min_y[t], terminal_y_center);
        hpwl_net += static_cast<long long>(max_x[t] - min_x[t] + max_y[t] - min_y[t]);
      }
    } else {
      int t_single = -1;
      for (int t = 0; t < num_tiers; ++t)
        if (node_count[t] == num_nodes_in_net) {
          t_single = t;
          break;
        }
      if (t_single >= 0) {
        hpwl_net +=
            static_cast<long long>(max_x[t_single] - min_x[t_single] + max_y[t_single] - min_y[t_single]);
      }
    }
    return hpwl_net;
  };

  auto compute_single_net_hpwl_after_move = [&](int net_id, int moved_node_id, int moved_to_t,
                                                int cut_status_after) -> long long {
    long long hpwl_net = 0;
    std::vector<T> max_x(num_tiers, -std::numeric_limits<T>::max());
    std::vector<T> min_x(num_tiers, std::numeric_limits<T>::max());
    std::vector<T> max_y(num_tiers, -std::numeric_limits<T>::max());
    std::vector<T> min_y(num_tiers, std::numeric_limits<T>::max());
    std::vector<int> node_count(num_tiers, 0);
    int num_nodes_in_net = 0;

    for (int pin_id = netpin_start[net_id]; pin_id < netpin_start[net_id + 1]; ++pin_id) {
      int flat_pin_id = flat_netpin[pin_id];
      int node_id = movable_node_for_pin(flat_pin_id);
      int t = (node_id == moved_node_id) ? moved_to_t : tier_of_node(node_id, tier);
      if (t < 0 || t >= num_tiers) {
        continue;
      }
      int index_pin = num_pins * t + flat_pin_id;
      max_x[t] = std::max(max_x[t], pin_x[index_pin]);
      min_x[t] = std::min(min_x[t], pin_x[index_pin]);
      max_y[t] = std::max(max_y[t], pin_y[index_pin]);
      min_y[t] = std::min(min_y[t], pin_y[index_pin]);
      node_count[t]++;
      num_nodes_in_net++;
    }
    if (num_nodes_in_net == 0) {
      return 0;
    }

    int cur_terminal_id = net_to_terminal_id[net_id];
    if (cut_status_after) {
      T terminal_x_center, terminal_y_center;
      if (cur_terminal_id == -1) {
        auto inner_min_x_it = std::max_element(min_x.begin(), min_x.end());
        auto inner_max_x_it = std::min_element(max_x.begin(), max_x.end());
        auto inner_min_y_it = std::max_element(min_y.begin(), min_y.end());
        auto inner_max_y_it = std::min_element(max_y.begin(), max_y.end());
        terminal_x_center = (*inner_min_x_it + *inner_max_x_it) / 2;
        terminal_y_center = (*inner_min_y_it + *inner_max_y_it) / 2;
      } else {
        terminal_x_center = terminal_x[cur_terminal_id] + (terminal_size_x + terminal_spacing) / 2;
        terminal_y_center = terminal_y[cur_terminal_id] + (terminal_size_y + terminal_spacing) / 2;
      }
      for (int t = 0; t < num_tiers; ++t) {
        if (node_count[t] == 0) {
          continue;
        }
        max_x[t] = std::max(max_x[t], terminal_x_center);
        min_x[t] = std::min(min_x[t], terminal_x_center);
        max_y[t] = std::max(max_y[t], terminal_y_center);
        min_y[t] = std::min(min_y[t], terminal_y_center);
        hpwl_net += static_cast<long long>(max_x[t] - min_x[t] + max_y[t] - min_y[t]);
      }
    } else {
      int t_single = -1;
      for (int t = 0; t < num_tiers; ++t) {
        if (node_count[t] == num_nodes_in_net) {
          t_single = t;
          break;
        }
      }
      if (t_single >= 0) {
        hpwl_net += static_cast<long long>(max_x[t_single] - min_x[t_single] +
                                           max_y[t_single] - min_y[t_single]);
      }
    }
    return hpwl_net;
  };

  // calculate the hpwl of each net
  std::vector<long long> net_hpwl(num_nets, 0);
  long long hpwl_sum = 0;
  #pragma omp parallel for num_threads(num_threads) reduction(+ : hpwl_sum) schedule(static)
  for (int net_id = 0; net_id < num_nets; ++net_id) {
    net_hpwl[net_id] = compute_single_net_hpwl(net_id, tier, cut_net_mask);
    hpwl_sum += net_hpwl[net_id];
  }
  LOG(INFO, "hpwl_sum: %lld", hpwl_sum);

  auto compute_node_gain = [&](int node_id) -> long long {
    long long hpwl_delta = 0;
    long long terminal_gain = 0;
    int from_t = tier_of_node(node_id, tier);
    if (from_t < 0) {
      return 0;
    }
    int to_t = 1 - from_t;

    for (int net_id : node_to_nets[node_id]) {
      int cut_before = cut_net_mask[net_id];
      int cut_after = is_cut_after_move(net_id, from_t, to_t);
      terminal_gain += cut_before - cut_after;

      long long hpwl_before = net_hpwl[net_id];
      long long hpwl_after =
          compute_single_net_hpwl_after_move(net_id, node_id, to_t, cut_after);
      hpwl_delta += hpwl_before - hpwl_after;
    }

    return hpwl_delta + 1000 * terminal_gain;
  };

  const double die_area = static_cast<double>(die_size_x) * static_cast<double>(die_size_y);
  std::vector<double> tier_area(num_tiers, 0.0);

  // Bin state tracking for optimization
  int num_fm_bins_x = 0, num_fm_bins_y = 0;
  double bin_w = 0, bin_h = 0;
  auto get_bin_index = [&](double x, double y) -> int {
    int bx = static_cast<int>(
        std::floor(std::max(0.0, std::min(x, static_cast<double>(die_size_x - 1e-3))) / bin_w));
    int by = static_cast<int>(
        std::floor(std::max(0.0, std::min(y, static_cast<double>(die_size_y - 1e-3))) / bin_h));
    bx = std::max(0, std::min(bx, num_fm_bins_x - 1));
    by = std::max(0, std::min(by, num_fm_bins_y - 1));
    return by * num_fm_bins_x + bx;
  };

  // --- build FM bin grid based on case scale ---
  // Large cases benefit from smaller bin workloads (more bins), while small
  // cases avoid too many tiny bins.
  int target_nodes_per_bin = 128;

  int target_bins = std::max(1, (num_movable_nodes + target_nodes_per_bin - 1) /
                                    target_nodes_per_bin);
  // Keep enough bins to feed OpenMP-parallel stages.
  int parallel_bin_factor = 4;

  target_bins = std::max(target_bins, std::max(1, num_threads * parallel_bin_factor));
  // Avoid over-fragmenting when bins become too tiny.
  target_bins = std::min(target_bins, std::max(1, num_movable_nodes / 24));

  double die_aspect = static_cast<double>(die_size_x) / std::max(1.0, static_cast<double>(die_size_y));
  int grid_x = std::max(1, static_cast<int>(std::ceil(std::sqrt(target_bins * die_aspect))));
  int grid_y = std::max(1, static_cast<int>(std::ceil(static_cast<double>(target_bins) / grid_x)));
  num_fm_bins_x = grid_x;
  num_fm_bins_y = grid_y;
  bin_w = static_cast<double>(die_size_x) / static_cast<double>(num_fm_bins_x);
  bin_h = static_cast<double>(die_size_y) / static_cast<double>(num_fm_bins_y);
  LOG(INFO,
      "FM bin config: movable_nodes=%d, target_nodes_per_bin=%d, target_bins=%d, "
      "parallel_bin_factor=%d, grid=%dx%d",
      num_movable_nodes, target_nodes_per_bin, target_bins, parallel_bin_factor, num_fm_bins_x,
      num_fm_bins_y);

  bool pass_flag = true;
  int pass_count = 0;
  const int max_passes = 5;  // safety guard against pathological non-termination
  while (pass_flag) {
    auto pass_start = std::chrono::high_resolution_clock::now();
    pass_count++;
    long long best_pass_gain = 0;
    int total_moves = 0, total_bins_processed = 0, converged_bins = 0;

    std::fill(tier_area.begin(), tier_area.end(), 0.0);
    for (int n = 0; n < num_movable_nodes; ++n) {
      int t = tier_of_node(n, tier);
      if (t >= 0) {
        tier_area[t] += node_area_cache[n];
      }
    }

    std::vector<int> candidate_nodes;
    candidate_nodes.reserve(num_movable_nodes);
    std::vector<char> candidate_mark(num_movable_nodes, 0);
    int duplicate_candidate_hits = 0;
    for (int net_id = 0; net_id < num_nets; ++net_id) {
      if (cut_net_mask[net_id]) {
        for (int node_id : net_to_nodes[net_id]) {
          if (!candidate_mark[node_id]) {
            candidate_mark[node_id] = 1;
            candidate_nodes.push_back(node_id);
          } else {
            duplicate_candidate_hits++;
          }
        }
      }
    }
    int num_candidate_nodes = candidate_nodes.size();
    std::vector<std::vector<int>> bin_to_nodes(num_fm_bins_x * num_fm_bins_y);
    // node center from flattened 2D
    for (int node_id : candidate_nodes) {
      double x = static_cast<double>(pos_2d_x[node_id]);
      double y = static_cast<double>(pos_2d_y[node_id]);
      int b = get_bin_index(x, y);
      bin_to_nodes[b].push_back(node_id);
    }

    // process each bin independently; locked only within bin
    std::vector<int> index_in_bin(num_movable_nodes, -1);
    std::vector<int> touched_in_bin;
    std::vector<int> affected_stamp(num_movable_nodes, 0);
    int affected_epoch = 1;
    std::vector<int> affected_nodes;
    for (int by = 0; by < num_fm_bins_y; ++by) {
      for (int bx = 0; bx < num_fm_bins_x; ++bx) {
        int bidx = by * num_fm_bins_x + bx;
        auto& nodes = bin_to_nodes[bidx];
        if (nodes.empty())
          continue;

        total_bins_processed++;

        // local gain map and priority queue
        std::vector<long long> local_gain(nodes.size(), 0);
        std::priority_queue<Item, std::vector<Item>, ItemComparator> pq;

        std::vector<char> locked_local(nodes.size(), 0);
        touched_in_bin.clear();
        touched_in_bin.reserve(nodes.size());
        for (size_t i = 0; i < nodes.size(); ++i) {
          index_in_bin[nodes[i]] = static_cast<int>(i);
          touched_in_bin.push_back(nodes[i]);
        }
        #pragma omp parallel for num_threads(num_threads) schedule(static)
        for (int i = 0; i < static_cast<int>(nodes.size()); ++i) {
          local_gain[i] = compute_node_gain(nodes[i]);
        }
        for (size_t i = 0; i < nodes.size(); ++i) {
          pq.push({local_gain[i], nodes[i]});
        }

        std::vector<int> move_order, move_from;
        std::vector<long long> move_gain;
        move_order.reserve(nodes.size());
        move_from.reserve(nodes.size());
        move_gain.reserve(nodes.size());

        long long cumulative_gain = 0, best_prefix_gain = 0;
        int best_prefix_idx = -1;

        // select within bin
        int moves_in_bin = 0;
        for (size_t step = 0; step < nodes.size(); ++step) {
          int pick = -1;
          long long pick_gain = std::numeric_limits<long long>::min();
          while (!pq.empty()) {
            Item it = pq.top();
            pq.pop();
            int node_id = it.node_id;
            int li = (node_id >= 0 && node_id < num_movable_nodes) ? index_in_bin[node_id] : -1;
            if (li < 0)
              continue;  // not in this bin
            if (locked_local[li] || local_gain[li] != it.gain)
              continue;

            // die-level hard area check
            int from_t_chk = tier_of_node(node_id, tier);
            if (from_t_chk < 0) {
              continue;
            }
            int to_t_chk = 1 - from_t_chk;
            double area_u = node_area_cache[node_id];
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
          int from_t = tier_of_node(node_id, tier);
          if (from_t < 0) {
            continue;
          }
          int to_t = 1 - from_t;

          // apply move
          int li = index_in_bin[node_id];
          locked_local[li] = 1;
          tier[node_id] = to_t;

          // Update density map incrementally
          updateDensityMapLauncher(pos_2d_x, pos_2d_y, node_size_x, node_size_y, num_movable_nodes,
                                   num_bins_x, num_bins_y, xl, yl, xh, yh, num_threads,
                                   atomic_add_op, buf_map_tier.data(), from_t, to_t, node_id);

          double area_u = node_area_cache[node_id];
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

          // update global hpwl, cut mask, and terminal count incrementally
          long long hpwl_delta_apply = 0;
          affected_nodes.clear();
          if (affected_epoch == INT_MAX) {
            std::fill(affected_stamp.begin(), affected_stamp.end(), 0);
            affected_epoch = 1;
          }
          int cur_epoch = affected_epoch++;
          for (int net_id : node_to_nets[node_id]) {
            int cut_before = cut_net_mask[net_id];
            long long before = net_hpwl[net_id];

            net_tier_count[net_id][from_t]--;
            net_tier_count[net_id][to_t]++;
            int cut_after =
                (net_tier_count[net_id][0] > 0 && net_tier_count[net_id][1] > 0) ? 1 : 0;
            cut_net_mask[net_id] = cut_after;
            num_terminals_tmp += cut_after - cut_before;

            long long after = compute_single_net_hpwl(net_id, tier, cut_net_mask);
            net_hpwl[net_id] = after;
            hpwl_delta_apply += (after - before);
            for (int v : net_to_nodes[net_id]) {
              if (v == node_id)
                continue;
              int li_v = index_in_bin[v];
              if (li_v >= 0 && affected_stamp[v] != cur_epoch) {
                affected_stamp[v] = cur_epoch;
                affected_nodes.push_back(v);
              }
            }
          }
          hpwl_sum += hpwl_delta_apply;

          // refresh gains for affected nodes in this bin
          for (int v : affected_nodes) {
            int li_v = index_in_bin[v];
            if (li_v < 0)
              continue;
            long long old = local_gain[li_v];
            local_gain[li_v] = compute_node_gain(v);
            if (local_gain[li_v] != old)
              pq.push({local_gain[li_v], v});
          }

          moves_in_bin++;
        }

        if (moves_in_bin == 0 || best_prefix_gain <= 0) {
          converged_bins++;
        }

        // rollback tail moves beyond best prefix
        for (int i = static_cast<int>(move_order.size()) - 1; i > best_prefix_idx; --i) {
          int rollback_id = move_order[i];
          int from_t = move_from[i];
          int to_t = 1 - from_t;
          tier[rollback_id] = from_t;

          // Update density map incrementally for rollback
          updateDensityMapLauncher(pos_2d_x, pos_2d_y, node_size_x, node_size_y, num_movable_nodes,
                                   num_bins_x, num_bins_y, xl, yl, xh, yh, num_threads,
                                   atomic_add_op, buf_map_tier.data(), to_t, from_t, rollback_id);

          double area_u = node_area_cache[rollback_id];
          tier_area[from_t] += area_u;
          tier_area[to_t] -= area_u;

          long long hpwl_delta_rollback = 0;
          for (int net_id : node_to_nets[rollback_id]) {
            int cut_before = cut_net_mask[net_id];
            long long before = net_hpwl[net_id];

            net_tier_count[net_id][to_t]--;
            net_tier_count[net_id][from_t]++;
            int cut_after =
                (net_tier_count[net_id][0] > 0 && net_tier_count[net_id][1] > 0) ? 1 : 0;
            cut_net_mask[net_id] = cut_after;
            num_terminals_tmp += cut_after - cut_before;

            long long after = compute_single_net_hpwl(net_id, tier, cut_net_mask);
            net_hpwl[net_id] = after;
            hpwl_delta_rollback += (after - before);
          }
          hpwl_sum += hpwl_delta_rollback;
        }

        for (int touched_node : touched_in_bin) {
          index_in_bin[touched_node] = -1;
        }

        best_pass_gain += best_prefix_gain;
        total_moves += moves_in_bin;
      }
    }

    // Pass-level resync keeps incremental cut/HPWL state honest without paying the cost per bin.
    num_terminals_tmp = recompute_cut_state();
    hpwl_sum = 0;
    #pragma omp parallel for num_threads(num_threads) reduction(+ : hpwl_sum) schedule(static)
    for (int net_id = 0; net_id < num_nets; ++net_id) {
      net_hpwl[net_id] = compute_single_net_hpwl(net_id, tier, cut_net_mask);
      hpwl_sum += net_hpwl[net_id];
    }

    pass_flag = best_pass_gain > 0 ? true : false;

    // Early termination if all bins have converged
    int total_non_empty_bins = 0;
    for (int i = 0; i < num_fm_bins_x * num_fm_bins_y; ++i) {
      if (!bin_to_nodes[i].empty()) {
        total_non_empty_bins++;
      }
    }

    if (converged_bins >= total_non_empty_bins) {
      LOG(INFO, "All bins converged (%d/%d), terminating early", converged_bins,
          total_non_empty_bins);
      pass_flag = false;
    }

    auto pass_end = std::chrono::high_resolution_clock::now();
    auto pass_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(pass_end - pass_start).count();
    LOG(INFO,
        "Pass %d Summary: candidates=%d (dedup_hits=%d), processed %d bins (%d converged), "
        "%d total moves, best gain: %lld, elapsed: %lld ms",
        pass_count, num_candidate_nodes, duplicate_candidate_hits, total_bins_processed,
        converged_bins, total_moves, best_pass_gain, static_cast<long long>(pass_ms));
    LOG(INFO, "num_terminals_tmp: %d", num_terminals_tmp);
    LOG(INFO, "hpwl: %lld", hpwl_sum);
    LOG(INFO, "Pass %d termination check: best_pass_gain=%lld, pass_flag=%s", pass_count,
        best_pass_gain, pass_flag ? "true" : "false");
    if (pass_count >= max_passes) {
      LOG(INFO, "Reached max_passes=%d, force stopping FM loop for safety", max_passes);
      pass_flag = false;
    }
  }
}

at::Tensor bin_based_fm_forward(
    at::Tensor tier, at::Tensor flat_netpin, at::Tensor netpin_start, at::Tensor pin2node_map,
    at::Tensor net_weights, int num_movable_nodes, at::Tensor node_size_x, at::Tensor node_size_y,
    at::Tensor pin_offset_x, at::Tensor pin_offset_y, float die_size_x, float die_size_y,
    pybind11::list row_height, int terminal_size_x, int terminal_size_y, int terminal_spacing,
    at::Tensor pin_pos, const std::vector<std::string>& node_names,
    const std::vector<std::string>& net_names, at::Tensor pos_2d, at::Tensor pos_terminal_legalized,
    std::string case_name, int num_terminals, const std::vector<std::string>& terminal_names,
    float top_die_max_util, float bottom_die_max_util, int num_nodes, int num_bins_x,
    int num_bins_y, double xl, double yl, double xh, double yh) {
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
  CHECK_FLAT_CPU(pos_2d);
  CHECK_EVEN(pos_2d);

  int num_nets = netpin_start.numel() - 1;
  int num_pins = pin2node_map.numel();

  // TODO: get num_tiers from param.json
  int num_tiers = 2;
  TORCH_CHECK(num_movable_nodes >= 0, "bin_based_fm num_movable_nodes must be non-negative");
  TORCH_CHECK(tier.numel() >= num_movable_nodes,
              "bin_based_fm tier length is smaller than num_movable_nodes");
  TORCH_CHECK(pos_2d.numel() >= 2 * num_nodes,
              "bin_based_fm pos_2d length is smaller than 2*num_nodes");
  TORCH_CHECK(pos_terminal_legalized.numel() >= 2 * num_terminals,
              "bin_based_fm terminal position length is smaller than 2*num_terminals");
  TORCH_CHECK(pin_pos.numel() >= 2 * num_tiers * num_pins,
              "bin_based_fm pin_pos length is smaller than 2*num_tiers*num_pins");
  TORCH_CHECK(node_size_x.numel() >= num_tiers * num_movable_nodes &&
                  node_size_y.numel() >= num_tiers * num_movable_nodes,
              "bin_based_fm node size tensors are smaller than num_tiers*num_movable_nodes");
  TORCH_CHECK(pin_offset_x.numel() >= num_tiers * num_pins &&
                  pin_offset_y.numel() >= num_tiers * num_pins,
              "bin_based_fm pin offset tensors are smaller than num_tiers*num_pins");

  at::Tensor cut_net_mask = at::zeros(num_nets, tier.options());

  DREAMPLACE_DISPATCH_FLOATING_TYPES(pos_2d, "binBasedFMLauncher", [&] {
    binBasedFMLauncher<scalar_t>(
        DREAMPLACE_TENSOR_DATA_PTR(tier, int), DREAMPLACE_TENSOR_DATA_PTR(flat_netpin, int),
        DREAMPLACE_TENSOR_DATA_PTR(netpin_start, int),
        DREAMPLACE_TENSOR_DATA_PTR(pin2node_map, int), num_nets, num_tiers, num_movable_nodes,
        num_pins, DREAMPLACE_TENSOR_DATA_PTR(node_size_x, scalar_t),
        DREAMPLACE_TENSOR_DATA_PTR(node_size_y, scalar_t),
        DREAMPLACE_TENSOR_DATA_PTR(pin_offset_x, scalar_t),
        DREAMPLACE_TENSOR_DATA_PTR(pin_offset_y, scalar_t), die_size_x, die_size_y, row_height,
        terminal_size_x, terminal_size_y, terminal_spacing,
        DREAMPLACE_TENSOR_DATA_PTR(cut_net_mask, int),
        DREAMPLACE_TENSOR_DATA_PTR(pin_pos, scalar_t),
        DREAMPLACE_TENSOR_DATA_PTR(pin_pos, scalar_t) + pin_pos.numel() / 2, node_names, net_names,
        DREAMPLACE_TENSOR_DATA_PTR(pos_2d, scalar_t),
        DREAMPLACE_TENSOR_DATA_PTR(pos_2d, scalar_t) + pos_2d.numel() / 2,
        DREAMPLACE_TENSOR_DATA_PTR(pos_terminal_legalized, scalar_t),
        DREAMPLACE_TENSOR_DATA_PTR(pos_terminal_legalized, scalar_t) +
            pos_terminal_legalized.numel() / 2,
        case_name, num_terminals, terminal_names, top_die_max_util, bottom_die_max_util,
        at::get_num_threads(), num_nodes, num_bins_x, num_bins_y, static_cast<scalar_t>(xl),
        static_cast<scalar_t>(yl), static_cast<scalar_t>(xh), static_cast<scalar_t>(yh));
  });

  return tier;
}

PLACER_END_NAMESPACE

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("bin_based_fm", &PLACER_NAMESPACE::bin_based_fm_forward, "bin_based_fm_forward");
}
