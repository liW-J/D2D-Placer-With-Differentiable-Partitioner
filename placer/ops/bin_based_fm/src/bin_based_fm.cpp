/*
 * @Author: JeanneWillis hi@jeannewillis.cn
 * @Date: 2025-09-21 17:01:58
 * @LastEditors: JeanneWillis hi@jeannewillis.cn
 * @LastEditTime: 2025-09-23 12:50:31
 * @FilePath: /D2D-placer/placer/ops/bin_based_fm/src/bin_based_fm.cpp
 * @Description: fm refinement
 */

#include <pybind11/pybind11.h>
// dreamplace
#include "utility/src/torch.h"
#include "utility/src/utils.h"

// 3d-placer parser
#include "include/common.h"
#include "placer/placer.h"

#include "utils_3d/src/partitioner.h"

PLACER_BEGIN_NAMESPACE

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
    const std::vector<std::string> &terminal_names, int num_threads) {

  Partitioner::getCutNetMask(cut_net_mask, num_nets, num_tiers, tier,
                             flat_netpin, netpin_start, pin2node_map,
                             num_movable_nodes);

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

  // initial total HPWL
  int hpwl = 0;
  for (int net_id = 0; net_id < num_nets; ++net_id) {
    hpwl += compute_single_net_hpwl(net_id, tier);
  }
  LOG(INFO, "HPWL: %d", hpwl);

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

  // build spatial bin (simple grid, bucket the candidates by location)
  int bins_x =
      std::max(1, static_cast<int>(std::sqrt(
                      static_cast<double>(num_movable_nodes) / 128.0)));
  int bins_y = bins_x;
  T bin_w = die_size_x / bins_x;
  T bin_h = die_size_y / bins_y;
  std::vector<std::vector<int>> bins(bins_x * bins_y);
  auto node_pos_x = pos_2d_x;
  auto node_pos_y = pos_2d_y;
  for (int node_id = 0; node_id < num_movable_nodes; ++node_id) {
    int bx = std::min(
        bins_x - 1, std::max(0, static_cast<int>(node_pos_x[node_id] / bin_w)));
    int by = std::min(
        bins_y - 1, std::max(0, static_cast<int>(node_pos_y[node_id] / bin_h)));
    bins[by * bins_x + bx].push_back(node_id);
  }

  std::vector<int> locked_node_mask(num_movable_nodes, 0);
  int refinement_flag = 1;
  int remaining_unlocked = num_movable_nodes;

  while (refinement_flag) {
    if (remaining_unlocked <= 0) {
      break;
    }
    int gain_max = -std::numeric_limits<int>::max();
    int node_id_max = -1;
    int hpwl_after_best = hpwl;
    int num_terminals_after_best = num_terminals;

    // scan the candidates bin by bin, evaluate the incremental gain
    for (size_t b = 0; b < bins.size(); ++b) {
      for (int node_id : bins[b]) {
        if (locked_node_mask[node_id])
          continue;

        // assume moving the node: only recalculate the HPWL of the adjacent
        // networks and estimate the cut change
        int delta_hpwl_sum = 0;
        int delta_terminals_sum = 0;

        // construct a temporary tier view: only used once, pass to lambda
        // to avoid copying large arrays, here we use a small stack array to
        // cover the tier of the current node implemented as: first remember the
        // old tier, modify, calculate again and then restore
        int old_tier = tier[node_id];
        int new_tier = 1 - old_tier;
        tier[node_id] = new_tier;

        for (int net_id : node_to_nets[node_id]) {
          // calculate the HPWL and cut state before and after
          // before: restore the old tier and calculate the HPWL
          tier[node_id] = old_tier;
          int hpwl_before = compute_single_net_hpwl(net_id, tier);
          bool was_cut = cut_net_mask[net_id] != 0;

          // after: change to the new tier and calculate the HPWL
          tier[node_id] = new_tier;
          int hpwl_after = compute_single_net_hpwl(net_id, tier);

          // cut determination: recalculate once (cost is the same as HPWL)
          // simplified: use the layer participation of hpwl_after and
          // hpwl_before as the cut determination, here we directly reuse the
          // logic: use the cut determination inside compute_single_net_hpwl
          // consistent with tier_ptr, recalculate the flag to avoid duplicate
          // code, here we approximate the cut change by the difference of HPWL
          // and the layer change of the node: if hpwl_after < hpwl_before and
          // the layer interaction decreases, the cut may decrease. more secure:
          // explicitly recalculate the cut: determine the cut: count the number
          // of nodes in each layer
          int num_nodes_in_net = 0;
          std::vector<int> node_count(num_tiers, 0);
          for (int pin_id = netpin_start[net_id];
               pin_id < netpin_start[net_id + 1]; ++pin_id) {
            int nid = pin2node_map[flat_netpin[pin_id]];
            node_count[tier[nid]]++;
            num_nodes_in_net++;
          }
          bool now_cut = true;
          for (int t = 0; t < num_tiers; ++t) {
            if (node_count[t] == num_nodes_in_net) {
              now_cut = false;
              break;
            }
          }

          // restore the node tier to the new tier to accumulate the delta
          tier[node_id] = new_tier;

          delta_hpwl_sum += (hpwl_before - hpwl_after);
          if (was_cut != now_cut) {
            delta_terminals_sum += now_cut ? 1 : -1;
          }
        }

        // restore the real tier
        tier[node_id] = old_tier;

        int hpwl_gain = delta_hpwl_sum;
        int num_terminals_tmp = num_terminals + delta_terminals_sum;
        int terminal_gain = (num_terminals - num_terminals_tmp);
        int gain =
            (hpwl_gain + 2000 * terminal_gain) * (num_nets - num_terminals_tmp);

        if (gain > gain_max) {
          gain_max = gain;
          node_id_max = node_id;
          hpwl_after_best = hpwl - hpwl_gain; // hpwl - delta
          num_terminals_after_best = num_terminals_tmp;
          LOG(INFO, "GAIN: %d; dHPWL: %d; dTERM: %d", gain, hpwl_gain,
              terminal_gain);
        }
      }
    }

    refinement_flag = gain_max > 0 ? 1 : 0;

    if (refinement_flag && node_id_max >= 0 && !locked_node_mask[node_id_max]) {
      // actually apply the move: update the cut of the networks related to the
      // node and the global HPWL
      int old_tier = tier[node_id_max];
      int new_tier = 1 - old_tier;
      tier[node_id_max] = new_tier;

      for (int net_id : node_to_nets[node_id_max]) {
        // update the cut flag
        int num_nodes_in_net = 0;
        std::vector<int> node_count(num_tiers, 0);
        for (int pin_id = netpin_start[net_id];
             pin_id < netpin_start[net_id + 1]; ++pin_id) {
          int nid = pin2node_map[flat_netpin[pin_id]];
          node_count[tier[nid]]++;
          num_nodes_in_net++;
        }
        bool now_cut = true;
        for (int t = 0; t < num_tiers; ++t) {
          if (node_count[t] == num_nodes_in_net) {
            now_cut = false;
            break;
          }
        }
        int before_flag = cut_net_mask[net_id];
        cut_net_mask[net_id] = now_cut ? 1 : 0;
        if (before_flag != cut_net_mask[net_id]) {
          num_terminals += (cut_net_mask[net_id] ? 1 : -1);
        }
      }

      // update the global HPWL: recalculate the affected nets and replace the
      // contribution to avoid storing hpwl for each net, here we approximate:
      // use hpwl_after_best
      hpwl = hpwl_after_best;
      locked_node_mask[node_id_max] = 1;
      remaining_unlocked--;
    }
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
    const std::vector<std::string> &terminal_names) {
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
        case_name, num_terminals, terminal_names, at::get_num_threads());
  });

  return tier;
}

PLACER_END_NAMESPACE

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("bin_based_fm", &PLACER_NAMESPACE::bin_based_fm_forward,
        "bin_based_fm_forward");
}
