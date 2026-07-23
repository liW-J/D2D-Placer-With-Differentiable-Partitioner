/*
 * @Date: 2025-06-17 13:30:24
 * @LastEditTime: 2025-10-15 13:03:26
 * @FilePath: /D2D-placer/placer/ops/hpwl_d2d/hpwl_d2d.cpp
 * @Description:
 */

#include <pybind11/pybind11.h>

#include "utility/src/torch.h"
#include "utility/src/utils.h"

// 3d-placer parser
#include "include/common.h"

PLACER_BEGIN_NAMESPACE

template <typename T>
int computeHPWLD2DLauncher(
    const T *pin_x, const T *pin_y, const int *flat_netpin,
    const int *netpin_start, const int *pin2node_map, const int *cut_net_mask,
    int num_nets, int num_pins, const int *tier, int num_tiers,
    const T *terminal_x, const T *terminal_y, int terminal_size_x,
    int terminal_size_y, int terminal_spacing, int num_terminals,
    const std::vector<std::string> &net_names,
    const std::vector<std::string> &terminal_names, int num_threads, T *hpwl);
/// @brief Compute half-perimeter wirelength
/// @param pos cell locations, array of x locations and then y locations
/// @param flat_netpin similar to the JA array in CSR format, which is flattened
/// from the net2pin map (array of array)
/// @param netpin_start similar to the IA array in CSR format, IA[i+1]-IA[i] is
/// the number of pins in each net, the length of IA is number of nets + 1
/// @param net_weights weight of nets
/// @param net_mask an array to record whether compute the where for a net or
/// not
at::Tensor hpwl_d2d_forward(at::Tensor pin_pos, at::Tensor flat_netpin,
                            at::Tensor netpin_start, at::Tensor pin2node_map,
                            at::Tensor net_weights, at::Tensor cut_net_mask,
                            at::Tensor tier, int num_tiers,
                            at::Tensor pos_terminal_legalized,
                            int terminal_size_x, int terminal_size_y,
                            int terminal_spacing, int num_terminals,
                            const std::vector<std::string> &net_names,
                            const std::vector<std::string> &terminal_names) {
  CHECK_FLAT_CPU(pin_pos);
  CHECK_EVEN(pin_pos);
  CHECK_CONTIGUOUS(pin_pos);
  CHECK_FLAT_CPU(flat_netpin);
  CHECK_CONTIGUOUS(flat_netpin);
  CHECK_FLAT_CPU(netpin_start);
  CHECK_CONTIGUOUS(netpin_start);
  CHECK_FLAT_CPU(pin2node_map);
  CHECK_CONTIGUOUS(pin2node_map);
  CHECK_FLAT_CPU(net_weights);
  CHECK_CONTIGUOUS(net_weights);
  CHECK_FLAT_CPU(cut_net_mask);
  CHECK_CONTIGUOUS(cut_net_mask);

  int num_nets = netpin_start.numel() - 1;
  int num_pins = pin2node_map.numel();

  at::Tensor hpwl = at::zeros(num_nets, pin_pos.options());
  DREAMPLACE_DISPATCH_FLOATING_TYPES(pin_pos, "computeHPWLD2DLauncher", [&] {
    computeHPWLD2DLauncher<scalar_t>(
        DREAMPLACE_TENSOR_DATA_PTR(pin_pos, scalar_t),
        DREAMPLACE_TENSOR_DATA_PTR(pin_pos, scalar_t) + pin_pos.numel() / 2,
        DREAMPLACE_TENSOR_DATA_PTR(flat_netpin, int),
        DREAMPLACE_TENSOR_DATA_PTR(netpin_start, int),
        DREAMPLACE_TENSOR_DATA_PTR(pin2node_map, int),
        DREAMPLACE_TENSOR_DATA_PTR(cut_net_mask, int), num_nets, num_pins,
        DREAMPLACE_TENSOR_DATA_PTR(tier, int), num_tiers,
        DREAMPLACE_TENSOR_DATA_PTR(pos_terminal_legalized, scalar_t),
        DREAMPLACE_TENSOR_DATA_PTR(pos_terminal_legalized, scalar_t) +
            pos_terminal_legalized.numel() / 2,
        terminal_size_x, terminal_size_y, terminal_spacing, num_terminals,
        net_names, terminal_names, at::get_num_threads(),
        DREAMPLACE_TENSOR_DATA_PTR(hpwl, scalar_t));
  });
  if (net_weights.numel()) {
    hpwl.mul_(net_weights);
  }
  return hpwl.sum();
}

template <typename T>
int computeHPWLD2DLauncher(
    const T *pin_x, const T *pin_y, const int *flat_netpin,
    const int *netpin_start, const int *pin2node_map, const int *cut_net_mask,
    int num_nets, int num_pins, const int *tier, int num_tiers,
    const T *terminal_x, const T *terminal_y, int terminal_size_x,
    int terminal_size_y, int terminal_spacing, int num_terminals,
    const std::vector<std::string> &net_names,
    const std::vector<std::string> &terminal_names, int num_threads, T *hpwl) {

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

  int hpwl_sum = 0;
  for (int net_id = 0; net_id < num_nets; ++net_id) {
    hpwl[net_id] = compute_single_net_hpwl(net_id, tier);
    hpwl_sum += hpwl[net_id];
  }

  return 0;
}

PLACER_END_NAMESPACE

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("hpwl_d2d", &PLACER_NAMESPACE::hpwl_d2d_forward, "hpwl_d2d_forward");
}
