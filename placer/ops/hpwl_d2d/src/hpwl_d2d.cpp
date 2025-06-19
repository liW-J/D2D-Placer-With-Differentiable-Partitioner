/*
 * @Author: JeanneWillis hi@jeannewillis.cn
 * @Date: 2025-06-17 13:30:24
 * @LastEditors: JeanneWillis hi@jeannewillis.cn
 * @LastEditTime: 2025-06-19 13:13:30
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
    const std::vector<std::string> &terminal_name, int num_threads, T *hpwl);
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
                            const std::vector<std::string> &terminal_name) {
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
        net_names, terminal_name, at::get_num_threads(),
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
    const std::vector<std::string> &terminal_name, int num_threads, T *hpwl) {
// #pragma omp parallel for num_threads(num_threads)
  for (int net_id = 0; net_id < num_nets; ++net_id) {

    if (cut_net_mask[net_id]) {

      int cur_terminal_id = 0;
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
      T terminal_x_center = terminal_x[cur_terminal_id] +
                            (terminal_size_x + terminal_spacing) / 2;
      T terminal_y_center = terminal_y[cur_terminal_id] +
                            (terminal_size_y + terminal_spacing) / 2;

      for (int tier_id = 0; tier_id < num_tiers; ++tier_id) {
        for (int pin_id = netpin_start[net_id];
             pin_id < netpin_start[net_id + 1]; pin_id++) {
          int node_id = pin2node_map[flat_netpin[pin_id]];
          int index_pin = num_pins * tier_id + flat_netpin[pin_id];
          if (tier[node_id] == tier_id) {
            // LOG(WARN, "pin_x: %f, pin_y: %f", pin_x[index_pin], pin_y[index_pin]);
            max_x[tier_id] = std::max(max_x[tier_id], pin_x[index_pin]);
            min_x[tier_id] = std::min(min_x[tier_id], pin_x[index_pin]);
            max_y[tier_id] = std::max(max_y[tier_id], pin_y[index_pin]);
            min_y[tier_id] = std::min(min_y[tier_id], pin_y[index_pin]);
          }
        }
      }

      for (int tier_id = 0; tier_id < num_tiers; ++tier_id) {
        // LOG(WARN, "tier_id: %d, terminal_x_center: %f, terminal_y_center: %f", tier_id, terminal_x_center, terminal_y_center);
        max_x[tier_id] = std::max(max_x[tier_id], terminal_x_center);
        min_x[tier_id] = std::min(min_x[tier_id], terminal_x_center);
        max_y[tier_id] = std::max(max_y[tier_id], terminal_y_center);
        min_y[tier_id] = std::min(min_y[tier_id], terminal_y_center);
        hpwl[net_id] +=
            max_x[tier_id] - min_x[tier_id] + max_y[tier_id] - min_y[tier_id];
        // LOG(INFO, "net_id: %d, tier_id: %d, hpwl: %f", net_id, tier_id, hpwl[net_id]);
      }
    } else {
      T max_x = -std::numeric_limits<T>::max();
      T min_x = std::numeric_limits<T>::max();
      T max_y = -std::numeric_limits<T>::max();
      T min_y = std::numeric_limits<T>::max();
      int tier_id = tier[pin2node_map[flat_netpin[netpin_start[net_id]]]];

      for (int pin_id = netpin_start[net_id]; pin_id < netpin_start[net_id + 1];
           pin_id++) {
        int index_pin = num_pins * tier_id + flat_netpin[pin_id];
        // LOG(WARN, "pin_x: %f, pin_y: %f", pin_x[index_pin], pin_y[index_pin]);
        min_x = std::min(min_x, pin_x[index_pin]);
        max_x = std::max(max_x, pin_x[index_pin]);
        min_y = std::min(min_y, pin_y[index_pin]);
        max_y = std::max(max_y, pin_y[index_pin]);
      }
      hpwl[net_id] += max_x - min_x + max_y - min_y;
      // LOG(INFO, "net_id: %d, tier_id: %d, hpwl: %f", net_id, tier_id, hpwl[net_id]);
    }
  }

  return 0;
}

PLACER_END_NAMESPACE

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("hpwl_d2d", &PLACER_NAMESPACE::hpwl_d2d_forward, "hpwl_d2d_forward");
}
