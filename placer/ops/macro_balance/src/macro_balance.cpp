/*
 * @Date: 2025-03-19 11:49:04
 * @LastEditTime: 2025-07-28 00:44:33
 * @FilePath:
 * /D2D-placer/placer/ops/macro_balance/src/macro_balance.cpp
 * @Description:
 */
#include <algorithm>
#include <pybind11/pybind11.h>
// dreamplace
#include "utility/src/torch.h"
#include "utility/src/utils.h"

// 3d-placer parser
#include "include/common.h"
#include "placer/placer.h"

#include "utils_3d/src/partitioner.h"

PLACER_BEGIN_NAMESPACE

enum NodeType { MOVABLE, TERMINAL, TERMINAL_NI };

template <typename T>
void macroBalanceLauncher(
    int *tier, const int *flat_netpin, const int *netpin_start,
    const int *pin2node_map, int num_nets, int num_tiers, int num_movable_nodes,
    int num_pins, const T *node_size_x, const T *node_size_y,
    const T *pin_offset_x, const T *pin_offset_y, float die_size_x,
    float die_size_y, pybind11::list row_height, int terminal_size_x,
    int terminal_size_y, int terminal_spacing, int *cut_net_mask,
    const T *pin_x, const T *pin_y, const T *terminal_x, const T *terminal_y,
    int num_terminals, const std::vector<std::string> &node_names,
    const std::vector<std::string> &net_names,
    const std::vector<std::string> &terminal_names, const T *pos_2d_x,
    const T *pos_2d_y, std::string case_name,
    const std::vector<std::string> &node_orient, const bool *movable_macro_mask,
    int num_movable_macro) {

  int balance_num = ceil(num_movable_macro / num_tiers);
  vector<int> num_movable_macro_tier(num_tiers, 0);
  for (int i = 0; i < num_movable_nodes; i++) {
    if (movable_macro_mask[i]) {
      tier[i] = (num_movable_macro_tier[tier[i]] < balance_num) ? tier[i]
                                                                : 1 - tier[i];
      num_movable_macro_tier[tier[i]]++;
    }
  }
}

at::Tensor macro_balance_forward(
    at::Tensor tier, at::Tensor flat_netpin, at::Tensor netpin_start,
    at::Tensor pin2node_map, at::Tensor net_weights, int num_movable_nodes,
    at::Tensor node_size_x, at::Tensor node_size_y, at::Tensor pin_offset_x,
    at::Tensor pin_offset_y, float die_size_x, float die_size_y,
    pybind11::list row_height, int terminal_size_x, int terminal_size_y,
    int terminal_spacing, at::Tensor pin_pos, at::Tensor pos_terminal_legalized,
    int num_terminals, const std::vector<std::string> &node_names,
    const std::vector<std::string> &net_names,
    const std::vector<std::string> &terminal_names, at::Tensor pos_2d,
    std::string case_name, const std::vector<std::string> &node_orient,
    at::Tensor movable_macro_mask) {
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

  CHECK_FLAT_CPU(pos_terminal_legalized);
  CHECK_EVEN(pos_terminal_legalized);
  CHECK_CONTIGUOUS(pos_terminal_legalized);

  int num_nets = netpin_start.numel() - 1;
  int num_pins = pin2node_map.numel();
  int num_movable_macro = movable_macro_mask.sum().item<int>();
  LOG(INFO, "num_movable_macro: %d", num_movable_macro);

  // TODO: get num_tiers from param.json
  int num_tiers = 2;
  at::Tensor cut_net_mask = at::zeros(num_nets, tier.options());

  DREAMPLACE_DISPATCH_FLOATING_TYPES(node_size_x, "macroBalanceLauncher", [&] {
    macroBalanceLauncher<scalar_t>(
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
        DREAMPLACE_TENSOR_DATA_PTR(pos_terminal_legalized, scalar_t),
        DREAMPLACE_TENSOR_DATA_PTR(pos_terminal_legalized, scalar_t) +
            pos_terminal_legalized.numel() / 2,
        num_terminals, node_names, net_names, terminal_names,
        DREAMPLACE_TENSOR_DATA_PTR(pos_2d, scalar_t),
        DREAMPLACE_TENSOR_DATA_PTR(pos_2d, scalar_t) + pos_2d.numel() / 2,
        case_name, node_orient,
        DREAMPLACE_TENSOR_DATA_PTR(movable_macro_mask, bool),
        num_movable_macro);
  });

  return tier;
}

PLACER_END_NAMESPACE

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("macro_balance", &PLACER_NAMESPACE::macro_balance_forward,
        "macro_balance_forward");
}
