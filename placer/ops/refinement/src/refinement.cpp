/*
 * @Author: JeanneWillis hi@jeannewillis.cn
 * @Date: 2025-03-19 11:49:04
 * @LastEditors: JeanneWillis hi@jeannewillis.cn
 * @LastEditTime: 2025-09-22 12:58:09
 * @FilePath: /D2D-placer/src/ops/partition/src/partition.cpp
 * @Description: partition
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
void refinementLauncher(
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

  int hpwl = Partitioner::computeHPWLD2D(
      pin_x, pin_y, flat_netpin, netpin_start, pin2node_map, cut_net_mask,
      num_nets, num_pins, num_movable_nodes, tier, num_tiers, terminal_x,
      terminal_y, terminal_size_x, terminal_size_y, terminal_spacing,
      num_terminals,
      net_names, terminal_names, num_threads);

  LOG(INFO, "HPWL: %d", hpwl);

  std::vector<int> locked_node_mask(num_movable_nodes, 0);
  int refinement_flag = 1;

  while (refinement_flag) {
    // int hpwl_gain_max = -std::numeric_limits<int>::max();
    int gain_max = -std::numeric_limits<int>::max();
    int node_id_max = -1;
    int hpwl_gain, terminal_gain, gain = 0;
    int hpwl_base = hpwl;
    int num_terminals_base = num_terminals;
    int init_num_terminals = num_terminals;

    for (int node_id = 0; node_id < num_movable_nodes; ++node_id) {
      if (!locked_node_mask[node_id]) {

        std::vector<int> tier_tmp(tier, tier + num_movable_nodes);
        std::vector<int> cut_net_mask_tmp(num_nets, 0);
        tier_tmp[node_id] = 1 - tier_tmp[node_id];

        int num_terminals_tmp = Partitioner::getCutNetMask(
            cut_net_mask_tmp.data(), num_nets, num_tiers, tier_tmp.data(),
            flat_netpin, netpin_start, pin2node_map, num_movable_nodes);

        int hpwl_tmp = Partitioner::computeHPWLD2D(
            pin_x, pin_y, flat_netpin, netpin_start, pin2node_map,
            cut_net_mask_tmp.data(), num_nets, num_pins, num_movable_nodes,
            tier_tmp.data(), num_tiers, terminal_x, terminal_y, terminal_size_x,
            terminal_size_y, terminal_spacing, init_num_terminals, net_names,
            terminal_names,
            num_threads);
        hpwl_gain = hpwl - hpwl_tmp;
        terminal_gain = num_terminals - num_terminals_tmp;
        gain =
            (hpwl_gain + 1000 * terminal_gain) * (num_nets - num_terminals_tmp);

        if (gain > gain_max) {
          gain_max = gain;
          node_id_max = node_id;
          hpwl_base = hpwl_tmp;
          num_terminals_base = num_terminals_tmp;
          LOG(INFO, "GAIN: %d; HPWL: %d, HPWL_TMP: %d; TERMINAL_TMP: %d", gain,
              hpwl, hpwl_tmp, num_terminals_tmp);
        }
      }
    }

    refinement_flag = gain_max > 0 ? 1 : 0;

    if (!locked_node_mask[node_id_max] && refinement_flag) {
      locked_node_mask[node_id_max] = 1;
      tier[node_id_max] = 1 - tier[node_id_max];
      hpwl = hpwl_base;
      num_terminals = num_terminals_base;
    }
  }
}

at::Tensor refinement_forward(
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

  // CHECK_FLAT_CPU(tier);
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

  DREAMPLACE_DISPATCH_FLOATING_TYPES(node_size_x, "refinementLauncher", [&] {
    refinementLauncher<scalar_t>(
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
  m.def("refinement", &PLACER_NAMESPACE::refinement_forward,
        "refinement_forward");
}
