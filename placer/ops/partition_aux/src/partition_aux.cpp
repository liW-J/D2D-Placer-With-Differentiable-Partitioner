/*
 * @Author: JeanneWillis hi@jeannewillis.cn
 * @Date: 2025-03-19 11:49:04
 * @LastEditors: JeanneWillis hi@jeannewillis.cn
 * @LastEditTime: 2025-07-23 15:39:40
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

enum NodeType { MOVABLE, TERMINAL, TERMINAL_NI };

template <typename T>
void terminal_insert(const int *tier, const T *pin_x, const T *pin_y,
                     const int *flat_netpin, const int *netpin_start,
                     const int *pin2node_map, int num_movable_nodes,
                     int num_nets, int num_pins, int num_tiers,
                     const int *cut_net_mask, int terminal_size_x,
                     int terminal_size_y, int terminal_spacing,
                     bool terminal_instert_flag, vector<AUX> &auxListRef,
                     bool terminal_legalize_flag, const T *terminal_x,
                     const T *terminal_y, int num_terminals,
                     const std::vector<std::string> &node_names,
                     const std::vector<std::string> &net_names,
                     const std::vector<std::string> &terminal_names) {
  LOG(WARN, "terminal_insert");

  int terminal_count = 0;
  for (int net_id = 0; net_id < num_nets; net_id++) {
    vector<T> max_x(num_tiers, -std::numeric_limits<T>::max());
    vector<T> min_x(num_tiers, std::numeric_limits<T>::max());
    vector<T> max_y(num_tiers, -std::numeric_limits<T>::max());
    vector<T> min_y(num_tiers, std::numeric_limits<T>::max());
    if (cut_net_mask[net_id]) {
      for (int tier_id = 0; tier_id < num_tiers; ++tier_id) {
        for (int pin_id = netpin_start[net_id];
             pin_id < netpin_start[net_id + 1]; pin_id++) {
          int index_pin = num_pins * tier_id + flat_netpin[pin_id];
          int node_id = pin2node_map[flat_netpin[pin_id]];
          if (tier[node_id] == tier_id) {
            max_x[tier_id] = std::max(max_x[tier_id], pin_x[index_pin]);
            min_x[tier_id] = std::min(min_x[tier_id], pin_x[index_pin]);
            max_y[tier_id] = std::max(max_y[tier_id], pin_y[index_pin]);
            min_y[tier_id] = std::min(min_y[tier_id], pin_y[index_pin]);
          }
        }
      }
      // get inster bonding
      auto inner_min_x_it = max_element(min_x.begin(), min_x.end());
      auto inner_max_x_it = min_element(max_x.begin(), max_x.end());
      auto inner_min_y_it = max_element(min_y.begin(), min_y.end());
      auto inner_max_y_it = min_element(max_y.begin(), max_y.end());
      T inner_min_x = *inner_min_x_it;
      T inner_max_x = *inner_max_x_it;
      T inner_min_y = *inner_min_y_it;
      T inner_max_y = *inner_max_y_it;
      // LOG(WARN,
      //     "inner_min_x: %f, inner_max_x: %f, inner_min_y: %f, inner_max_y:
      //     %f", inner_min_x, inner_max_x, inner_min_y, inner_max_y);
      T center_x = (inner_min_x + inner_max_x) / 2;
      T center_y = (inner_min_y + inner_max_y) / 2;

      terminal_count++;
      for (int tier_id = 0; tier_id < num_tiers; ++tier_id) {
        // LOG(INFO, "terminal_count: %d", terminal_count);

        if (terminal_legalize_flag) {
          int x, y;
          // LOG(INFO, "net_names[net_id]: %s", net_names[net_id]);
          for (int terminal_id = 0; terminal_id < num_terminals;
               ++terminal_id) {
            if (net_names[net_id] == terminal_names[terminal_id]) {
              // dreamplace pos is left-bottom corner
              x = terminal_x[terminal_id] + terminal_spacing / 2;
              y = terminal_y[terminal_id] + terminal_spacing / 2;
              // LOG(INFO, "net_id: %d, terminal_count: %d, i: %d", net_id,
              //     terminal_count, i);
              break;
            }
          }
          auxListRef[tier_id].add_node(net_names[net_id], terminal_size_x,
                                       terminal_size_y, x, y, TERMINAL_NI);
        } else {
          auxListRef[tier_id].add_node(net_names[net_id], terminal_size_x,
                                       terminal_size_y, center_x, center_y,
                                       TERMINAL_NI);
          // LOG(DEBUG, "Intersection Center: (%f, %f)", center_x, center_y);
        }
        auxListRef[tier_id].add_pin(net_names[net_id], net_names[net_id], 'O',
                                    0, 0);
      }
    }
  }
}

template <typename T>
void partitionAuxLauncher(
    const int *tier, const int *flat_netpin, const int *netpin_start,
    const int *pin2node_map, int num_nets, int num_tiers, int num_movable_nodes,
    int num_pins, const T *node_size_x, const T *node_size_y,
    const T *pin_offset_x, const T *pin_offset_y, float die_size_x,
    float die_size_y, pybind11::list row_height, int terminal_size_x,
    int terminal_size_y, int terminal_spacing, int *cut_net_mask,
    const T *pin_x, const T *pin_y, bool terminal_instert_flag,
    bool terminal_legalize_flag, const T *terminal_x, const T *terminal_y,
    int num_terminals, const std::vector<std::string> &node_names,
    const std::vector<std::string> &net_names,
    const std::vector<std::string> &terminal_names, const T *pos_2d_x,
    const T *pos_2d_y, std::string case_name,
    const std::vector<std::string> &node_orient) {
  string aux_dir = "./run_tmp/" + case_name + "/partition/";
  char IO_type;
  vector<AUX> aux_list(num_tiers);

  // init aux files
  for (int tier_id = 0; tier_id < num_tiers; ++tier_id) {
    string tier_name = "tier" + to_string(tier_id);
    aux_list[tier_id] = AUX(aux_dir, tier_name);
  }

  for (int net_id = 0; net_id < num_nets; ++net_id) {
    int num_nodes_in_net = 0;
    vector<int> node_count(num_tiers, 0);
    for (int tier_id = 0; tier_id < num_tiers; ++tier_id) {
      for (int pin_id = netpin_start[net_id]; pin_id < netpin_start[net_id + 1];
           ++pin_id) {
        int node_id = pin2node_map[flat_netpin[pin_id]];
        if (tier[node_id] == tier_id) {
          int index_node = num_movable_nodes * tier_id + node_id;
          node_count[tier_id]++;
          num_nodes_in_net++;
          if (!aux_list[tier_id].check_net_exist(net_names[net_id])) {
            aux_list[tier_id].add_net(net_names[net_id]);
          }
          if (!aux_list[tier_id].check_node_exist(node_names[node_id])) {
            // LOG(DEBUG, "node %d, tier %d, node_size_x %f, node_size_y %f",
            // node_id, tier_id, node_size_x[index], node_size_y[index]);
            int node_pos_x = (pos_2d_x == nullptr) ? 0 : pos_2d_x[node_id];
            int node_pos_y = (pos_2d_y == nullptr) ? 0 : pos_2d_y[node_id];
            aux_list[tier_id].add_node(
                node_names[node_id], node_size_x[index_node],
                node_size_y[index_node], node_pos_x, node_pos_y, MOVABLE);
          }
          if (!aux_list[tier_id].check_pin_exist(net_names[net_id],
                                                 node_names[node_id])) {
            int index_pin = num_pins * tier_id + flat_netpin[pin_id];
            (pin_id == netpin_start[net_id]) ? IO_type = 'I' : IO_type = 'O';
            // unable center func type, use explicit type
            // because pin_offset_x and pin_offset_y are relative to the node
            // center in dreamplace so we need to subtract the node_size / 2
            // when add pin to aux file
            aux_list[tier_id].add_pin(
                net_names[net_id], node_names[node_id], IO_type,
                static_cast<float>(pin_offset_x[index_pin] -
                                   ceil(node_size_x[index_node] / 2)),
                static_cast<float>(pin_offset_y[index_pin] -
                                   ceil(node_size_y[index_node] / 2)));
          }
        }
      }
    }
    // if all nodes are in the same tier, not cut
    if (std::find(node_count.begin(), node_count.end(), num_nodes_in_net) ==
        node_count.end()) {
      cut_net_mask[net_id] = 1;
      // LOG(DEBUG, "cut.");
    }
  }

  if (terminal_instert_flag) {
    terminal_insert(tier, pin_x, pin_y, flat_netpin, netpin_start, pin2node_map,
                    num_movable_nodes, num_nets, num_pins, num_tiers,
                    cut_net_mask, terminal_size_x, terminal_size_y,
                    terminal_spacing, terminal_instert_flag, aux_list,
                    terminal_legalize_flag, terminal_x, terminal_y,
                    num_terminals, node_names, net_names, terminal_names);
  }

  // write aux files
  for (int tier_id = 0; tier_id < num_tiers; ++tier_id) {
    // pybind11 type convert
    int tier_row_height = pybind11::cast<float>(row_height[tier_id]);
    aux_list[tier_id].set_default_rows(die_size_x, tier_row_height,
                                       die_size_y / tier_row_height);
    // LOG(DEBUG, "test-row-height: %d", tier_row_height);
    // LOG(DEBUG, "die_size_x: %f", die_size_x);
    // LOG(DEBUG, "die_size_y: %f", die_size_y);
    // sort node by name
    aux_list[tier_id].sort_node();

    aux_list[tier_id].write_files(node_orient);
  }

  Partitioner::countRelatedNodesInCutNets(tier, flat_netpin, netpin_start,
                                          pin2node_map, num_movable_nodes,
                                          num_nets, num_tiers, cut_net_mask);
}

at::Tensor partition_aux_forward(
    at::Tensor tier, at::Tensor flat_netpin, at::Tensor netpin_start,
    at::Tensor pin2node_map, at::Tensor net_weights, int num_movable_nodes,
    at::Tensor node_size_x, at::Tensor node_size_y, at::Tensor pin_offset_x,
    at::Tensor pin_offset_y, float die_size_x, float die_size_y,
    pybind11::list row_height, int terminal_size_x, int terminal_size_y,
    int terminal_spacing, at::Tensor pin_pos, bool terminal_instert_flag,
    bool terminal_legalize_flag, at::Tensor pos_terminal_legalized,
    int num_terminals, const std::vector<std::string> &node_names,
    const std::vector<std::string> &net_names,
    const std::vector<std::string> &terminal_names, at::Tensor pos_2d,
    std::string case_name, const std::vector<std::string> &node_orient) {
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

  // TODO: get num_tiers from param.json
  int num_tiers = 2;
  at::Tensor cut_net_mask = at::zeros(num_nets, tier.options());

  DREAMPLACE_DISPATCH_FLOATING_TYPES(node_size_x, "partitionAuxLauncher", [&] {
    partitionAuxLauncher<scalar_t>(
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
        terminal_instert_flag, terminal_legalize_flag,
        DREAMPLACE_TENSOR_DATA_PTR(pos_terminal_legalized, scalar_t),
        DREAMPLACE_TENSOR_DATA_PTR(pos_terminal_legalized, scalar_t) +
            pos_terminal_legalized.numel() / 2,
        num_terminals, node_names, net_names, terminal_names,
        DREAMPLACE_TENSOR_DATA_PTR(pos_2d, scalar_t),
        DREAMPLACE_TENSOR_DATA_PTR(pos_2d, scalar_t) + pos_2d.numel() / 2,
        case_name, node_orient);
  });

  return cut_net_mask;
}

PLACER_END_NAMESPACE

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("partition_aux", &PLACER_NAMESPACE::partition_aux_forward,
        "partition_aux_forward");
}
