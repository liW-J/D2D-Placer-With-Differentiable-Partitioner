/*
 * @Date: 2025-03-19 11:49:04
 * @LastEditTime: 2025-09-21 17:17:41
 * @FilePath: /D2D-placer/src/ops/partition/src/partition.cpp
 * @Description: partition
 */
#include <pybind11/pybind11.h>
#include <cctype>
// dreamplace
#include "utility/src/torch.h"
#include "utility/src/utils.h"

// 3d-placer parser
#include "include/common.h"
#include "placer/placer.h"

PLACER_BEGIN_NAMESPACE

enum NodeType { MOVABLE, TERMINAL, TERMINAL_NI };

static bool isBookshelfStringChar(char c) {
  const unsigned char uc = static_cast<unsigned char>(c);
  return std::isalnum(uc) || c == '_' || c == ',' || c == '.' || c == '$' ||
         c == '-' || c == '[' || c == ']' || c == '/';
}

static bool isBookshelfReservedToken(const std::string &name) {
  std::string lower;
  lower.reserve(name.size());
  for (char c : name) {
    lower.push_back(static_cast<char>(std::tolower(
        static_cast<unsigned char>(c))));
  }
  static const std::vector<std::string> reserved = {
      "terminal", "ucla", "netdegree", "scl", "nodes", "nets", "pl",
      "wts", "shapes", "route", "aux", "fixed", "fixed_ni", "placed",
      "unplaced", "o", "i", "b", "n", "s", "w", "e", "fn", "fs",
      "fw", "fe"};
  return std::find(reserved.begin(), reserved.end(), lower) != reserved.end();
}

static std::string bookshelfName(const std::string &raw,
                                 const std::string &prefix, int index,
                                 bool force_prefix = false) {
  bool changed = force_prefix || raw.empty() ||
                 !std::isalpha(static_cast<unsigned char>(raw.front())) ||
                 isBookshelfReservedToken(raw);
  std::string body;
  body.reserve(raw.empty() ? 4 : raw.size());
  for (char c : raw) {
    if (isBookshelfStringChar(c)) {
      body.push_back(c);
    } else {
      body.push_back('_');
      changed = true;
    }
  }
  if (body.empty()) {
    body = "anon";
  }
  if (!changed) {
    return raw;
  }
  return prefix + std::to_string(index) + "_" + body;
}

static std::string bookshelfNodeName(const std::string &raw, int node_id) {
  return bookshelfName(raw, "X", node_id);
}

static std::string bookshelfNetName(const std::string &raw, int net_id) {
  return bookshelfName(raw, "NET", net_id);
}


template <typename T>
void terminal_insert(
    const int *tier, const T *pin_x, const T *pin_y, const int *flat_netpin,
    const int *netpin_start, const int *pin2node_map, int num_movable_nodes,
    int num_nets, int num_pins, int num_tiers, const int *cut_net_mask,
    AUX &terminalAuxRef, int terminal_size_x, int terminal_size_y,
    int terminal_spacing, const std::vector<std::string> &node_names,
    const std::vector<std::string> &net_names, bool terminal_legalize_flag,
    const T *terminal_x, const T *terminal_y, int num_terminals,
    const std::vector<std::string> &terminal_names) {
  LOG(WARN, "terminal_insert");

  int terminal_count = 0;
  for (int net_id = 0; net_id < num_nets; net_id++) {
    vector<T> max_x(num_tiers, -std::numeric_limits<T>::max());
    vector<T> min_x(num_tiers, std::numeric_limits<T>::max());
    vector<T> max_y(num_tiers, -std::numeric_limits<T>::max());
    vector<T> min_y(num_tiers, std::numeric_limits<T>::max());
    if (cut_net_mask[net_id]) {
      const std::string net_name = bookshelfNetName(net_names[net_id], net_id);
      for (int tier_id = 0; tier_id < num_tiers; ++tier_id) {
        for (int pin_id = netpin_start[net_id];
             pin_id < netpin_start[net_id + 1]; pin_id++) {
          int index_pin = num_pins * tier_id + flat_netpin[pin_id];
          int node_id = pin2node_map[flat_netpin[pin_id]];
          if (node_id < 0 || node_id >= num_movable_nodes) {
            continue;
          }
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
      // Bonding center from cross-tier pin bbox; DreamPlace pos is left-bottom.
      T center_x = (inner_min_x + inner_max_x) / 2;
      T center_y = (inner_min_y + inner_max_y) / 2;
      const T term_w =
          static_cast<T>(terminal_size_x + terminal_spacing);
      const T term_h =
          static_cast<T>(terminal_size_y + terminal_spacing);
      const float pin_ox = static_cast<float>(term_w) / 2.f;
      const float pin_oy = static_cast<float>(term_h) / 2.f;

      terminal_count++;
      // LOG(INFO, "terminal_count: %d", terminal_count);
      if (terminal_legalize_flag) {
        int x, y;
        // LOG(INFO, "net_names[net_id]: %s", net_names[net_id]);
        for (int terminal_id = 0; terminal_id < num_terminals; ++terminal_id) {
          if (net_name == terminal_names[terminal_id]) {
            // dreamplace pos is left-bottom corner (includes spacing box)
            x = terminal_x[terminal_id];
            y = terminal_y[terminal_id];
            // LOG(INFO, "net_id: %d, terminal_count: %d, i: %d", net_id,
            //     terminal_count, i);
            break;
          }
        }
        terminalAuxRef.add_node(
            net_name, terminal_size_x + terminal_spacing,
            terminal_size_y + terminal_spacing, x, y, MOVABLE);
      } else {
        const T x = center_x - term_w / 2;
        const T y = center_y - term_h / 2;
        terminalAuxRef.add_node(
            net_name, terminal_size_x + terminal_spacing,
            terminal_size_y + terminal_spacing, x, y, MOVABLE);
      }
      for (int tier_id = 0; tier_id < num_tiers; ++tier_id) {
        string tier_net_name = net_name + "_T" + to_string(tier_id);

        // LOG(DEBUG, "Intersection Center: (%f, %f)", center_x, center_y);

        terminalAuxRef.add_pin(tier_net_name, net_name, 'O', pin_ox, pin_oy);
      }
    }
  }
}

template <typename T>
void terminalAuxLauncher(
    const int *tier, const int *flat_netpin, const int *netpin_start,
    const int *pin2node_map, int num_nets, int num_tiers, int num_movable_nodes,
    int num_pins, const T *node_size_x, const T *node_size_y,
    const T *pin_offset_x, const T *pin_offset_y, float die_size_x,
    float die_size_y, pybind11::list row_height, int terminal_size_x,
    int terminal_size_y, int terminal_spacing, int *cut_net_mask,
    const T *pin_x, const T *pin_y, const std::vector<std::string> &node_names,
    const std::vector<std::string> &net_names, const T *pos_2d_x,
    const T *pos_2d_y, std::string aux_dir, bool terminal_legalize_flag,
    const T *terminal_x, const T *terminal_y, int num_terminals,
    const std::vector<std::string> &terminal_names) {
  if (!aux_dir.empty() && aux_dir.back() != '/') {
    aux_dir += "/";
  }
  char IO_type;

  AUX terminal_aux = AUX(aux_dir, "terminal");

  for (int net_id = 0; net_id < num_nets; ++net_id) {
    const std::string net_name = bookshelfNetName(net_names[net_id], net_id);
    int num_nodes_in_net = 0;
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
      cut_net_mask[net_id] = 1;

      for (int tier_id = 0; tier_id < num_tiers; ++tier_id) {
        string tier_net_name = net_name + "_T" + to_string(tier_id);

        if (!terminal_aux.check_net_exist(tier_net_name)) {
          terminal_aux.add_net(tier_net_name);
          for (int pin_id = netpin_start[net_id];
               pin_id < netpin_start[net_id + 1]; ++pin_id) {
            int node_id = pin2node_map[flat_netpin[pin_id]];
            if (node_id < 0 || node_id >= num_movable_nodes) {
              continue;
            }
            if (tier[node_id] == tier_id) {
              int index_node = num_movable_nodes * tier_id + node_id;

              const std::string node_name = bookshelfNodeName(node_names[node_id], node_id);
              if (!terminal_aux.check_node_exist(node_name)) {
                int node_pos_x = pos_2d_x[node_id];
                int node_pos_y = pos_2d_y[node_id];
                terminal_aux.add_node(node_name, 0, 0, node_pos_x,
                                      node_pos_y, TERMINAL_NI);
              }
              if (!terminal_aux.check_pin_exist(tier_net_name, node_name)) {
                int index_pin = num_pins * tier_id + pin_id;
                (pin_id == netpin_start[net_id]) ? IO_type = 'I'
                                                 : IO_type = 'O';
                // when add pin to aux file
                terminal_aux.add_pin(
                    tier_net_name, node_name, IO_type,
                    static_cast<float>(pin_offset_x[index_pin] -
                                       ceil(node_size_x[index_node] / 2)),
                    static_cast<float>(pin_offset_y[index_pin] -
                                       ceil(node_size_y[index_node] / 2)));
              }
            }
          }
        }
      }
    }
  }

  terminal_insert(tier, pin_x, pin_y, flat_netpin, netpin_start, pin2node_map,
                  num_movable_nodes, num_nets, num_pins, num_tiers,
                  cut_net_mask, terminal_aux, terminal_size_x, terminal_size_y,
                  terminal_spacing, node_names, net_names,
                  terminal_legalize_flag, terminal_x, terminal_y, num_terminals,
                  terminal_names);

  // write aux files
  int tier_row_height = terminal_size_y + terminal_spacing;
  terminal_aux.set_default_rows(die_size_x, tier_row_height,
                                die_size_y / tier_row_height);
  // sort node by name
  terminal_aux.sort_node();
  terminal_aux.write_files();
}

at::Tensor terminal_aux_forward(
    at::Tensor tier, at::Tensor flat_netpin, at::Tensor netpin_start,
    at::Tensor pin2node_map, at::Tensor net_weights, int num_movable_nodes,
    at::Tensor node_size_x, at::Tensor node_size_y, at::Tensor pin_offset_x,
    at::Tensor pin_offset_y, float die_size_x, float die_size_y,
    pybind11::list row_height, int terminal_size_x, int terminal_size_y,
    int terminal_spacing, at::Tensor pin_pos,
    const std::vector<std::string> &node_names,
    const std::vector<std::string> &net_names, at::Tensor pos_2d,
    std::string aux_dir, bool terminal_legalize_flag,
    at::Tensor pos_terminal_legalized, int num_terminals,
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

  CHECK_FLAT_CPU(pos_terminal_legalized);
  CHECK_EVEN(pos_terminal_legalized);
  CHECK_CONTIGUOUS(pos_terminal_legalized);

  int num_nets = netpin_start.numel() - 1;
  int num_pins = pin2node_map.numel();

  // TODO: get num_tiers from param.json
  int num_tiers = 2;
  at::Tensor cut_net_mask = at::zeros(num_nets, tier.options());

  DREAMPLACE_DISPATCH_FLOATING_TYPES(node_size_x, "terminalAuxLauncher", [&] {
    terminalAuxLauncher<scalar_t>(
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
        aux_dir, terminal_legalize_flag,
        DREAMPLACE_TENSOR_DATA_PTR(pos_terminal_legalized, scalar_t),
        DREAMPLACE_TENSOR_DATA_PTR(pos_terminal_legalized, scalar_t) +
            pos_terminal_legalized.numel() / 2,
        num_terminals, terminal_names);
  });

  return cut_net_mask;
}

PLACER_END_NAMESPACE

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("terminal_aux", &PLACER_NAMESPACE::terminal_aux_forward,
        "terminal_aux_forward");
}
