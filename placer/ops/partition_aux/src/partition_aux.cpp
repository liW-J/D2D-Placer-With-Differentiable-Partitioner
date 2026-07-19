/*
 * @Author: JeanneWillis hi@jeannewillis.cn
 * @Date: 2025-03-19 11:49:04
 * @LastEditors: JeanneWillis hi@jeannewillis.cn
 * @LastEditTime: 2025-07-23 15:39:40
 * @FilePath: /D2D-placer/src/ops/partition/src/partition.cpp
 * @Description: partition
 */
#include <pybind11/pybind11.h>
#include <chrono>
#include <cctype>
#include <unordered_map>
// dreamplace
#include "utility/src/torch.h"
#include "utility/src/utils.h"

// 3d-placer parser
#include "include/common.h"
#include "placer/placer.h"

#include "utils_3d/src/partitioner.h"

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
  vector<T> max_x(num_tiers);
  vector<T> min_x(num_tiers);
  vector<T> max_y(num_tiers);
  vector<T> min_y(num_tiers);
  std::unordered_map<std::string, int> terminal_name_to_id;
  if (terminal_legalize_flag) {
    terminal_name_to_id.reserve(num_terminals);
    for (int terminal_id = 0; terminal_id < num_terminals; ++terminal_id) {
      terminal_name_to_id.emplace(terminal_names[terminal_id], terminal_id);
    }
  }

  for (int net_id = 0; net_id < num_nets; net_id++) {
    if (!cut_net_mask[net_id]) {
      continue;
    }
    std::fill(max_x.begin(), max_x.end(),
              -std::numeric_limits<T>::max());
    std::fill(min_x.begin(), min_x.end(),
              std::numeric_limits<T>::max());
    std::fill(max_y.begin(), max_y.end(),
              -std::numeric_limits<T>::max());
    std::fill(min_y.begin(), min_y.end(),
              std::numeric_limits<T>::max());

    const std::string net_name = bookshelfNetName(net_names[net_id], net_id);
    // Each pin belongs to exactly one tier.  A single pass is sufficient;
    // the previous implementation traversed every net once per tier.
    for (int pin_id = netpin_start[net_id];
         pin_id < netpin_start[net_id + 1]; pin_id++) {
      const int original_pin_id = flat_netpin[pin_id];
      const int node_id = pin2node_map[original_pin_id];
      if (node_id < 0 || node_id >= num_movable_nodes) {
        continue;
      }
      const int tier_id = tier[node_id];
      if (tier_id < 0 || tier_id >= num_tiers) {
        continue;
      }
      const int index_pin = num_pins * tier_id + original_pin_id;
      max_x[tier_id] = std::max(max_x[tier_id], pin_x[index_pin]);
      min_x[tier_id] = std::min(min_x[tier_id], pin_x[index_pin]);
      max_y[tier_id] = std::max(max_y[tier_id], pin_y[index_pin]);
      min_y[tier_id] = std::min(min_y[tier_id], pin_y[index_pin]);
    }

    const T inner_min_x = *max_element(min_x.begin(), min_x.end());
    const T inner_max_x = *min_element(max_x.begin(), max_x.end());
    const T inner_min_y = *max_element(min_y.begin(), min_y.end());
    const T inner_max_y = *min_element(max_y.begin(), max_y.end());
    // Bonding center from cross-tier pin bbox; DREAMPlace pos is left-bottom.
    const T center_x = (inner_min_x + inner_max_x) / 2;
    const T center_y = (inner_min_y + inner_max_y) / 2;
    const float pin_ox = static_cast<float>(terminal_size_x) / 2.f;
    const float pin_oy = static_cast<float>(terminal_size_y) / 2.f;

    terminal_count++;
    for (int tier_id = 0; tier_id < num_tiers; ++tier_id) {
      if (terminal_legalize_flag) {
        const auto found = terminal_name_to_id.find(net_name);
        if (found != terminal_name_to_id.end()) {
          const int terminal_id = found->second;
          // dp_terminal pos is outer-box left-bottom; tier NI is terminalSize
          // box with the same bonding center -> shift by spacing/2.
          const int x = terminal_x[terminal_id] + terminal_spacing / 2;
          const int y = terminal_y[terminal_id] + terminal_spacing / 2;
          auxListRef[tier_id].add_node(net_name, terminal_size_x,
                                       terminal_size_y, x, y, TERMINAL_NI);
        } else {
          LOG(WARN, "terminal %s is missing from legalized terminal map; "
                    "fall back to intersection center", net_name.c_str());
          const T x = center_x - static_cast<T>(terminal_size_x) / 2;
          const T y = center_y - static_cast<T>(terminal_size_y) / 2;
          auxListRef[tier_id].add_node(net_name, terminal_size_x,
                                       terminal_size_y, x, y, TERMINAL_NI);
        }
      } else {
        const T x = center_x - static_cast<T>(terminal_size_x) / 2;
        const T y = center_y - static_cast<T>(terminal_size_y) / 2;
        auxListRef[tier_id].add_node(net_name, terminal_size_x,
                                     terminal_size_y, x, y, TERMINAL_NI);
      }
      auxListRef[tier_id].add_pin(net_name, net_name, 'O', pin_ox, pin_oy);
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
    const T *pos_2d_y, std::string aux_dir,
    const std::vector<std::string> &node_orient) {
  if (!aux_dir.empty() && aux_dir.back() != '/') {
    aux_dir += "/";
  }
  char IO_type;
  vector<AUX> aux_list(num_tiers);

  // init aux files
  for (int tier_id = 0; tier_id < num_tiers; ++tier_id) {
    string tier_name = "tier" + to_string(tier_id);
    aux_list[tier_id] = AUX(aux_dir, tier_name);
  }

  // AUX::check_node_exist scans every previously inserted node.  Calling it
  // for every pin makes tier Bookshelf construction quadratic on large
  // designs.  Node ids are dense here, so two compact marker arrays provide
  // O(1) node-existence and per-net duplicate-pin checks.
  vector<unsigned char> node_added(num_movable_nodes, 0);
  vector<int> node_seen_in_net(num_movable_nodes, -1);
  vector<int> node_count(num_tiers, 0);
  vector<unsigned char> net_added(num_tiers, 0);
  const auto aux_build_begin = std::chrono::steady_clock::now();
  LOG(INFO, "building tier AUX topology for %d nets, %d pins, %d nodes",
      num_nets, num_pins, num_movable_nodes);

  for (int net_id = 0; net_id < num_nets; ++net_id) {
    if (net_id > 0 && net_id % 500000 == 0) {
      const double seconds = std::chrono::duration<double>(
          std::chrono::steady_clock::now() - aux_build_begin).count();
      LOG(INFO, "tier AUX topology progress: %d/%d nets (%.1f s)", net_id,
          num_nets, seconds);
    }
    const std::string net_name = bookshelfNetName(net_names[net_id], net_id);
    int num_nodes_in_net = 0;
    std::fill(node_count.begin(), node_count.end(), 0);
    std::fill(net_added.begin(), net_added.end(), 0);
    for (int pin_id = netpin_start[net_id];
         pin_id < netpin_start[net_id + 1]; ++pin_id) {
      const int original_pin_id = flat_netpin[pin_id];
      const int node_id = pin2node_map[original_pin_id];
      if (node_id < 0 || node_id >= num_movable_nodes ||
          node_seen_in_net[node_id] == net_id) {
        continue;
      }
      node_seen_in_net[node_id] = net_id;

      const int tier_id = tier[node_id];
      if (tier_id < 0 || tier_id >= num_tiers) {
        continue;
      }
      const int index_node = num_movable_nodes * tier_id + node_id;
      node_count[tier_id]++;
      num_nodes_in_net++;

      if (!net_added[tier_id]) {
        aux_list[tier_id].add_net(net_name);
        net_added[tier_id] = 1;
      }
      const std::string node_name = bookshelfNodeName(node_names[node_id],
                                                       node_id);
      if (!node_added[node_id]) {
        const int node_pos_x =
            (pos_2d_x == nullptr) ? 0 : pos_2d_x[node_id];
        const int node_pos_y =
            (pos_2d_y == nullptr) ? 0 : pos_2d_y[node_id];
        aux_list[tier_id].add_node(node_name, node_size_x[index_node],
                                   node_size_y[index_node], node_pos_x,
                                   node_pos_y, MOVABLE);
        node_added[node_id] = 1;
      }

      const int index_pin = num_pins * tier_id + original_pin_id;
      IO_type = (pin_id == netpin_start[net_id]) ? 'I' : 'O';
      // pin offsets are relative to the node center in DREAMPlace; AUX uses
      // offsets relative to the lower-left corner.
      aux_list[tier_id].add_pin(
          net_name, node_name, IO_type,
          static_cast<float>(pin_offset_x[index_pin] -
                             ceil(node_size_x[index_node] / 2)),
          static_cast<float>(pin_offset_y[index_pin] -
                             ceil(node_size_y[index_node] / 2)));
    }
    // if all nodes are in the same tier, not cut
    if (num_nodes_in_net > 0 &&
        std::find(node_count.begin(), node_count.end(), num_nodes_in_net) ==
        node_count.end()) {
      cut_net_mask[net_id] = 1;
      // LOG(DEBUG, "cut.");
    }
  }
  const double aux_build_seconds = std::chrono::duration<double>(
      std::chrono::steady_clock::now() - aux_build_begin).count();
  LOG(INFO, "tier AUX topology built in %.1f s", aux_build_seconds);

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
    std::string aux_dir, const std::vector<std::string> &node_orient) {
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
        aux_dir, node_orient);
  });

  return cut_net_mask;
}

PLACER_END_NAMESPACE

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("partition_aux", &PLACER_NAMESPACE::partition_aux_forward,
        "partition_aux_forward");
}
