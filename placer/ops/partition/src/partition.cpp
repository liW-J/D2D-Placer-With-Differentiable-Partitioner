/*
 * @Author: JeanneWillis hi@jeannewillis.cn
 * @Date: 2025-03-19 11:49:04
 * @LastEditors: JeanneWillis hi@jeannewillis.cn
 * @LastEditTime: 2025-04-14 01:57:13
 * @FilePath: /D2D-placer/src/ops/partition/src/partition.cpp
 * @Description: partition
 */
#include <pybind11/pybind11.h>
// dreamplace
#include "utility/src/torch.h"
#include "utility/src/utils.h"

// 3d-placer parser
#include "placer/placer.h"
#include "include/common.h"

PLACER_BEGIN_NAMESPACE

string node2name(int node_id)
{
  return "C" + to_string(node_id+1);
}

string net2name(int net_id)
{
  return "N" + to_string(net_id+1);
}

template <typename T>
void partitionLauncher(const T *tier, const int *flat_netpin, const int *netpin_start, const int *pin2node_map,
                       const unsigned char *net_mask, int num_nets, int num_tiers, int num_movable_nodes,
                       const T *node_size_x, const T *node_size_y, const T *pin_offset_x, const T *pin_offset_y,
                       float die_size_x, float die_size_y, pybind11::list row_height)
{
  string aux_dir = "./run_tmp/case1/partition/";
  char IO_type;
  vector<AUX> aux_list(num_tiers);

  for (int tier_id = 0; tier_id < num_tiers; ++tier_id)
  {
    string tier_name = "tier" + to_string(tier_id);
    aux_list[tier_id] = AUX(aux_dir, tier_name);
    for (int net_id = 0; net_id < num_nets; ++net_id)
    {
      if (net_mask[net_id])
      {
        for (int pin_id = netpin_start[net_id]; pin_id < netpin_start[net_id + 1]; ++pin_id)
        {
          int node_id = pin2node_map[flat_netpin[pin_id]];
          if (tier[node_id] == tier_id && !aux_list[tier_id].check_node_exist(node2name(node_id)))
          {
            if (!aux_list[tier_id].check_net_exist(net2name(net_id)))
            {
              aux_list[tier_id].add_net(net2name(net_id));
            }
            int index = num_movable_nodes * tier_id + node_id;
            LOG(DEBUG, "node %d, tier %d, node_size_index %d, node_size_x %f, node_size_y %f", node_id, tier_id, index, node_size_x[index], node_size_y[index]);
            // TODO: cell width and height
            aux_list[tier_id].add_node(node2name(node_id), node_size_x[index], node_size_y[index], 0, 0, 0);
            (pin_id == netpin_start[net_id]) ? IO_type = 'I' : IO_type = 'O';
            // unable center func type, use explicit type
            aux_list[tier_id].add_pin(net2name(net_id), node2name(node_id), IO_type,
                                      static_cast<float>(pin_offset_x[index]), static_cast<float>(pin_offset_y[index]));
          }
        }
      }
    }
    // pybind11 type convert
    int tier_row_height = pybind11::cast<float>(row_height[tier_id]);
    aux_list[tier_id].set_default_rows(die_size_x, tier_row_height, die_size_y / tier_row_height);
    LOG(DEBUG, "test-row-height: %d", tier_row_height);
    LOG(DEBUG, "die_size_x: %f", die_size_x);
    LOG(DEBUG, "die_size_y: %f", die_size_y);
    // Attention: still not add terminal
    aux_list[tier_id].write_files();
  }
}

int partition_forward(at::Tensor tier, at::Tensor flat_netpin, at::Tensor netpin_start,
                      at::Tensor pin2node_map, at::Tensor net_weights,
                      at::Tensor net_mask, int num_movable_nodes,
                      at::Tensor node_size_x, at::Tensor node_size_y,
                      at::Tensor pin_offset_x, at::Tensor pin_offset_y,
                      float die_size_x, float die_size_y,
                      pybind11::list row_height)
{
  CHECK_FLAT_CPU(flat_netpin);
  CHECK_CONTIGUOUS(flat_netpin);
  CHECK_FLAT_CPU(netpin_start);
  CHECK_CONTIGUOUS(netpin_start);
  CHECK_FLAT_CPU(pin2node_map);
  CHECK_CONTIGUOUS(pin2node_map);
  CHECK_FLAT_CPU(net_weights);
  CHECK_CONTIGUOUS(net_weights);
  CHECK_FLAT_CPU(net_mask);
  CHECK_CONTIGUOUS(net_mask);

  CHECK_CONTIGUOUS(pin_offset_x);
  CHECK_CONTIGUOUS(pin_offset_y);
  CHECK_CONTIGUOUS(node_size_x);
  CHECK_CONTIGUOUS(node_size_y);

  // CHECK_FLAT_CPU(tier);
  CHECK_CONTIGUOUS(tier);

  int num_nets = netpin_start.numel() - 1;
  int num_tiers = 2;

  DREAMPLACE_DISPATCH_FLOATING_TYPES(tier, "partitionLauncher", [&]
                                     { partitionLauncher<scalar_t>(
                                           DREAMPLACE_TENSOR_DATA_PTR(tier, scalar_t),
                                           DREAMPLACE_TENSOR_DATA_PTR(flat_netpin, int),
                                           DREAMPLACE_TENSOR_DATA_PTR(netpin_start, int),
                                           DREAMPLACE_TENSOR_DATA_PTR(pin2node_map, int),
                                           DREAMPLACE_TENSOR_DATA_PTR(net_mask, unsigned char),
                                           num_nets,
                                           num_tiers,
                                           num_movable_nodes,
                                           DREAMPLACE_TENSOR_DATA_PTR(node_size_x, scalar_t),
                                           DREAMPLACE_TENSOR_DATA_PTR(node_size_y, scalar_t),
                                           DREAMPLACE_TENSOR_DATA_PTR(pin_offset_x, scalar_t),
                                           DREAMPLACE_TENSOR_DATA_PTR(pin_offset_y, scalar_t),
                                           die_size_x,
                                           die_size_y,
                                           row_height); });

  return 0;
}

PLACER_END_NAMESPACE

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
  m.def("partition", &PLACER_NAMESPACE::partition_forward, "partition_forward");
}
