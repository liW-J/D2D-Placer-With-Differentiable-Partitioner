/*
 * @Author: JeanneWillis hi@jeannewillis.cn
 * @Date: 2025-03-19 11:49:04
 * @LastEditors: JeanneWillis hi@jeannewillis.cn
 * @LastEditTime: 2025-04-10 16:12:04
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
  return "C" + to_string(node_id);
}

template <typename T>
void partitionLauncher(const T *tier, const int *flat_netpin, const int *netpin_start, const int *pin2node_map,
                const unsigned char *net_mask, int num_nets, int num_tiers, int num_threads)
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
      aux_list[tier_id].add_net(to_string(net_id));
      if (net_mask[net_id])
      {
        for (int pin_id = netpin_start[net_id]; pin_id < netpin_start[net_id + 1]; ++pin_id)
        {
          int node_id = pin2node_map[flat_netpin[pin_id]];
          if (tier[node_id] == tier_id && !aux_list[tier_id].check_node_exist(node2name(node_id)))
          {
            LOG(DEBUG, "node %d, tier %d", node_id, tier_id);
            // TODO: cell width and height
            aux_list[tier_id].add_node(node2name(node_id), 0, 0, 0, 0, 0);
            (pin_id == netpin_start[net_id]) ? IO_type = 'I' : IO_type = 'O';
            // TODO: pin offset
            aux_list[tier_id].add_pin(to_string(net_id), node2name(node_id), IO_type, 0, 0);
          }
        }
      }
    }
    aux_list[tier_id].set_default_rows(10, 10, 10);
    // Attention: still not add terminal
    aux_list[tier_id].write_files();
  }
}

int partition_forward(at::Tensor tier, at::Tensor flat_netpin, at::Tensor netpin_start,
                      at::Tensor pin2node_map, at::Tensor net_weights,
                      at::Tensor net_mask, int num_movable_nodes)
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

  CHECK_FLAT_CPU(tier);
  CHECK_EVEN(tier);
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
        num_movable_nodes,
        num_tiers); 
  });

  return 0;
}

PLACER_END_NAMESPACE

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
  m.def("partition", &PLACER_NAMESPACE::partition_forward, "partition_forward");
}
