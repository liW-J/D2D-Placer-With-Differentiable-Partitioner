/*
 * @Author: JeanneWillis hi@jeannewillis.cn
 * @Date: 2025-04-08 12:35:48
 * @LastEditors: JeanneWillis hi@jeannewillis.cn
 * @LastEditTime: 2025-04-23 19:07:48
 * @FilePath: /D2D-placer/placer/ops/hmetis/src/hmetis.cpp
 * @Description:
 */

#include <pybind11/pybind11.h>
// dreamplace
#include "utility/src/torch.h"
#include "utility/src/utils.h"
// 3d-placer parser
#include "placer/placer.h"
#include "include/common.h"

PLACER_BEGIN_NAMESPACE

template <typename T>
struct NodeState
{
  T max_x = -std::numeric_limits<T>::max();
  T min_x = std::numeric_limits<T>::max();
  T max_y = -std::numeric_limits<T>::max();
  T min_y = std::numeric_limits<T>::max();
};

template <typename T>
void tier_assign(T *tier, const T *x, const T *y, const int *flat_netpin, const int *netpin_start,
                 const int *pin2node_map, int num_movable_nodes, int num_nets, int num_tiers)
{
  LOG(WARN, "tier_assign");

  int reassign_count = 0;
  for (int net_id = 0; net_id < num_nets; net_id++)
  {
    map<int, NodeState<T>> node_state_map;
    int net_degree = netpin_start[net_id + 1] - netpin_start[net_id];
    LOG(ERROR, "net_id: %d, net_degree: %d", net_id, net_degree);

    T max_x = -std::numeric_limits<T>::max();
    T min_x = std::numeric_limits<T>::max();
    T max_y = -std::numeric_limits<T>::max();
    T min_y = std::numeric_limits<T>::max();

    for (int pin_id = netpin_start[net_id]; pin_id < netpin_start[net_id + 1]; pin_id++)
    {

      // every node only exist one in every net
      // which means every pin connect to different node
      // so node_id is only use to assign tier for pin
      min_x = std::min(min_x, x[pin_id]);
      max_x = std::max(max_x, x[pin_id]);
      min_y = std::min(min_y, y[pin_id]);
      max_y = std::max(max_y, y[pin_id]);
    }

    // LOG(WARN, "outer_min_x: %f, outer_max_x: %f, outer_min_y: %f, outer_max_y: %f", min_x, max_x, min_y, max_y);
    T cut_x = (min_x + max_x) / 2;
    T cut_y = (min_y + max_y) / 2;
    // LOG(WARN, "net_id: %d, cut_x: %f, cut_y: %f", net_id, cut_x, cut_y);
    for (int pin_id = netpin_start[net_id]; pin_id < netpin_start[net_id + 1]; pin_id++)
    {
      int node_id = pin2node_map[flat_netpin[pin_id]];
      // LOG(ERROR, "pin_id: %d, x: %f, tier: %f, node_id: %d", pin_id, x[pin_id], tier[node_id], node_id);
      if (x[pin_id] < cut_x && tier[node_id] == -1)
      {
        tier[node_id] = 0;
        // LOG(INFO, "pin_id: %d, x: %f, cut_x: %f", pin_id, x[pin_id], cut_x);
      }else if (x[pin_id] >= cut_x && tier[node_id] == -1)
      {
        tier[node_id] = 1;
        // LOG(INFO, "pin_id: %d, x: %f, cut_x: %f", pin_id, x[pin_id], cut_x);
      }else{
        LOG(DEBUG, "have assigned");
        // TODO: argue which is the best
        reassign_count++;
      }
    }
  }

  // check nodes all been assign
  auto it = std::find(tier, tier + num_movable_nodes, -1);
  if (it != tier + num_movable_nodes)
  {
    LOG(ERROR, "node_id: %d, tier: %d. NOT assign!", it - tier, *it);
  }
}

template <typename T>
int avgCutPartitionLauncher(T *tier, const int *flat_netpin, const int *netpin_start, const int *pin2node_map,
                            int num_nets, int num_tiers, int num_movable_nodes, int num_pins,
                            const T *x, const T *y)
{
  tier_assign(tier, x, y, flat_netpin, netpin_start, pin2node_map,
              num_movable_nodes, num_nets, num_tiers);

  // for (int i = 0; i < num_movable_nodes; ++i)
  // {
  //   tier[i] = hgr.get_part_result(to_string(i));
  // }

  LOG(INFO, "Running avg_cut completed");
  return 0;
}

at::Tensor avg_cut_forward(at::Tensor flat_netpin, at::Tensor netpin_start,
                           at::Tensor pin2node_map, at::Tensor net_weights,
                           int num_movable_nodes, at::Tensor pos)
{
  CHECK_FLAT_CPU(flat_netpin);
  CHECK_CONTIGUOUS(flat_netpin);
  CHECK_FLAT_CPU(netpin_start);
  CHECK_CONTIGUOUS(netpin_start);
  CHECK_FLAT_CPU(pin2node_map);
  CHECK_CONTIGUOUS(pin2node_map);
  CHECK_FLAT_CPU(net_weights);
  CHECK_CONTIGUOUS(net_weights);

  CHECK_FLAT_CPU(pos);
  CHECK_EVEN(pos);
  CHECK_CONTIGUOUS(pos);

  int num_nets = netpin_start.numel() - 1;
  int num_pins = pin2node_map.numel();

  // TODO: get num_tiers from param.json
  int num_tiers = 2;
  at::Tensor tier = at::full(num_movable_nodes, -1, pos.options());

  DREAMPLACE_DISPATCH_FLOATING_TYPES(pos, "avgCutPartitionLauncher", [&]
                                     { avgCutPartitionLauncher<scalar_t>(
                                           DREAMPLACE_TENSOR_DATA_PTR(tier, scalar_t),
                                           DREAMPLACE_TENSOR_DATA_PTR(flat_netpin, int),
                                           DREAMPLACE_TENSOR_DATA_PTR(netpin_start, int),
                                           DREAMPLACE_TENSOR_DATA_PTR(pin2node_map, int),
                                           num_nets,
                                           num_tiers,
                                           num_movable_nodes,
                                           num_pins,
                                           DREAMPLACE_TENSOR_DATA_PTR(pos, scalar_t),
                                           DREAMPLACE_TENSOR_DATA_PTR(pos, scalar_t) + pos.numel() / 2); });
  return tier;
}

PLACER_END_NAMESPACE

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
  m.def("avg_cut", &PLACER_NAMESPACE::avg_cut_forward, "avg_cut_forward");
}
