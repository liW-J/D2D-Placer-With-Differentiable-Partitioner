/*
 * @Author: JeanneWillis hi@jeannewillis.cn
 * @Date: 2025-04-08 12:35:48
 * @LastEditors: JeanneWillis hi@jeannewillis.cn
 * @LastEditTime: 2025-08-24 15:24:13
 * @FilePath: /D2D-placer/placer/ops/parts_reader/src/parts_reader.cpp
 * @Description:
 */

#include <pybind11/pybind11.h>
// dreamplace
#include "utility/src/torch.h"
#include "utility/src/utils.h"
// 3d-placer parser
#include "include/common.h"
#include "placer/placer.h"

PLACER_BEGIN_NAMESPACE

int partsReaderLauncher(int *tier, const int *flat_netpin,
                        const int *netpin_start, const int *pin2node_map,
                        const unsigned char *net_mask, int num_nets,
                        int num_movable_nodes, int num_threads,
                        std::string partitioner_path) {

  // write hgr file
  HGR hgr(partitioner_path, "circuit");

  for (int net_id = 0; net_id < num_nets; ++net_id) {
    hgr.add_net(to_string(net_id));
    if (net_mask[net_id]) {
      for (int pin_id = netpin_start[net_id]; pin_id < netpin_start[net_id + 1];
           ++pin_id) {
        hgr.add_node(to_string(net_id),
                     to_string(pin2node_map[flat_netpin[pin_id]]));
      }
    }
  }

  hgr.read_part_result(2);

  assert(hgr.get_part_size(0) + hgr.get_part_size(1) == num_movable_nodes);
  LOG(INFO, "hmetis partition result: %d : %d", hgr.get_part_size(0),
      hgr.get_part_size(1));

  for (int i = 0; i < num_movable_nodes; ++i) {
    tier[i] = hgr.get_part_result(to_string(i));
  }

  LOG(INFO, "part reading completed");
  return 0;
}

at::Tensor parts_reader_forward(at::Tensor tier, at::Tensor flat_netpin,
                                at::Tensor netpin_start,
                                at::Tensor pin2node_map, at::Tensor net_weights,
                                at::Tensor net_mask, int num_movable_nodes,
                                std::string partitioner_path) {
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

  int num_nets = netpin_start.numel() - 1;

  partsReaderLauncher(DREAMPLACE_TENSOR_DATA_PTR(tier, int),
                      DREAMPLACE_TENSOR_DATA_PTR(flat_netpin, int),
                      DREAMPLACE_TENSOR_DATA_PTR(netpin_start, int),
                      DREAMPLACE_TENSOR_DATA_PTR(pin2node_map, int),
                      DREAMPLACE_TENSOR_DATA_PTR(net_mask, unsigned char),
                      num_nets, num_movable_nodes, at::get_num_threads(),
                      partitioner_path);

  return tier;
}

PLACER_END_NAMESPACE

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("parts_reader", &PLACER_NAMESPACE::parts_reader_forward,
        "parts_reader_forward");
}