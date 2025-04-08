/*
 * @Author: JeanneWillis hi@jeannewillis.cn
 * @Date: 2025-04-08 12:35:48
 * @LastEditors: JeanneWillis hi@jeannewillis.cn
 * @LastEditTime: 2025-04-08 23:55:05
 * @FilePath: /D2D-placer/placer/ops/hmetis_partition/src/hmetis_partition.cpp
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

int hmetisPartitionLauncher(const int* flat_netpin, const int* netpin_start, const int* pin2node_map,
                        const unsigned char* net_mask, int num_nets, int num_threads) {

  // write hgr file
  HGR hgr("./run_tmp/case1/", "circuit");

  #pragma omp parallel for num_threads(num_threads)
  for (int i = 0; i < num_nets; ++i) {

    if (net_mask[i]) {
      for (int j = netpin_start[i]; j < netpin_start[i + 1]; ++j) {
        LOG(INFO, "net %d, pin %d, node %d", i, flat_netpin[j], pin2node_map[flat_netpin[j]]);
      }
    }
  }

  return 0;
}

int hmetis_partition_forward(at::Tensor flat_netpin, at::Tensor netpin_start,
                             at::Tensor pin2node_map, at::Tensor net_weights,
                             at::Tensor net_mask)
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
  
  LOG(INFO, "hmetis_partition_forward---begin");

  int num_nets = netpin_start.numel() - 1;
  // at::Tensor hpwl = at::zeros(num_nets, pos.options());

  hmetisPartitionLauncher(
      DREAMPLACE_TENSOR_DATA_PTR(flat_netpin, int),
      DREAMPLACE_TENSOR_DATA_PTR(netpin_start, int),
      DREAMPLACE_TENSOR_DATA_PTR(pin2node_map, int),
      DREAMPLACE_TENSOR_DATA_PTR(net_mask, unsigned char), num_nets,
      at::get_num_threads());
  // if (net_weights.numel()) {
  //   hpwl.mul_(net_weights);
  // }
  return true;

  LOG(INFO, "hmetis_partition_forward---end");
}

PLACER_END_NAMESPACE

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
  m.def("hmetis_partition", &PLACER_NAMESPACE::hmetis_partition_forward, "hmetis_partition_forward");
}
