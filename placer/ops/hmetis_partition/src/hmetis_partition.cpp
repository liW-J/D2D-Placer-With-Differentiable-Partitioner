/*
 * @Author: JeanneWillis hi@jeannewillis.cn
 * @Date: 2025-04-08 12:35:48
 * @LastEditors: JeanneWillis hi@jeannewillis.cn
 * @LastEditTime: 2025-04-08 21:13:17
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

int hmetis_partition_forward(at::Tensor flat_netpin, at::Tensor netpin_start,
                             at::Tensor pin2net_map, at::Tensor net_weights,
                             at::Tensor net_mask)
{
  CHECK_FLAT_CPU(flat_netpin);
  CHECK_CONTIGUOUS(flat_netpin);
  CHECK_FLAT_CPU(netpin_start);
  CHECK_CONTIGUOUS(netpin_start);
  CHECK_FLAT_CPU(pin2net_map);
  CHECK_CONTIGUOUS(pin2net_map);
  CHECK_FLAT_CPU(net_weights);
  CHECK_CONTIGUOUS(net_weights);
  CHECK_FLAT_CPU(net_mask);
  CHECK_CONTIGUOUS(net_mask);
  // write hgr file
  HGR hgr("./run_tmp/case2/", "circuit");
  LOG(WARN, "hmetis_partition_forward");
}

PLACER_END_NAMESPACE

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
  m.def("hmetis_partition", &PLACER_NAMESPACE::hmetis_partition_forward, "hmetis_partition_forward");
}
