/*
 * @Author: JeanneWillis hi@jeannewillis.cn
 * @Date: 2025-03-19 11:49:04
 * @LastEditors: JeanneWillis hi@jeannewillis.cn
 * @LastEditTime: 2025-03-24 23:48:31
 * @FilePath: /D2D-placer/src/ops/partition/src/partition.cpp
 * @Description: partition
 */
#include <pybind11/pybind11.h>
// 3d-placer mincut_partition by hmetis
#include "placer/placer.h"

PLACER_BEGIN_NAMESPACE

int partition_forward()
{
  return 0;
}

PLACER_END_NAMESPACE

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("partition", &PLACER_NAMESPACE::partition_forward, "partition_forward");
}
