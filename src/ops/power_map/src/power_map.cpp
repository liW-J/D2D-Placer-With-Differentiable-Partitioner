/*
 * @Author: JeanneWillis hi@jeannewillis.cn
 * @Date: 2025-03-18 16:21:55
 * @LastEditors: JeanneWillis hi@jeannewillis.cn
 * @LastEditTime: 2025-03-24 23:31:47
 * @FilePath: /D2D-placer/src/ops/power_map/src/power_map.cpp
 * @Description: Compute power map on CPU
 */
#include <pybind11/pybind11.h>
#include "utility/src/torch.h"
#include "utility/src/utils.h"

#include "include/common.h"

PLACER_BEGIN_NAMESPACE

template <typename T, typename AtomicOp>
void distributeBox2Bin(const int num_bins_x, const int num_bins_y, 
                       const T xl, const T yl, const T xh, const T yh, 
                       const T bin_size_x, const T bin_size_y,
                       T bxl, T byl, T bxh, T byh, AtomicOp atomic_add_op, 
                       typename AtomicOp::type* buf_map, T power) {
  // density overflow function plus power
  auto computeDensityFunc = [](T node_xl, T node_xh, T bin_xl, T bin_xh, T node_power) {
    return DREAMPLACE_STD_NAMESPACE::max(
        T(0.0),
        DREAMPLACE_STD_NAMESPACE::min(node_xh, bin_xh) -
            DREAMPLACE_STD_NAMESPACE::max(node_xl, bin_xl))/(node_xh - node_xl) * node_power;
  };
  // x direction
  int bin_index_xl = int((bxl - xl) / bin_size_x);
  int bin_index_xh = int(ceil((bxh - xl) / bin_size_x)) + 1;  // exclusive
  bin_index_xl = DREAMPLACE_STD_NAMESPACE::max(bin_index_xl, 0);
  bin_index_xh = DREAMPLACE_STD_NAMESPACE::min(bin_index_xh, num_bins_x);

  // y direction
  int bin_index_yl = int((byl - yl - 2 * bin_size_y) / bin_size_y);
  int bin_index_yh =
      int(ceil((byh - yl + 2 * bin_size_y) / bin_size_y)) + 1;  // exclusive
  bin_index_yl = DREAMPLACE_STD_NAMESPACE::max(bin_index_yl, 0);
  bin_index_yh = DREAMPLACE_STD_NAMESPACE::min(bin_index_yh, num_bins_y);

  for (int k = bin_index_xl; k < bin_index_xh; ++k) {
    T bin_xl = xl + bin_size_x * k;
    T bin_xh = DREAMPLACE_STD_NAMESPACE::min(bin_xl + bin_size_x, xh); 
    // special treatment for rightmost bins 
    if (k + 1 == num_bins_x) {
      bin_xh = bxh; 
    }
    T px = computeDensityFunc(bxl, bxh, bin_xl, bin_xh, power);
    for (int h = bin_index_yl; h < bin_index_yh; ++h) {
      T bin_yl = yl + bin_size_y * h; 
      T bin_yh = DREAMPLACE_STD_NAMESPACE::min(bin_yl + bin_size_y, yh); 
      // special treatment for upmost bins
      if (h + 1 == num_bins_y) {
        bin_yh = byh; 
      }
      T py = computeDensityFunc(byl, byh, bin_yl, bin_yh, power);

      // still area
      atomic_add_op(&buf_map[k * num_bins_y + h], px * py);
    }
  }
}

/// @brief compute density map
/// @param x_tensor cell x locations
/// @param y_tensor cell y locations
/// @param node_size_x_tensor cell width array
/// @param node_size_y_tensor cell height array
/// @param num_nodes number of cells
/// @param num_bins_x number of bins in horizontal bins
/// @param num_bins_y number of bins in vertical bins
/// @param xl left boundary
/// @param yl bottom boundary
/// @param xh right boundary
/// @param yh top boundary
/// @param num_threads number of threads
/// @param atomic_add_op functional object for atomic add 
/// @param buf_map 2D density map in column-major to write
template <typename T, typename AtomicOp>
int computePowerMapLauncher(const T* x_tensor, const T* y_tensor,
                              const T* node_size_x_tensor,
                              const T* node_size_y_tensor,
                              const T* power_tensor,
                              const int num_nodes,
                              const int num_bins_x, const int num_bins_y,
                              const T xl, const T yl, const T xh, const T yh,
                              int num_threads, AtomicOp atomic_add_op,
                              typename AtomicOp::type* buf_map) {
  // power_map_tensor should be initialized outside

  T bin_size_x = (xh - xl) / num_bins_x; 
  T bin_size_y = (yh - yl) / num_bins_y; 

#pragma omp parallel for num_threads(num_threads)
  for (int i = 0; i < num_nodes; ++i) {
    T bxl = x_tensor[i];
    T byl = y_tensor[i];
    T bxh = bxl + node_size_x_tensor[i];
    T byh = byl + node_size_y_tensor[i];
    T power = power_tensor[i];
    distributeBox2Bin(num_bins_x, num_bins_y, 
        xl, yl, xh, yh, 
        bin_size_x, bin_size_y, 
        bxl, byl, bxh, byh, 
        atomic_add_op, buf_map, power);
  }

  return 0;
}

/// @brief Compute density map.
/// @param pos cell locations, array of x locations and then y locations
/// @param node_size_x_tensor cell width array
/// @param node_size_y_tensor cell height array
/// @param initial_power_map initial density map
/// @param xl left boundary
/// @param yl bottom boundary
/// @param xh right boundary
/// @param yh top boundary
/// @param num_bins_x number of bins in x direction
/// @param num_bins_y number of bins in y direction
/// @param range_begin begin index 
/// @param range_end end index 
/// @param deterministic_flag whether ensure run-to-run determinism 
/// @return power map
at::Tensor power_map_forward(at::Tensor pos, at::Tensor node_size_x,
                               at::Tensor node_size_y, at::Tensor initial_power_map, 
                               double xl, double yl, double xh, double yh,
                               int num_bins_x, int num_bins_y, 
                               int range_begin, int range_end, 
                               int deterministic_flag, at::Tensor power) {
  CHECK_FLAT_CPU(pos);
  CHECK_EVEN(pos);
  CHECK_CONTIGUOUS(pos);

  at::Tensor power_map = initial_power_map.clone();

  DREAMPLACE_DISPATCH_FLOATING_TYPES(
      pos, "computePowerMapLauncher", [&] {
        if (deterministic_flag) {
            double diearea = (xh - xl) * (yh - yl);
            int integer_bits = DREAMPLACE_STD_NAMESPACE::max((int)ceil(log2(diearea)) + 1, 32);
            int fraction_bits = DREAMPLACE_STD_NAMESPACE::max(64 - integer_bits, 0);
            long scale_factor = (1L << fraction_bits);
            int num_bins = num_bins_x * num_bins_y;

            std::vector<long> buf_map(num_bins, 0);
            DREAMPLACE_NAMESPACE::AtomicAdd<long> atomic_add_op(scale_factor);

            computePowerMapLauncher<scalar_t, decltype(atomic_add_op)>(
                DREAMPLACE_TENSOR_DATA_PTR(pos, scalar_t) + range_begin,
                DREAMPLACE_TENSOR_DATA_PTR(pos, scalar_t) + pos.numel() / 2 + range_begin,
                DREAMPLACE_TENSOR_DATA_PTR(node_size_x, scalar_t) + range_begin,
                DREAMPLACE_TENSOR_DATA_PTR(node_size_y, scalar_t) + range_begin,
                DREAMPLACE_TENSOR_DATA_PTR(power, scalar_t) + range_begin,
                range_end - range_begin, 
                num_bins_x, num_bins_y, 
                xl, yl, xh, yh, 
                at::get_num_threads(),
                atomic_add_op, buf_map.data());

            DREAMPLACE_NAMESPACE::scaleAdd(DREAMPLACE_TENSOR_DATA_PTR(power_map, scalar_t),
                     buf_map.data(), 1.0 / scale_factor, num_bins,
                     at::get_num_threads());
        } else {
            DREAMPLACE_NAMESPACE::AtomicAdd<scalar_t> atomic_add_op;
            computePowerMapLauncher<scalar_t, decltype(atomic_add_op)>(
                DREAMPLACE_TENSOR_DATA_PTR(pos, scalar_t) + range_begin,
                DREAMPLACE_TENSOR_DATA_PTR(pos, scalar_t) + pos.numel() / 2 + range_begin,
                DREAMPLACE_TENSOR_DATA_PTR(node_size_x, scalar_t) + range_begin,
                DREAMPLACE_TENSOR_DATA_PTR(node_size_y, scalar_t) + range_begin,
                DREAMPLACE_TENSOR_DATA_PTR(power, scalar_t) + range_begin,
                range_end - range_begin, 
                num_bins_x, num_bins_y, 
                xl, yl, xh, yh, 
                at::get_num_threads(),
                atomic_add_op, 
                DREAMPLACE_TENSOR_DATA_PTR(power_map, scalar_t)); 
        }
      });

  return power_map;
}

PLACER_END_NAMESPACE

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("power_map", &PLACER_NAMESPACE::power_map_forward, "power_map_forward");
}



