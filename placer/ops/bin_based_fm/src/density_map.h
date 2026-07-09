/**
 * @file   density_map.cpp
 * @author Yibo Lin
 * @date   Jun 2018
 * @brief  Compute density map on CPU
 */
#include "utility/src/torch.h"
#include "utility/src/utils.h"

#include "include/common.h"

PLACER_BEGIN_NAMESPACE

template <typename T, typename AtomicOp>
T distributeBox2Bin(const int num_bins_x, const int num_bins_y, const T xl, const T yl, const T xh,
                    const T yh, const T bin_size_x, const T bin_size_y, T bxl, T byl, T bxh, T byh,
                    AtomicOp atomic_add_op, typename AtomicOp::type* buf_map, int multiplier) {

  int distributed_num_bins = 0;
  T distributed_area = 0;
  // density overflow function
  auto computeDensityFunc = [](T node_xl, T node_xh, T bin_xl, T bin_xh) {
    return DREAMPLACE_STD_NAMESPACE::max(T(0.0),
                                         DREAMPLACE_STD_NAMESPACE::min(node_xh, bin_xh) -
                                             DREAMPLACE_STD_NAMESPACE::max(node_xl, bin_xl));
  };
  // x direction
  int bin_index_xl = int((bxl - xl) / bin_size_x);
  int bin_index_xh = int(ceil((bxh - xl) / bin_size_x)) + 1;  // exclusive
  bin_index_xl = DREAMPLACE_STD_NAMESPACE::max(bin_index_xl, 0);
  bin_index_xh = DREAMPLACE_STD_NAMESPACE::min(bin_index_xh, num_bins_x);

  // y direction
  int bin_index_yl = int((byl - yl) / bin_size_y);
  int bin_index_yh = int(ceil((byh - yl) / bin_size_y)) + 1;  // exclusive
  bin_index_yl = DREAMPLACE_STD_NAMESPACE::max(bin_index_yl, 0);
  bin_index_yh = DREAMPLACE_STD_NAMESPACE::min(bin_index_yh, num_bins_y);

  for (int k = bin_index_xl; k < bin_index_xh; ++k) {
    T bin_xl = xl + bin_size_x * k;
    T bin_xh = DREAMPLACE_STD_NAMESPACE::min(bin_xl + bin_size_x, xh);
    // special treatment for rightmost bins
    if (k + 1 == num_bins_x) {
      bin_xh = bxh;
    }
    T px = computeDensityFunc(bxl, bxh, bin_xl, bin_xh);
    for (int h = bin_index_yl; h < bin_index_yh; ++h) {
      T bin_yl = yl + bin_size_y * h;
      T bin_yh = DREAMPLACE_STD_NAMESPACE::min(bin_yl + bin_size_y, yh);
      // special treatment for upmost bins
      if (h + 1 == num_bins_y) {
        bin_yh = byh;
      }
      T py = computeDensityFunc(byl, byh, bin_yl, bin_yh);

      // still area
      atomic_add_op(&buf_map[k * num_bins_y + h], px * py * multiplier);
      distributed_num_bins += 1;
      distributed_area += buf_map[k * num_bins_y + h];
    }
  }
  if (distributed_num_bins == 0) {
    return 0;
  }
  return distributed_area / distributed_num_bins;
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
/// @param buf_map_tier 2D density map in column-major to write
template <typename T, typename AtomicOp>
int computeDensityMapLauncher(const T* x_tensor, const T* y_tensor, const T* node_size_x_tensor,
                              const T* node_size_y_tensor, const int num_nodes,
                              const int num_bins_x, const int num_bins_y, const T xl, const T yl,
                              const T xh, const T yh, int num_threads, AtomicOp atomic_add_op,
                              typename AtomicOp::type* buf_map_tier, int* tier, int num_tiers) {
  // density_map_tensor should be initialized outside

  T bin_size_x = (xh - xl) / num_bins_x;
  T bin_size_y = (yh - yl) / num_bins_y;
  // double bin_area = bin_size_x * bin_size_y;
  int num_bins = static_cast<int>(num_bins_x * num_bins_y);

  // #pragma omp parallel for num_threads(num_threads)
  for (int i = 0; i < num_nodes; ++i) {
    T bxl = x_tensor[i];
    T byl = y_tensor[i];
    int tier_id = tier[i];
    if (tier_id < 0 || tier_id >= num_tiers) {
      continue;
    }
    T bxh = bxl + node_size_x_tensor[tier_id * num_nodes + i];
    T byh = byl + node_size_y_tensor[tier_id * num_nodes + i];
    distributeBox2Bin(static_cast<int>(num_bins_x), static_cast<int>(num_bins_y), xl, yl, xh, yh,
                      bin_size_x, bin_size_y, bxl, byl, bxh, byh, atomic_add_op,
                      buf_map_tier + tier_id * num_bins, 1);
  }
  return 0;
}

/**
 * @brief Update density map for a single node
 * @return average bin density after updating the node's position
 */
template <typename T, typename AtomicOp>
double updateDensityMapLauncher(const T* x_tensor, const T* y_tensor, const T* node_size_x_tensor,
                                const T* node_size_y_tensor, const int num_nodes,
                                const int num_bins_x, const int num_bins_y, const T xl, const T yl,
                                const T xh, const T yh, int num_threads, AtomicOp atomic_add_op,
                                typename AtomicOp::type* buf_map_tier, int from_tier, int to_tier,
                                int node_id) {

  // Calculate bin sizes
  T bin_size_x = (xh - xl) / num_bins_x;
  T bin_size_y = (yh - yl) / num_bins_y;
  double bin_area = bin_size_x * bin_size_y;
  int num_bins = static_cast<int>(num_bins_x * num_bins_y);

  // Get node position and sizes for both tiers
  if (node_id < 0 || node_id >= num_nodes || from_tier < 0 || to_tier < 0) {
    return 0;
  }
  T bxl = x_tensor[node_id];
  T byl = y_tensor[node_id];

  // Calculate node dimensions for both tiers
  T bxh_from = bxl + node_size_x_tensor[from_tier * num_nodes + node_id];
  T byh_from = byl + node_size_y_tensor[from_tier * num_nodes + node_id];
  T bxh_to = bxl + node_size_x_tensor[to_tier * num_nodes + node_id];
  T byh_to = byl + node_size_y_tensor[to_tier * num_nodes + node_id];

  // Remove density contribution from the old tier using subtraction
  distributeBox2Bin(static_cast<int>(num_bins_x), static_cast<int>(num_bins_y), xl, yl, xh, yh,
                    bin_size_x, bin_size_y, bxl, byl, bxh_from, byh_from, atomic_add_op,
                    buf_map_tier + from_tier * num_bins, -1);

  // Add density contribution to the new tier using addition
  T average_area = distributeBox2Bin(static_cast<int>(num_bins_x), static_cast<int>(num_bins_y), xl,
                                     yl, xh, yh, bin_size_x, bin_size_y, bxl, byl, bxh_to, byh_to,
                                     atomic_add_op, buf_map_tier + to_tier * num_bins, 1);

  return average_area / bin_area;
}

/**
 * @brief Update density map for a single node
 * @return average bin density after updating the node's position
 */
template <typename T, typename AtomicOp>
double getTerminalDensityMapLauncher(const T* x_tensor, const T* y_tensor,
                                     const T* node_size_x_tensor, const T* node_size_y_tensor,
                                     const int num_nodes, const int num_bins_x,
                                     const int num_bins_y, const T xl, const T yl, const T xh,
                                     const T yh, int num_threads, AtomicOp atomic_add_op,
                                     typename AtomicOp::type* buf_map_terminal, T terminal_x_center,
                                     T terminal_y_center, int multiplier) {

  // Calculate bin sizes
  T bin_size_x = (xh - xl) / num_bins_x;
  T bin_size_y = (yh - yl) / num_bins_y;
  double bin_area = bin_size_x * bin_size_y;

  // Get node position and sizes for both tiers
  T bxl = terminal_x_center - node_size_x_tensor[0] / 2;
  T byl = terminal_y_center - node_size_y_tensor[0] / 2;

  // Calculate node dimensions for both tiers
  T bxh = terminal_x_center + node_size_x_tensor[0] / 2;
  T byh = terminal_y_center + node_size_y_tensor[0] / 2;

  // Add density contribution to the new tier using addition
  T average_area = distributeBox2Bin(static_cast<int>(num_bins_x), static_cast<int>(num_bins_y), xl,
                                     yl, xh, yh, bin_size_x, bin_size_y, bxl, byl, bxh, byh,
                                     atomic_add_op, buf_map_terminal, multiplier);

  return average_area / bin_area;
}

PLACER_END_NAMESPACE
