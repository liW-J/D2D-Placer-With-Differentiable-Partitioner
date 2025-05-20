/*
 * @Author: JeanneWillis hi@jeannewillis.cn
 * @Date: 2025-04-08 12:35:48
 * @LastEditors: JeanneWillis hi@jeannewillis.cn
 * @LastEditTime: 2025-05-18 22:18:51
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
int hmetisPartitionLauncher(T *tier, const int *flat_netpin, const int *netpin_start, const int *pin2node_map,
                            const unsigned char *net_mask, int num_nets, int num_movable_nodes, int num_threads)
{

  // write hgr file
  HGR hgr("./run_tmp/case2_hidden/", "circuit");

  for (int net_id = 0; net_id < num_nets; ++net_id)
  {
    hgr.add_net(to_string(net_id));
    if (net_mask[net_id])
    {
      for (int pin_id = netpin_start[net_id]; pin_id < netpin_start[net_id + 1]; ++pin_id)
      {
        // LOG(DEBUG, "net %d, pin %d, node %d", net_id, pin_id, pin2node_map[flat_netpin[pin_id]]);
        hgr.add_node(to_string(net_id), to_string(pin2node_map[flat_netpin[pin_id]]));
      }
    }
  }
  hgr.write_hgr();

  // run hmetis
  string cmd = "bin/hmetis -ufactor=0.7 ./run_tmp/case2_hidden/circuit.hgr 2 > ./run_tmp/case2_hidden/circuit-hmetis.log";
  system(cmd.c_str());
  hgr.read_part_result(2);

  assert(hgr.get_part_size(0) + hgr.get_part_size(1) == num_movable_nodes);
  LOG(INFO, "hmetis partition result: %d : %d", hgr.get_part_size(0), hgr.get_part_size(1));

  // partiton result save to ./run_tmp/case2_hidden/circuit.part.2
  for (int i = 0; i < num_movable_nodes; ++i)
  {
    tier[i] = hgr.get_part_result(to_string(i));
  }

  LOG(INFO, "Running hmetis completed");
  return 0;
}


// template <typename T>
// int binBasedPartitionLauncher(T *tier, const int *flat_netpin, const int *netpin_start, const int *pin2node_map,
//                             const unsigned char *net_mask, int num_nets, int num_movable_nodes, int num_threads)
// {
//   double cutline = 0.5;
//   double width_avg0 = 0;
//   double width_avg1 = 0;
//   int bins_per_row = 1;
//   int bins_per_col = 1;
//   bool _multiLevel = false;
//   int bin_width = 1000 / bins_per_row;
//   int bin_height = 1000 / bins_per_col;
//   int bin_num = bins_per_row * bins_per_col;
//   vector <vector <vector <Cell_C*>>> bins(bins_per_row, vector< vector <Cell_C*>> (bins_per_col, vector <Cell_C*> ()));   


//   double used_area[2] = {0.0, 0.0};
//   double maxArea[2];
//   double totalArea[2];
//   for (int ind=0; ind<bin_num; ind++) {
//     int i = bins_size[ind].second.first;
//     int j = bins_size[ind].second.second;
//     Partitioner* partitioner = new Partitioner();
//     totalArea[0] = (double) _pChip->get_die(0)->get_width() * (double) _pChip->get_die(0)->get_height() * _pChip->get_die(0)->get_max_util();
//     totalArea[1] = (double) _pChip->get_die(1)->get_width() * (double) _pChip->get_die(1)->get_height() * _pChip->get_die(1)->get_max_util();
//     maxArea[0] = totalArea[0] - used_area[0];
//     maxArea[1] = totalArea[1] - used_area[1];
//     partitioner->parseInput(_vCell, _pChip, bins[i][j], maxArea, cutline, false);
//     partitioner->initial_partition();
//     partitioner->partition(2,2,3, true);
//     vector<vector<int> >& cellPart = partitioner->get_part_result();
//     for (int k=0; k<2; ++k){
//       for(int cellId : cellPart[k]){ 
//         Cell_C* cell = _vCell[cellId];
//         used_area[k] += cell->get_width(_pChip->get_die(k)->get_techId()) * cell->get_height(_pChip->get_die(k)->get_techId());
//         if (used_area[k] <= totalArea[k]) {
//           tier[cellId] = k;
//         } else {
//           used_area[k] -= cell->get_width(_pChip->get_die(k)->get_techId()) * cell->get_height(_pChip->get_die(k)->get_techId());
//           tier[cellId] = 1 - k;
//           used_area[1 - k] += cell->get_width() * cell->get_height();
//         }
//       }
//     }
//   }

//   return 0;
// }

at::Tensor hmetis_forward(at::Tensor pos, at::Tensor flat_netpin, at::Tensor netpin_start,
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

  CHECK_FLAT_CPU(pos);
  CHECK_EVEN(pos);
  CHECK_CONTIGUOUS(pos);

  at::Tensor tier = at::zeros(num_movable_nodes, pos.options());
  int num_nets = netpin_start.numel() - 1;

  DREAMPLACE_DISPATCH_FLOATING_TYPES(pos, "hmetisPartitionLauncher", [&]
                                     { hmetisPartitionLauncher<scalar_t>(
                                           DREAMPLACE_TENSOR_DATA_PTR(tier, scalar_t),
                                           DREAMPLACE_TENSOR_DATA_PTR(flat_netpin, int),
                                           DREAMPLACE_TENSOR_DATA_PTR(netpin_start, int),
                                           DREAMPLACE_TENSOR_DATA_PTR(pin2node_map, int),
                                           DREAMPLACE_TENSOR_DATA_PTR(net_mask, unsigned char),
                                           num_nets,
                                           num_movable_nodes,
                                           at::get_num_threads()); });
  return tier;
}

PLACER_END_NAMESPACE

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
  m.def("hmetis", &PLACER_NAMESPACE::hmetis_forward, "hmetis_forward");
}
