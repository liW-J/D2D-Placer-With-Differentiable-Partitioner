/*
 * @Date: 2025-04-08 12:35:48
 * @LastEditTime: 2025-06-16 16:50:24
 * @FilePath: /D2D-placer/placer/ops/hmetis/src/hmetis.cpp
 * @Description:
 */

#include <pybind11/pybind11.h>
// dreamplace
#include "utility/src/torch.h"
#include "utility/src/utils.h"
// 3d-placer parser
#include "include/common.h"
#include "placer/placer.h"

#include "utils_3d/src/partitioner.h"

PLACER_BEGIN_NAMESPACE

template <typename T>
int multiBipartitionLauncher(int *tier, const int *flat_netpin,
                             const int *netpin_start, const int *pin2node_map,
                             const unsigned char *net_mask, int num_nets,
                             int num_movable_nodes, int num_threads,
                             int *cut_net_mask, const int *flat_nodepin,
                             const int *nodepin_start, const int *pin2net_map) {

  // write hgr file
  HGR hgr("./run_tmp/case2_hidden/", "circuit");

  for (int net_id = 0; net_id < num_nets; ++net_id) {
    hgr.add_net(to_string(net_id));
    if (net_mask[net_id]) {
      for (int pin_id = netpin_start[net_id]; pin_id < netpin_start[net_id + 1];
           ++pin_id) {
        // LOG(DEBUG, "net %d, pin %d, node %d", net_id, pin_id,
        // pin2node_map[flat_netpin[pin_id]]);
        hgr.add_node(to_string(net_id),
                     to_string(pin2node_map[flat_netpin[pin_id]]));
      }
    }
  }
  hgr.write_hgr();

  // run hmetis
  string cmd = "bin/hmetis -ufactor=0.7 ./run_tmp/case2_hidden/circuit.hgr 2 > "
               "./run_tmp/case2_hidden/circuit-hmetis.log";
  system(cmd.c_str());
  hgr.read_part_result(2);

  assert(hgr.get_part_size(0) + hgr.get_part_size(1) == num_movable_nodes);
  LOG(INFO, "hmetis partition result: %d : %d", hgr.get_part_size(0),
      hgr.get_part_size(1));

  // partiton result save to ./run_tmp/case2_hidden/circuit.part.2
  for (int i = 0; i < num_movable_nodes; ++i) {
    tier[i] = hgr.get_part_result(to_string(i));
  }

  int num_tiers = 2;
  vector<vector<int>> tier_nets(num_tiers);
  vector<int> related_cut_tier_nets;
  for (int net_id = 0; net_id < num_nets; net_id++) {
    NetStatus net_status = Partitioner::check_net_cut(
        tier, flat_netpin, netpin_start, pin2node_map, num_movable_nodes,
        num_nets, num_tiers, net_id);

    if (!net_status.is_cut_net) {
      tier_nets[net_status.tier_id].push_back(net_id);
    }
  }

  for (int i = 0; i < num_tiers; i++) {
    LOG(INFO, "tier_nets[%d] = %d", i, tier_nets[i].size());

    // write hgr file
    HGR hgr("./run_tmp/case2_hidden/", "circuit-" + to_string(i));

    for (int net_id : tier_nets[i]) {
      hgr.add_net(to_string(net_id));
      if (net_mask[net_id]) {
        for (int pin_id = netpin_start[net_id];
             pin_id < netpin_start[net_id + 1]; ++pin_id) {
          hgr.add_node(to_string(net_id),
                       to_string(pin2node_map[flat_netpin[pin_id]]));
        }
      }
    }
    hgr.write_hgr();

    // run hmetis
    string cmd = "bin/hmetis -ufactor=0.7 ./run_tmp/case2_hidden/circuit-" +
                 to_string(i) + ".hgr 2 > ./run_tmp/case2_hidden/circuit-" +
                 to_string(i) + "-hmetis.log";
    system(cmd.c_str());
    hgr.read_part_result(2);

    assert(hgr.get_part_size(0) + hgr.get_part_size(1) == num_movable_nodes);
    LOG(INFO, "hmetis partition result: %d : %d", hgr.get_part_size(0),
        hgr.get_part_size(1));

    // partiton result save to ./run_tmp/case2_hidden/circuit.part.2
    for (int net_id : tier_nets[i]) {
      for (int pin_id = netpin_start[net_id]; pin_id < netpin_start[net_id + 1];
           ++pin_id) {
        int node_id = pin2node_map[flat_netpin[pin_id]];
        tier[node_id] = hgr.get_part_result(to_string(node_id));
      }
    }

    vector<vector<int>> tier_nets_2(num_tiers);
    vector<int> related_cut_tier_nets_2;
    for (int net_id : tier_nets[i]) {
      NetStatus net_status = Partitioner::check_net_cut(
          tier, flat_netpin, netpin_start, pin2node_map, num_movable_nodes,
          num_nets, num_tiers, net_id);
      cut_net_mask[net_id] = net_status.is_cut_net;
      if (!net_status.is_cut_net) {
        tier_nets_2[net_status.tier_id].push_back(net_id);
      }
    }

    for (int j = 0; j < num_tiers; j++) {
      LOG(INFO, "tier_nets[%d] = %d", j, tier_nets_2[j].size());

      // write hgr file
      HGR hgr("./run_tmp/case2_hidden/",
              "circuit-" + to_string(i) + "-" + to_string(j));

      for (int net_id : tier_nets_2[j]) {
        hgr.add_net(to_string(net_id));
        if (net_mask[net_id]) {
          for (int pin_id = netpin_start[net_id];
               pin_id < netpin_start[net_id + 1]; ++pin_id) {
            hgr.add_node(to_string(net_id),
                         to_string(pin2node_map[flat_netpin[pin_id]]));
          }
        }
      }
      hgr.write_hgr();

      // run hmetis
      string cmd = "bin/hmetis -ufactor=0.7 ./run_tmp/case2_hidden/circuit-" +
                   to_string(i) + "-" + to_string(j) + ".hgr 2 > " +
                   "./run_tmp/case2_hidden/circuit-" + to_string(i) + "-" +
                   to_string(j) + "-hmetis.log";
      system(cmd.c_str());
      hgr.read_part_result(2);

      assert(hgr.get_part_size(0) + hgr.get_part_size(1) == num_movable_nodes);
      LOG(INFO, "hmetis partition result: %d : %d", hgr.get_part_size(0),
          hgr.get_part_size(1));

      // partiton result save to ./run_tmp/case2_hidden/circuit.part.2
      for (int net_id : tier_nets_2[j]) {
        for (int pin_id = netpin_start[net_id];
             pin_id < netpin_start[net_id + 1]; ++pin_id) {
          int node_id = pin2node_map[flat_netpin[pin_id]];
          tier[node_id] = hgr.get_part_result(to_string(node_id));
        }
      }

      vector<vector<int>> tier_nets_3(num_tiers);
      vector<int> related_cut_tier_nets_3;
      for (int net_id : tier_nets_2[j]) {
        NetStatus net_status = Partitioner::check_net_cut(
            tier, flat_netpin, netpin_start, pin2node_map, num_movable_nodes,
            num_nets, num_tiers, net_id);
        cut_net_mask[net_id] = net_status.is_cut_net;
        if (!net_status.is_cut_net) {
          tier_nets_3[net_status.tier_id].push_back(net_id);
        }
      }

      for (int k = 0; k < num_tiers; k++) {
        LOG(INFO, "tier_nets[%d] = %d", k, tier_nets_3[k].size());

        // write hgr file
        HGR hgr("./run_tmp/case2_hidden/", "circuit-" + to_string(i) + "-" +
                                               to_string(j) + "-" +
                                               to_string(k));

        for (int net_id : tier_nets_3[k]) {
          hgr.add_net(to_string(net_id));
          if (net_mask[net_id]) {
            for (int pin_id = netpin_start[net_id];
                 pin_id < netpin_start[net_id + 1]; ++pin_id) {
              hgr.add_node(to_string(net_id),
                           to_string(pin2node_map[flat_netpin[pin_id]]));
            }
          }
        }
        hgr.write_hgr();

        // run hmetis
        string cmd = "bin/hmetis -ufactor=0.7 ./run_tmp/case2_hidden/circuit-" +
                     to_string(i) + "-" + to_string(j) + "-" + to_string(k) +
                     ".hgr 2 > " + "./run_tmp/case2_hidden/circuit-" +
                     to_string(i) + "-" + to_string(j) + "-" + to_string(k) +
                     "-hmetis.log";
        system(cmd.c_str());
        hgr.read_part_result(2);

        assert(hgr.get_part_size(0) + hgr.get_part_size(1) ==
               num_movable_nodes);
        LOG(INFO, "hmetis partition result: %d : %d", hgr.get_part_size(0),
            hgr.get_part_size(1));

        // partiton result save to ./run_tmp/case2_hidden/circuit.part.2
        for (int net_id : tier_nets_3[k]) {
          for (int pin_id = netpin_start[net_id];
               pin_id < netpin_start[net_id + 1]; ++pin_id) {
            int node_id = pin2node_map[flat_netpin[pin_id]];
            tier[node_id] = hgr.get_part_result(to_string(node_id));
          }
        }

        vector<vector<int>> tier_nets_4(num_tiers);
        vector<int> related_cut_tier_nets_4;
        for (int net_id : tier_nets_3[k]) {
          NetStatus net_status = Partitioner::check_net_cut(
              tier, flat_netpin, netpin_start, pin2node_map, num_movable_nodes,
              num_nets, num_tiers, net_id);
          cut_net_mask[net_id] = net_status.is_cut_net;
          // if (net_status.is_cut_net) {
          //   for (int pin_id = netpin_start[net_id];
          //        pin_id < netpin_start[net_id + 1]; ++pin_id) {
          //     int node_id = pin2node_map[flat_netpin[pin_id]];
          //     for (int all_pin_id = nodepin_start[node_id];
          //          all_pin_id < nodepin_start[node_id + 1]; ++all_pin_id) {
          //       int all_net_id = pin2net_map[flat_nodepin[all_pin_id]];
          //       related_cut_tier_nets_4.push_back(all_net_id);
          //     }
          //   }
          // }
          if (!net_status.is_cut_net) {
            tier_nets_4[net_status.tier_id].push_back(net_id);
          }
        }
      }
    }
  }

  Partitioner::countRelatedNodesInCutNets(tier, flat_netpin, netpin_start,
                                          pin2node_map, num_movable_nodes,
                                          num_nets, num_tiers, cut_net_mask);
  return 0;
}

at::Tensor
multi_bipartition_forward(at::Tensor pos, at::Tensor flat_netpin,
                          at::Tensor netpin_start, at::Tensor pin2node_map,
                          at::Tensor net_weights, at::Tensor net_mask,
                          int num_movable_nodes, at::Tensor flat_nodepin,
                          at::Tensor nodepin_start, at::Tensor pin2net_map) {
  CHECK_FLAT_CPU(flat_netpin);
  CHECK_CONTIGUOUS(flat_netpin);
  CHECK_FLAT_CPU(netpin_start);
  CHECK_CONTIGUOUS(netpin_start);
  CHECK_FLAT_CPU(pin2node_map);
  CHECK_CONTIGUOUS(pin2node_map);
  CHECK_FLAT_CPU(flat_nodepin);
  CHECK_CONTIGUOUS(flat_nodepin);
  CHECK_FLAT_CPU(nodepin_start);
  CHECK_CONTIGUOUS(nodepin_start);
  CHECK_FLAT_CPU(pin2net_map);
  CHECK_CONTIGUOUS(pin2net_map);

  CHECK_FLAT_CPU(net_weights);
  CHECK_CONTIGUOUS(net_weights);
  CHECK_FLAT_CPU(net_mask);
  CHECK_CONTIGUOUS(net_mask);

  CHECK_FLAT_CPU(pos);
  CHECK_EVEN(pos);
  CHECK_CONTIGUOUS(pos);

  at::Tensor tier = at::zeros(num_movable_nodes, pos.options()).to(at::kInt);

  int num_nets = netpin_start.numel() - 1;
  at::Tensor cut_net_mask = at::zeros(num_nets, pos.options()).to(at::kInt);

  DREAMPLACE_DISPATCH_FLOATING_TYPES(pos, "multiBipartitionLauncher", [&] {
    multiBipartitionLauncher<scalar_t>(
        DREAMPLACE_TENSOR_DATA_PTR(tier, int),
        DREAMPLACE_TENSOR_DATA_PTR(flat_netpin, int),
        DREAMPLACE_TENSOR_DATA_PTR(netpin_start, int),
        DREAMPLACE_TENSOR_DATA_PTR(pin2node_map, int),
        DREAMPLACE_TENSOR_DATA_PTR(net_mask, unsigned char), num_nets,
        num_movable_nodes, at::get_num_threads(),
        DREAMPLACE_TENSOR_DATA_PTR(cut_net_mask, int),
        DREAMPLACE_TENSOR_DATA_PTR(flat_nodepin, int),
        DREAMPLACE_TENSOR_DATA_PTR(nodepin_start, int),
        DREAMPLACE_TENSOR_DATA_PTR(pin2net_map, int));
  });
  return tier;
}

PLACER_END_NAMESPACE

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("multi_bipartition", &PLACER_NAMESPACE::multi_bipartition_forward,
        "multi_bipartition_forward");
}