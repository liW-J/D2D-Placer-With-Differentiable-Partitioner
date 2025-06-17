/*
 * @Author: JeanneWillis hi@jeannewillis.cn
 * @Date: 2025-05-26 14:47:09
 * @LastEditors: JeanneWillis hi@jeannewillis.cn
 * @LastEditTime: 2025-06-16 18:00:12
 * @FilePath: /D2D-placer/placer/ops/utils_3d/src/partition.h
 * @Description:
 */

#ifndef UTILS_3D_PARTITIONER_H
#define UTILS_3D_PARTITIONER_H

// dreamplace
#include "utility/src/torch.h"
#include "utility/src/utils.h"
// 3d-placer parser
#include "include/common.h"
#include "placer/placer.h"

PLACER_BEGIN_NAMESPACE

struct NetStatus {
  int is_cut_net;
  int tier_id;

  NetStatus(int is_cut_net = 0, int tier_id = 0)
      : is_cut_net(is_cut_net), tier_id(tier_id) {}
};

struct Partitioner {

  static inline NetStatus check_net_cut(int *tier, const int *flat_netpin,
                                        const int *netpin_start,
                                        const int *pin2node_map,
                                        int num_movable_nodes, int num_nets,
                                        int num_tiers, int net_id) {

    int is_cut_net = 0;
    int num_nodes_in_net = 0;
    int tier_id = -1;
    vector<int> node_count(num_tiers, 0);

    // count nodes in each tier for check if net is cut
    for (int tier_id = 0; tier_id < num_tiers; ++tier_id) {
      for (int pin_id = netpin_start[net_id]; pin_id < netpin_start[net_id + 1];
           ++pin_id) {
        int node_id = pin2node_map[flat_netpin[pin_id]];
        if (tier[node_id] == tier_id) {
          int index_node = num_movable_nodes * tier_id + node_id;
          node_count[tier_id]++;
          num_nodes_in_net++;
        }
      }
    }

    // if all nodes are in the same tier, not cut
    if (std::find(node_count.begin(), node_count.end(), num_nodes_in_net) ==
        node_count.end()) {
      is_cut_net = 1;
    } else {
      tier_id =
          std::find(node_count.begin(), node_count.end(), num_nodes_in_net) -
          node_count.begin();
    }
    return NetStatus(is_cut_net, tier_id);
  }

  static inline std::vector<int>
  countRelatedNodesInCutNets(const int *tier, const int *flat_netpin,
                             const int *netpin_start, const int *pin2node_map,
                             int num_movable_nodes, int num_nets, int num_tiers,
                             int *cut_net_mask) {

    std::vector<int> nodes_per_tier(num_tiers, 0);

    for (int net_id = 0; net_id < num_nets; ++net_id) {
      if (cut_net_mask[net_id]) {
        for (int pin_id = netpin_start[net_id];
             pin_id < netpin_start[net_id + 1]; ++pin_id) {
          int node_id = pin2node_map[flat_netpin[pin_id]];

          int node_tier = tier[node_id];
          nodes_per_tier[node_tier]++;
        }
      }
    }
    for (int i = 0; i < num_tiers; ++i) {
      LOG(INFO, "nodes_per_tier[%d] = %d", i, nodes_per_tier[i]);
    }

    return nodes_per_tier;
  }

  // static inline std::vector<int> countConnectedNodesInRelatedNets(
  //     int cur_node_id, const int *tier, const int *flat_netpin,
  //     const int *netpin_start, const int *pin2node_map, const int
  //     *flat_nodepin, const int *nodepin_start, const int *pin2net_map, int
  //     num_movable_nodes, int num_nets, int num_tiers, int *cut_net_mask) {

  //   std::vector<int> from_nodes_per_net(num_nets, 0);
  //   std::vector<int> to_nodes_per_net(num_nets, 0);

  //   int cur_node_tier = tier[cur_node_id];
  //   for (int node2pin_id = nodepin_start[cur_node_id];
  //        node2pin_id < nodepin_start[cur_node_id + 1]; ++node2pin_id) {
  //     int net_id = pin2net_map[flat_nodepin[node2pin_id]];
  //     for (int net2pin_id = netpin_start[net_id];
  //          net2pin_id < netpin_start[net_id + 1]; ++net2pin_id) {
  //       int node_id = pin2node_map[flat_netpin[net2pin_id]];
  //       if (tier[node_id] == cur_node_tier) {
  //         from_nodes_per_net[net_id]++;
  //       } else {
  //         to_nodes_per_net[net_id]++;
  //       }
  //     }
  //   }

  //   return from_nodes_per_net;
  // }

  // static inline std::vector<int>
  // dieUtilization(int cur_node_id, const int *tier, const int *flat_netpin,
  //                const int *netpin_start, const int *pin2node_map,
  //                const int *flat_nodepin, const int *nodepin_start,
  //                const int *pin2net_map, int num_movable_nodes, int num_nets,
  //                int num_tiers, int *cut_net_mask) {}

  static inline void
  getCutNetMask(int *cut_net_mask, int num_nets, int num_tiers, const int *tier,
                const int *flat_netpin, const int *netpin_start,
                const int *pin2node_map, int num_movable_nodes) {

    for (int net_id = 0; net_id < num_nets; ++net_id) {
      int num_nodes_in_net = 0;
      vector<int> node_count(num_tiers, 0);

      // count nodes in each tier for check if net is cut
      for (int tier_id = 0; tier_id < num_tiers; ++tier_id) {
        for (int pin_id = netpin_start[net_id];
             pin_id < netpin_start[net_id + 1]; ++pin_id) {
          int node_id = pin2node_map[flat_netpin[pin_id]];
          if (tier[node_id] == tier_id) {
            int index_node = num_movable_nodes * tier_id + node_id;
            node_count[tier_id]++;
            num_nodes_in_net++;
          }
        }
      }

      // if all nodes are in the same tier, not cut
      if (std::find(node_count.begin(), node_count.end(), num_nodes_in_net) ==
          node_count.end()) {
        cut_net_mask[net_id] = 1;
      }
    }
  }
};

PLACER_END_NAMESPACE

#endif
