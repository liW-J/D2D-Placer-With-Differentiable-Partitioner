'''
Author: JeanneWillis hi@jeannewillis.cn
Date: 2025-04-14 03:21:27
LastEditors: JeanneWillis hi@jeannewillis.cn
LastEditTime: 2025-09-14 21:37:03
FilePath: /D2D-placer/placer/tools/pos_flattened.py
Description: 
'''
import time
import logging
import torch


class PosFlattened:

    def __init__(self, params, data_2d, data_tier):
        self.params = params
        self.data_2d = data_2d
        self.data_tier = data_tier

    def map_tier_to_2d(self, node_names):
        """
        @brief Map tier movable node names to flattened-2D placedb node ids.

        NOTE: do NOT assume node "Ck" has 2D node id k-1. dreamplace's
        PlaceDB reorders movable nodes internally (e.g. large movable
        macros are handled separately), so the only safe mapping is the
        2D placedb's node_name2id_map.
        """
        name2id = self.data_2d.placedb.node_name2id_map
        ids = []
        for name in node_names:
            key = name.decode('utf-8') if isinstance(name, bytes) else name
            idx = name2id.get(key)
            if idx is None:
                idx = name2id.get(name)
            if idx is None:
                raise KeyError(
                    "pos_flattened: tier node %s not found in flattened-2D "
                    "placedb" % key)
            ids.append(idx)
        return torch.tensor(ids, dtype=torch.int64)

    def __call__(self, tier, pos_2d, pos_tier):
        """
            @brief 
            @param 
            """
        num_tiers = self.params.num_tiers
        num_nodes_2d = len(pos_2d) // 2

        tt = time.time()
        for i in range(num_tiers):
            num_movable = self.data_tier[i].placedb.num_movable_nodes
            node_x = pos_tier[i][:num_movable]
            node_y = pos_tier[i][len(pos_tier[i]) // 2:len(pos_tier[i]) // 2 +
                                 num_movable]
            ids_2d = self.map_tier_to_2d(
                self.data_tier[i].placedb.node_names[:num_movable]).to(
                    pos_2d.device)
            pos_2d.data[ids_2d] = node_x
            pos_2d.data[num_nodes_2d + ids_2d] = node_y

        logging.info("pos_flattened takes %.3f seconds" % (time.time() - tt))

        return pos_2d
