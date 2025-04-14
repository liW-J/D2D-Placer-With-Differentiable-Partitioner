'''
Author: JeanneWillis hi@jeannewillis.cn
Date: 2025-04-14 03:21:27
LastEditors: JeanneWillis hi@jeannewillis.cn
LastEditTime: 2025-04-14 04:52:06
FilePath: /D2D-placer/placer/tools/PosFlattened.py
Description: 
'''
import time
import logging
import torch


class PosFlattened:

    def __init__(self, params, placedb, placedb_tier):
        self.params = params
        self.placedb = placedb
        self.placedb_tier = placedb_tier

    def pos_flattened(self, tier, pos_2d, pos_tier):
        """
            @brief 
            @param 
            """
        num_tiers = self.params.num_tiers
        # net_mask = torch.zeros(self.placedb.num_nets)

        tt = time.time()
        for i in range(num_tiers):

            node_x = pos_tier[i][:self.placedb_tier[i].num_movable_nodes]
            node_y = pos_tier[i][len(pos_tier[i]) // 2:len(pos_tier[i]) // 2 +
                                 self.placedb_tier[i].num_movable_nodes]

            pos_2d[:self.placedb.num_movable_nodes][tier == i].data.copy_(
                node_x)
            pos_2d[len(pos_2d) // 2:len(pos_2d) // 2 +
                   self.placedb.num_movable_nodes][tier == i].data.copy_(
                       node_y)

        logging.info("pos_flattened takes %.3f seconds" % (time.time() - tt))

        return pos_2d
