'''
Author: JeanneWillis hi@jeannewillis.cn
Date: 2025-04-14 03:21:27
LastEditors: JeanneWillis hi@jeannewillis.cn
LastEditTime: 2025-05-08 13:54:34
FilePath: /D2D-placer/placer/tools/PosFlattened.py
Description: 
'''
import time
import logging
import torch
import re


class PosFlattened:

    def __init__(self, params, placedb, placedb_tier):
        self.params = params
        self.placedb = placedb
        self.placedb_tier = placedb_tier
        
    def sort_node(self, node_names):
        """
        @brief 
        @param 
        """
        # create a boolean mask, identify the nodes start with 'C'
        c_mask = torch.tensor([name.startswith(b'C') for name in node_names])
        # get the index of the nodes that satisfy the condition
        c_indices = torch.nonzero(c_mask, as_tuple=True)[0]

        c_numbers = []
        for idx in c_indices:
            name = node_names[idx].decode('utf-8')  
            # extract the number after C
            match = re.match(r'C(\d+)', name)
            if match:
                c_numbers.append(int(match.group(1)))
            else:
                c_numbers.append(0) 
        
        # convert the extracted numbers to a PyTorch tensor
        c_numbers = torch.tensor(c_numbers, dtype=torch.int64)
        
        # index start from 0
        return c_numbers-1


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
            c_numbers = self.sort_node(self.placedb_tier[i].node_names)
            pos_2d[:self.placedb.num_movable_nodes][c_numbers] = node_x
            pos_2d[len(pos_2d) // 2:len(pos_2d) // 2 +
                   self.placedb.num_movable_nodes][c_numbers] = node_y

        logging.info("pos_flattened takes %.3f seconds" % (time.time() - tt))

        return pos_2d
