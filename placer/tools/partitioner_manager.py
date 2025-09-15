'''
Author: JeanneWillis hi@jeannewillis.cn
Date: 2025-09-15 19:31:41
LastEditors: JeanneWillis hi@jeannewillis.cn
LastEditTime: 2025-09-15 20:45:07
FilePath: /D2D-placer/placer/tools/partitioner_manager.py
Description: 
generate partitioner file from placer data structure
use torch tensor as input
'''

import torch
import numpy as np
import os

from placer.configure import compile_configurations


class PartitionerManager:

    def __init__(self, data_collections_2d, placedb_2d, output_dir):
        self.data_collections_2d = data_collections_2d
        self.placedb_2d = placedb_2d
        self.output_dir = output_dir

        self.flat_net2pin_map = data_collections_2d.flat_net2pin_map
        self.flat_net2pin_start_map = data_collections_2d.flat_net2pin_start_map
        self.pin2node_map = data_collections_2d.pin2node_map
        self.num_movable_nodes = placedb_2d.num_movable_nodes

    def hgr_generator(self, hgr_name):
        """
      generate hgr file from placer data structure
      
      Args:
          case_name: case name (str)
      
      Returns:
          str: generated hgr file path
      """

        output_file = f"{self.output_dir}/{hgr_name}.hgr"

        # convert torch tensor to numpy array
        if isinstance(self.flat_net2pin_map, torch.Tensor):
            flat_net2pin_map = self.flat_net2pin_map.cpu().numpy()
        if isinstance(self.flat_net2pin_start_map, torch.Tensor):
            flat_net2pin_start_map = self.flat_net2pin_start_map.cpu().numpy()
        if isinstance(self.pin2node_map, torch.Tensor):
            pin2node_map = self.pin2node_map.cpu().numpy()

        # calculate net number
        num_nets = len(flat_net2pin_start_map) - 1

        # create output directory
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        # generate hgr file
        with open(output_file, 'w') as f:
            # first line: net number and node number
            f.write(f"{num_nets} {self.num_movable_nodes}\n")

            # write nodes for each net
            for net_id in range(num_nets):
                # get all pins for this net
                start_idx = flat_net2pin_start_map[net_id]
                end_idx = flat_net2pin_start_map[net_id + 1]

                # collect all unique nodes for this net
                net_nodes = set()
                for pin_id in range(start_idx, end_idx):
                    node_id = pin2node_map[flat_net2pin_map[pin_id]]
                    if node_id < self.num_movable_nodes:  # only include movable nodes
                        net_nodes.add(node_id)

                # convert node index to 1-based and write to file
                if net_nodes:  # only write non-empty net
                    node_list = sorted(list(net_nodes))
                    node_str = " ".join(
                        [str(node_id + 1) for node_id in node_list])
                    f.write(f"{node_str}\n")
                else:
                    f.write("\n")  # empty net also write empty line

        print(f"HGR file generated: {output_file}")
        print(f"Net number: {num_nets}, Node number: {self.num_movable_nodes}")
        
    def parts_reader(self, parts_name):
        """
        generate parts file from placer data structure
        """
        parts_file = f"{self.output_dir}/{parts_name}.hgr.part.2"
        
        tier = torch.zeros(self.num_movable_nodes, dtype=torch.int32)
        with open(parts_file, 'r') as f:
            for i in range(self.num_movable_nodes):
                tier[i] = int(f.readline().strip())
                
        print(f"Parts file generated: {parts_file}")
        print(f"Node number: {self.num_movable_nodes}")
        
        return tier
