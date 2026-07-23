'''
Date: 2025-09-15 19:31:41
LastEditTime: 2025-09-16 20:09:50
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

    def hgr_generator(self, hgr_name, bin_nodes=None, bin_node_count=None):
        """
        generate hgr file from placer data structure
        
        Args:
            hgr_name: case name (str)
            bin_nodes: torch.Tensor of shape (max_nodes_per_bin,) - node indices in the bin (optional)
            bin_node_count: int - number of nodes in the bin (optional)
        
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

        if bin_nodes is not None and bin_node_count is not None:
            # Bin-based hgr generation
            self.hgr_generator_bin(output_file, flat_net2pin_map,
                                   flat_net2pin_start_map, pin2node_map,
                                   num_nets, bin_nodes, bin_node_count)
        else:
            # Original hgr generation
            self.hgr_generator_original(output_file, flat_net2pin_map,
                                        flat_net2pin_start_map, pin2node_map,
                                        num_nets)

    def hgr_generator_original(self, output_file, flat_net2pin_map,
                               flat_net2pin_start_map, pin2node_map, num_nets):
        """original hgr generation method (no bin-based)"""
        # generate hgr file
        with open(output_file, 'w+') as f:
            # Reserve a fixed-width header and rewrite it after empty nets have
            # been filtered.  A fixed width avoids buffering millions of
            # hyperedges or making a second pass just to count them.
            f.write(f"{0:20d} {self.num_movable_nodes}\n")
            written_nets = 0

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
                    written_nets += 1

            f.seek(0)
            f.write(f"{written_nets:20d} {self.num_movable_nodes}\n")

        print(f"HGR file generated: {output_file}")
        print(
            f"Net number: {written_nets}/{num_nets}, "
            f"Node number: {self.num_movable_nodes}")

    def hgr_generator_bin(self, output_file, flat_net2pin_map,
                          flat_net2pin_start_map, pin2node_map, num_nets,
                          bin_nodes, bin_node_count):
        """
        generate hgr file for specific bin
        
        Args:
            output_file: output file path
            flat_net2pin_map: pin to net mapping
            flat_net2pin_start_map: start index of each net's pins
            pin2node_map: pin to node mapping
            num_nets: total number of nets
            bin_nodes: node indices in the bin
            bin_node_count: number of nodes in the bin
        """
        # create mapping from global node id to node id in the bin
        bin_node_to_local = {}
        for i in range(bin_node_count):
            global_node_id = bin_nodes[i].item()
            if global_node_id >= 0:  # valid node index
                bin_node_to_local[global_node_id] = i

        # count nets that involve nodes in the bin
        bin_nets = set()
        for net_id in range(num_nets):
            start_idx = flat_net2pin_start_map[net_id]
            end_idx = flat_net2pin_start_map[net_id + 1]

            # check if the net contains nodes in the bin
            for pin_id in range(start_idx, end_idx):
                node_id = pin2node_map[flat_net2pin_map[pin_id]]
                if node_id in bin_node_to_local:
                    bin_nets.add(net_id)
                    break

        # generate hgr file
        with open(output_file, 'w') as f:
            # first line: net number and node number
            f.write(f"{len(bin_nets)} {bin_node_count}\n")

            # write nodes for each net
            for net_id in sorted(bin_nets):
                start_idx = flat_net2pin_start_map[net_id]
                end_idx = flat_net2pin_start_map[net_id + 1]

                # collect all unique nodes for this net that are in the bin
                net_nodes = set()
                for pin_id in range(start_idx, end_idx):
                    node_id = pin2node_map[flat_net2pin_map[pin_id]]
                    if node_id in bin_node_to_local:  # only include nodes in this bin
                        net_nodes.add(
                            bin_node_to_local[node_id])  # use local node id

                # convert node index to 1-based and write to file
                if net_nodes:  # only write non-empty net
                    node_list = sorted(list(net_nodes))
                    node_str = " ".join(
                        [str(node_id + 1) for node_id in node_list])
                    f.write(f"{node_str}\n")
                else:
                    f.write("\n")  # empty net also write empty line

        print(f"Bin HGR file generated: {output_file}")
        print(
            f"Bin net number: {len(bin_nets)}, Bin node number: {bin_node_count}"
        )

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
