'''
Author: JeanneWillis hi@jeannewillis.cn
Date: 2025-09-10 15:19:57
LastEditors: JeanneWillis hi@jeannewillis.cn
LastEditTime: 2026-03-04 22:07:03
FilePath: /D2D-placer/placer/tools/thirdparty_api/tritonpart_base.py
Description: 
'''
from placer.configure import compile_configurations
import logging
import os
import torch
import numpy as np

logger = logging.getLogger(__name__)


class TritonPartBase:

    def __init__(self, params, bin_size=(4, 4), enable_bin_based=True):
        super().__init__()

        self.params = params
        self.bin_size = bin_size  # (bin_x, bin_y)
        self.enable_bin_based = enable_bin_based
        self.tritonpart_tcl_path = compile_configurations[
            "PLACER_INSTALL_DIR"] + "/scripts/tritonpart.tcl"

        self.placement_file = f"{params.run_tmp_dir_root}/{params.case_name}.flattened-2d.embedding.dat"
        self.pl_file = f"{params.result_dir_root}/flattened-2d/flattened-2d.ntup.pl"

        os.environ["TRITONPART_CASE_NAME"] = params.case_name
        os.environ["PLACER_RUNTMP_DIR"] = params.run_tmp_dir_root

    def convert_gp_pl_to_embedding(self):
        """
        convert .gp.pl to .embedding.dat
        
        Args:
            output_file: output .embedding.dat file path
        """

        # check if input file exists
        if not os.path.exists(self.pl_file):
            print(f"error: input file {self.pl_file} does not exist")
            return False

        # determine output file path
        output_file = self.placement_file
        coordinates = []

        # read and parse .gp.pl file
        try:
            with open(self.pl_file, 'r') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()

                    # skip empty lines and header lines
                    if not line or line.startswith('UCLA'):
                        continue

                    # parse format: node name x coordinate y coordinate : FS
                    parts = line.split()
                    if len(parts) >= 3:
                        try:
                            x = float(parts[1])
                            y = float(parts[2])
                            coordinates.append((x, y))
                        except ValueError:
                            print(
                                f"warning: failed to parse line {line_num}: {line}"
                            )
                            continue
                    else:
                        print(
                            f"warning: line {line_num} format is incorrect: {line}"
                        )
                        continue
        except Exception as e:
            print(f"error: failed to read file: {e}")
            return False

        if not coordinates:
            print("error: no valid coordinates found")
            return False

        # find coordinate range
        x_coords = [coord[0] for coord in coordinates]
        y_coords = [coord[1] for coord in coordinates]

        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)

        # normalize coordinates to [0, 1] range
        normalized_coords = []
        for x, y in coordinates:
            norm_x = (x - x_min) / (x_max - x_min) if x_max > x_min else 0.0
            norm_y = (y - y_min) / (y_max - y_min) if y_max > y_min else 0.0
            normalized_coords.append((norm_x, norm_y))

        # write to output file
        with open(output_file, 'w') as f:
            for x, y in normalized_coords:
                f.write(f"{x:.15e}, {y:.15e}\n")

        return True

    def assign_nodes_to_bins(self, pos, num_movable_nodes):
        """
        assign nodes to bins based on node positions
        
        Args:
            pos: position tensor, format: pos.x = pos[:num_movable_nodes], pos.y = pos[pos.size(0)/2:pos.size(0)/2+num_movable_nodes]
            num_movable_nodes: number of movable nodes
            
        Returns:
            bin_nodes: torch.Tensor of shape (bin_x, bin_y, max_nodes_per_bin) store node indices in each bin
            bin_node_counts: torch.Tensor of shape (bin_x, bin_y) store number of nodes in each bin
        """
        # extract x and y coordinates
        pos_x = pos[:num_movable_nodes]
        pos_y = pos[pos.size(0) // 2:pos.size(0) // 2 + num_movable_nodes]

        # calculate coordinate range
        x_min, x_max = pos_x.min().item(), pos_x.max().item()
        y_min, y_max = pos_y.min().item(), pos_y.max().item()

        # calculate bin size
        bin_width = (x_max -
                     x_min) / self.bin_size[0] if x_max > x_min else 1.0
        bin_height = (y_max -
                      y_min) / self.bin_size[1] if y_max > y_min else 1.0

        # calculate which bin each node belongs to
        bin_x_indices = torch.floor(
            (pos_x - x_min) / bin_width).clamp(0, self.bin_size[0] - 1).long()
        bin_y_indices = torch.floor(
            (pos_y - y_min) / bin_height).clamp(0,
                                                self.bin_size[1] - 1).long()

        # initialize bin storage structure
        max_nodes_per_bin = 0
        bin_node_counts = torch.zeros(self.bin_size, dtype=torch.long)

        # calculate number of nodes in each bin
        for i in range(num_movable_nodes):
            bx, by = bin_x_indices[i].item(), bin_y_indices[i].item()
            bin_node_counts[bx, by] += 1
            max_nodes_per_bin = max(max_nodes_per_bin,
                                    bin_node_counts[bx, by].item())

        # create bin_nodes tensor
        bin_nodes = torch.full(
            (self.bin_size[0], self.bin_size[1], max_nodes_per_bin),
            -1,
            dtype=torch.long)
        bin_node_counts_temp = torch.zeros(self.bin_size, dtype=torch.long)

        # assign nodes to corresponding bins
        for i in range(num_movable_nodes):
            bx, by = bin_x_indices[i].item(), bin_y_indices[i].item()
            idx = bin_node_counts_temp[bx, by].item()
            bin_nodes[bx, by, idx] = i
            bin_node_counts_temp[bx, by] += 1

        return bin_nodes, bin_node_counts

    def convert_gp_pl_to_embedding_bin(self, bin_x, bin_y, bin_nodes,
                                       bin_node_count):
        """
        convert .gp.pl to .embedding.dat for specific bin
        
        Args:
            bin_x, bin_y: bin coordinates
            bin_nodes: node indices in the bin
            bin_node_count: number of nodes in the bin
        """
        bin_name = f"bin_{bin_x}_{bin_y}"
        bin_placement_file = f"{self.params.run_tmp_dir_root}/{self.params.case_name}_{bin_name}.flattened-2d.embedding.dat"

        # check if input file exists
        if not os.path.exists(self.pl_file):
            print(f"error: input file {self.pl_file} does not exist")
            return False

        # read .pl file and extract coordinates of nodes in the bin
        coordinates = []
        node_positions = {}  # mapping of node_name to coordinates

        try:
            with open(self.pl_file, 'r') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()

                    # skip empty lines and header lines
                    if not line or line.startswith('UCLA'):
                        continue

                    # parse format: node name x coordinate y coordinate : FS
                    parts = line.split()
                    if len(parts) >= 3:
                        try:
                            node_name = parts[0]
                            x = float(parts[1])
                            y = float(parts[2])
                            node_positions[node_name] = (x, y)
                        except ValueError:
                            print(
                                f"warning: failed to parse line {line_num}: {line}"
                            )
                            continue
        except Exception as e:
            print(f"error: failed to read file: {e}")
            return False

        # extract coordinates of nodes in the bin
        # note: here we need to match the actual node naming rule
        # assume the node name format is node_<index>, we need to adjust according to the actual situation
        for i in range(bin_node_count):
            node_idx = bin_nodes[i].item()
            if node_idx >= 0:  # valid node index
                # here we need to get the node name according to the actual node naming rule
                # temporarily use node_idx as the coordinate index
                if node_idx < len(node_positions):
                    # assume the keys of node_positions are in order
                    node_names = list(node_positions.keys())
                    if node_idx < len(node_names):
                        node_name = node_names[node_idx]
                        if node_name in node_positions:
                            coordinates.append(node_positions[node_name])

        if not coordinates:
            print(
                f"warning: no valid coordinates found for bin {bin_x}_{bin_y}")
            return False

        # calculate coordinate range and normalize
        x_coords = [coord[0] for coord in coordinates]
        y_coords = [coord[1] for coord in coordinates]

        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)

        # normalize coordinates to [0, 1] range
        normalized_coords = []
        for x, y in coordinates:
            norm_x = (x - x_min) / (x_max - x_min) if x_max > x_min else 0.0
            norm_y = (y - y_min) / (y_max - y_min) if y_max > y_min else 0.0
            normalized_coords.append((norm_x, norm_y))

        # write to output file
        with open(bin_placement_file, 'w') as f:
            for x, y in normalized_coords:
                f.write(f"{x:.15e}, {y:.15e}\n")

        return True

    def partitioning_bin(self, bin_x, bin_y):
        """
        run tritonpart for specific bin
        
        Args:
            bin_x, bin_y: bin coordinates
        """
        bin_name = f"bin_{bin_x}_{bin_y}"

        # set environment variable
        os.environ[
            "TRITONPART_CASE_NAME"] = f"{self.params.case_name}_{bin_name}"

        exit_code = os.system(f"openroad {self.tritonpart_tcl_path}")
        if exit_code != 0:
            print(
                f"error: failed to run tritonpart.tcl for bin {bin_x}_{bin_y}")
            return False
        return True

    def merge_bin_results(self, bin_nodes, bin_node_counts, num_movable_nodes):
        """
        merge partition results of all bins
        
        Args:
            bin_nodes: torch.Tensor of shape (bin_x, bin_y, max_nodes_per_bin)
            bin_node_counts: torch.Tensor of shape (bin_x, bin_y)
            num_movable_nodes: total number of movable nodes
            
        Returns:
            tier: torch.Tensor of shape (num_movable_nodes,) 包含所有nodes的partition结果
        """
        tier = torch.zeros(num_movable_nodes, dtype=torch.int32)

        for bx in range(self.bin_size[0]):
            for by in range(self.bin_size[1]):
                bin_name = f"bin_{bx}_{by}"
                bin_parts_file = f"{self.params.run_tmp_dir_root}/{self.params.case_name}_{bin_name}.hgr.part.2"

                if os.path.exists(bin_parts_file):
                    # read partition result of the bin
                    with open(bin_parts_file, 'r') as f:
                        bin_tier = []
                        for i in range(bin_node_counts[bx, by].item()):
                            line = f.readline().strip()
                            if line:
                                bin_tier.append(int(line))

                    # map partition result of the bin back to global node indices
                    for i, partition_id in enumerate(bin_tier):
                        node_idx = bin_nodes[bx, by, i].item()
                        if node_idx >= 0:  # valid node index
                            tier[node_idx] = partition_id
                else:
                    print(f"warning: parts file not found for bin {bx}_{by}")

        return tier

    def partitioning(self):

        exit_code = os.system(f"openroad {self.tritonpart_tcl_path}")
        if exit_code != 0:
            print(f"error: failed to run tritonpart.tcl")
            return False
        return True

    def flow(self,
             hgr_generator_op,
             parts_reader_op,
             pos=None,
             num_movable_nodes=None):
        if self.enable_bin_based and pos is not None and num_movable_nodes is not None:
            return self.flow_bin_based(hgr_generator_op, pos,
                                       num_movable_nodes)
        else:
            return self.flow_original(hgr_generator_op, parts_reader_op)

    def flow_original(self, hgr_generator_op, parts_reader_op):
        """original flow method (no bin-based)"""
        hgr_generator_op(self.params.case_name)
        self.convert_gp_pl_to_embedding()
        self.partitioning()
        tier = parts_reader_op(self.params.case_name)
        return tier

    def flow_bin_based(self, hgr_generator_op, pos, num_movable_nodes):
        """
        Bin-based partition flow
        
        Args:
            hgr_generator_op: hgr generation function
            pos: position tensor
            num_movable_nodes: number of movable nodes
            
        Returns:
            tier: merged partition result
        """
        print(f"Starting bin-based partition with bin size {self.bin_size}")

        # 1. assign nodes to bins
        bin_nodes, bin_node_counts = self.assign_nodes_to_bins(
            pos, num_movable_nodes)

        # 2.1 generate hgr file for each bin
        for bx in range(self.bin_size[0]):
            for by in range(self.bin_size[1]):
                if bin_node_counts[
                        bx, by].item() > 0:  # only process non-empty bins
                    bin_name = f"bin_{bx}_{by}"
                    print(
                        f"Processing bin {bx}_{by} with {bin_node_counts[bx, by].item()} nodes"
                    )

                    # generate hgr file for each bin
                    hgr_generator_op(f"{self.params.case_name}_{bin_name}",
                                     bin_nodes[bx, by],
                                     bin_node_counts[bx, by].item())

        # 3. convert .gp.pl to .embedding.dat for each bin
        for bx in range(self.bin_size[0]):
            for by in range(self.bin_size[1]):
                if bin_node_counts[
                        bx, by].item() > 0:  # only process non-empty bins
                    self.convert_gp_pl_to_embedding_bin(
                        bx, by, bin_nodes[bx, by], bin_node_counts[bx,
                                                                   by].item())

        # 4. run tritonpart for each bin
        for bx in range(self.bin_size[0]):
            for by in range(self.bin_size[1]):
                if bin_node_counts[
                        bx, by].item() > 0:  # only process non-empty bins
                    self.partitioning_bin(bx, by)

        # 5. merge partition results of all bins
        tier = self.merge_bin_results(bin_nodes, bin_node_counts,
                                      num_movable_nodes)

        print(
            f"Bin-based partition completed. Total nodes: {num_movable_nodes}")
        return tier


if __name__ == "__main__":
    TritonPartBase().partitioning("case2")
