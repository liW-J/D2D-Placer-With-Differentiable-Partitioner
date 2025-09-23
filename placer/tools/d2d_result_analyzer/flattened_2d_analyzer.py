"""
Flattened-2D Placement Analyzer
Analyze flattened-2d placement results and calculate HPWL
"""

import re
import os
from typing import Dict, Tuple, List
import numpy as np


class Flattened2DAnalyzer:
    """Analyze flattened-2d placement file and calculate HPWL"""

    def __init__(self, flattened_pl_file: str):
        """
        Initialize the analyzer
        
        Args:
            flattened_pl_file: flattened-2d placement file path (.gp.pl format)
        """
        self.flattened_pl_file = flattened_pl_file
        self.instance_positions = {}  # instance_name -> (x, y) position
        self.lib_cells = {}  # lib cell information

    def parse_placement_file(self):
        """Parse placement file"""
        print(f"Parsing flattened-2d placement file: {self.flattened_pl_file}")

        if not os.path.exists(self.flattened_pl_file):
            raise FileNotFoundError(
                f"Placement file not found: {self.flattened_pl_file}")

        with open(self.flattened_pl_file, 'r') as f:
            lines = f.readlines()

        # skip header information
        data_started = False
        for line in lines:
            line = line.strip()
            if not line or line.startswith('UCLA'):
                continue

            # parse placement line: C1 3507 4290 : N
            parts = line.split()
            if len(parts) >= 3:
                instance_name = parts[0]
                x = float(parts[1])
                y = float(parts[2])
                self.instance_positions[instance_name] = (x, y)
                data_started = True

    def calculate_net_hpwl_flattened(self, net_instances: List[str],
                                     lib_cells: Dict,
                                     instance_types: Dict) -> float:
        """
        Calculate HPWL of net in flattened-2d using actual pin positions
        
        Args:
            net_instances: list of instances connected to the net
            lib_cells: lib cell information
            instance_types: instance to lib cell type mapping
            
        Returns:
            HPWL
        """
        if not net_instances:
            return 0.0

        # collect all pin positions
        pin_positions = []

        for instance_name in net_instances:
            if instance_name not in self.instance_positions:
                continue

            instance_x, instance_y = self.instance_positions[instance_name]

            # obtain the lib cell type from the instance name
            lib_cell_type = instance_types.get(instance_name)
            if not lib_cell_type or lib_cell_type not in lib_cells:
                # if there is no lib cell information, use the instance center position
                pin_positions.append((instance_x, instance_y))
                continue

            lib_cell = lib_cells[lib_cell_type]
            
            if lib_cell.get('pins'):
                # use the first pin's offset position
                first_pin_name = list(lib_cell['pins'].keys())[0]
                pin_offset = lib_cell['pins'][first_pin_name]
                pin_x = instance_x + pin_offset['x_offset']
                pin_y = instance_y + pin_offset['y_offset']
                pin_positions.append((pin_x, pin_y))
            else:
                # if there is no pin information, use the instance center position
                cell_width = lib_cell.get('width', 0)
                cell_height = lib_cell.get('height', 0)
                center_x = instance_x + cell_width / 2
                center_y = instance_y + cell_height / 2
                pin_positions.append((center_x, center_y))

        if len(pin_positions) < 2:
            return 0.0

        # calculate HPWL
        x_coords = [pos[0] for pos in pin_positions]
        y_coords = [pos[1] for pos in pin_positions]

        hpwl = (max(x_coords) - min(x_coords)) + (max(y_coords) -
                                                  min(y_coords))
        return hpwl

    def analyze_crossing_nets_flattened(self, crossing_nets_data: List[Dict],
                                        net_instances_map: Dict,
                                        lib_cells: Dict,
                                        instance_types: Dict) -> List[Dict]:
        """
        Analyze crossing nets in flattened-2d HPWL
        
        Args:
            crossing_nets_data: original data of crossing nets
            net_instances_map: net name to instance list mapping
            lib_cells: lib cell information
            instance_types: instance to lib cell type mapping
            
        Returns:
            crossing nets data with flattened-2d HPWL information
        """
        print("Analyzing crossing nets in flattened-2d HPWL...")

        enhanced_nets = []

        for net_data in crossing_nets_data:
            net_name = net_data['net_name']

            # get all instances of the net
            net_instances = net_instances_map.get(net_name, [])

            # calculate flattened-2d HPWL
            hpwl_flattened = self.calculate_net_hpwl_flattened(
                net_instances, lib_cells, instance_types)

            # create enhanced net data
            enhanced_net = net_data.copy()
            enhanced_net['hpwl_flattened_2d'] = hpwl_flattened

            # calculate HPWL comparison
            hpwl_top = net_data.get('hpwl_top', 0)
            hpwl_bottom = net_data.get('hpwl_bottom', 0)
            hpwl_total_partitioned = hpwl_top + hpwl_bottom

            enhanced_net['hpwl_total_partitioned'] = hpwl_total_partitioned
            enhanced_net[
                'hpwl_improvement'] = hpwl_total_partitioned - hpwl_flattened
            enhanced_net['hpwl_improvement_ratio'] = (
                (hpwl_total_partitioned - hpwl_flattened) /
                hpwl_total_partitioned *
                100 if hpwl_total_partitioned > 0 else 0)

            enhanced_nets.append(enhanced_net)

        return enhanced_nets
