'''
Author: JeanneWillis hi@jeannewillis.cn
Date: 2025-03-17 18:36:28
LastEditors: JeanneWillis hi@jeannewillis.cn
LastEditTime: 2025-07-19 17:57:13
FilePath: /D2D-placer/placer/tools/power.py
Description: 
'''
import numpy as np
from typing import Dict, Tuple

class Power:
    def __init__(self):
        self.power_cache: Dict[Tuple[float, float], float] = {}
        self.base_area = 3.088
        self.area_factor = 25.0
        self.noise_factor = 0.15
        self.area_scale = 142.67
        
    def estimate_power_with_noise(self, area):
        """
        estimate power with noise
        """
        # base linear relation: about 20-30 units of power per unit area
        base_power = area * self.area_factor
        # add random noise (normal distribution, standard deviation is 15% of the base value)
        noise = np.random.normal(0, base_power * self.noise_factor)
    
        # ensure power is positive
        estimated_power = max(base_power + noise, 0)
        
        return estimated_power

    def generate_power_for_cells(self, placedb):
        """generate power for nodes"""    
        
        placedb.node_power = np.zeros(placedb.num_physical_nodes, dtype = placedb.dtype)
        areas = placedb.node_size_x * placedb.node_size_y
        
        self.area_factor = self.area_factor / (self.area_scale**2)
        
        for i in range(placedb.num_physical_nodes):
            node_key = (placedb.node_size_x[i], placedb.node_size_y[i])
            if node_key in self.power_cache:
                power = self.power_cache[node_key]
            else:
                power = self.estimate_power_with_noise(areas[i])
                self.power_cache[node_key] = power
                    
            placedb.node_power[i] = power
            
        return True