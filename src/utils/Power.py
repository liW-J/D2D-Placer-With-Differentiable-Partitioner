'''
Author: JeanneWillis hi@jeannewillis.cn
Date: 2025-03-17 18:36:28
LastEditors: JeanneWillis hi@jeannewillis.cn
LastEditTime: 2025-03-17 18:50:57
FilePath: /D2D-placer/src/Power.py
Description: 
'''
import numpy as np

class Power:
    def __init__(self):
        pass

    def estimate_power_with_noise(area):
        """
        estimate power with noise
        """
        # base linear relation: about 20-30 units of power per unit area
        base_power = area * 25.0
        
        # add random noise (normal distribution, standard deviation is 15% of the base value)
        noise = np.random.normal(0, base_power * 0.15)
        
        # ensure power is positive
        estimated_power = max(base_power + noise, 0)
        
        return estimated_power

    def generate_power_for_cells(self, cells):
        """generate power for a set of areas"""
        
        powers = {}
        for cell_name, area in cells.items():
            powers[cell_name] = self.estimate_power_with_noise(area)
            
        return True