'''
Author: JeanneWillis hi@jeannewillis.cn
Date: 2025-03-17 17:35:04
LastEditors: JeanneWillis hi@jeannewillis.cn
LastEditTime: 2025-03-17 17:35:13
FilePath: /D2D-placer/src/utils/liberty_power_extractor/power_test.py
Description: 
'''

import numpy as np

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

def generate_power_for_cells(areas):
    """generate power for a set of areas"""
    powers = {}
    for cell_name, area in areas.items():
        powers[cell_name] = estimate_power_with_noise(area)
    return powers

test_areas = {
    'CELL1': 0.532,  # small area cell
    'CELL2': 1.064,  # medium area cell
    'CELL3': 2.394,  # large area cell
}

results = generate_power_for_cells(test_areas)
for cell, power in results.items():
    print(f"{cell}: Area={test_areas[cell]:.3f}, Estimated Power={power:.3f}")