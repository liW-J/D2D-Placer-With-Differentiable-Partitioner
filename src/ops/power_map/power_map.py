'''
Author: JeanneWillis hi@jeannewillis.cn
Date: 2025-03-18 16:21:18
LastEditors: JeanneWillis hi@jeannewillis.cn
LastEditTime: 2025-03-20 18:07:51
FilePath: /D2D-placer/src/ops/power_map/power_map.py
Description: Compute power map on CPU
'''
import logging
logger = logging.getLogger(__name__)

import ops.power_map.power_map_cpp as power_map_cpp
import torch
from torch.autograd import Function
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class PowerMapFunction(Function):
    @staticmethod
    def forward(pos, node_size_x, node_size_y, 
                initial_power_map, xl, yl, xh, yh, 
                num_bins_x, num_bins_y, 
                range_begin, range_end, 
                deterministic_flag, power):
        
        func = power_map_cpp.power_map
        output = func(pos.view(pos.numel()), node_size_x, node_size_y,
                      initial_power_map, xl, yl, xh, yh, 
                      num_bins_x, num_bins_y, 
                      range_begin, range_end, 
                      deterministic_flag, power)
        return output
    
class PowerMap(object):
    def __init__(self, node_size_x, node_size_y, 
                 xl, yl, xh, yh, num_bins_x, num_bins_y, 
                 range_list, 
                 deterministic_flag, 
                 power,
                 initial_power_map=None):
        
        super(PowerMap, self).__init__()
        self.node_size_x = node_size_x
        self.node_size_y = node_size_y
        self.xl = xl
        self.yl = yl
        self.xh = xh
        self.yh = yh
        self.num_bins_x = num_bins_x
        self.num_bins_y = num_bins_y
        self.range_list = range_list
        self.deterministic_flag = deterministic_flag
        self.initial_power_map = initial_power_map
        self.power = power
    def forward(self, pos):
        """
        @brief API 
        @param pos cell locations. The array consists of x locations of movable cells, fixed cells, and filler cells, then y locations of them 
        """
        if self.initial_power_map is None:
            self.initial_power_map = torch.zeros(self.num_bins_x, self.num_bins_y, dtype=pos.dtype, device=pos.device)
            #plot(self.initial_power_map.clone().div(self.bin_size_x*self.bin_size_y).cpu().numpy(), 'initial_power_map')

        power_map = self.initial_power_map
        for index_range in self.range_list: 
            if index_range[0] < index_range[1]: 
                power_map = PowerMapFunction.forward(
                    pos=pos,
                    node_size_x=self.node_size_x,
                    node_size_y=self.node_size_y,
                    initial_power_map=power_map,
                    xl=self.xl,
                    yl=self.yl,
                    xh=self.xh,
                    yh=self.yh,
                    num_bins_x=self.num_bins_x,
                    num_bins_y=self.num_bins_y,
                    range_begin=index_range[0],
                    range_end=index_range[1], 
                    deterministic_flag=self.deterministic_flag,
                    power=self.power)
        breakpoint()

        return power_map
      
    def __call__(self, pos):
        return self.forward(pos)

def plot(power_map, name):
    """
    @brief density map contour and heat map 
    """
    print(np.amax(power_map))
    print(np.mean(power_map))
    fig = plt.figure(figsize=(4, 3))
    ax = fig.gca(projection='3d')

    x = np.arange(power_map.shape[0])
    y = np.arange(power_map.shape[1])

    x, y = np.meshgrid(x, y)
    ax.plot_surface(x, y, power_map, alpha=0.8)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('density')

    #plt.tight_layout()
    plt.savefig(name + ".3d.png")

    plt.clf()

    fig, ax = plt.subplots()

    ax.pcolor(power_map)

    # Loop over data dimensions and create text annotations.
    #for i in range(power_map.shape[0]):
    #    for j in range(power_map.shape[1]):
    #        text = ax.text(j, i, power_map[i, j],
    #                ha="center", va="center", color="w")
    fig.tight_layout()
    plt.savefig(name + ".2d.png")

if __name__ == "__main__":
    pass
