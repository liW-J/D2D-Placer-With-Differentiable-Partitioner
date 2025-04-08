'''
Author: JeanneWillis hi@jeannewillis.cn
Date: 2025-03-19 11:47:31
LastEditors: JeanneWillis hi@jeannewillis.cn
LastEditTime: 2025-04-08 12:22:12
FilePath: /D2D-placer/placer/ops/partition/partition.py
Description: partition flattened 2D placement to 2 Die
'''
from torch.autograd import Function

import logging
logger = logging.getLogger(__name__)

import ops.partition.partition_cpp as partition_cpp

class PartitionFunction(Function):
    @staticmethod
    def forward(pos, die_size_x, die_size_y):
        func = partition_cpp.partition
        input_args = [pos, die_size_x, die_size_y]
        output = func(input_args)
        
        return output
    
class Partition(object):
    def __init__(self, pos, die_size_x, die_size_y):
        self.pos = pos
        self.die_size_x = die_size_x
        self.die_size_y = die_size_y
        
    def __call__(self):
        return PartitionFunction.forward(self.pos, self.die_size_x, self.die_size_y)

if __name__ == "__main__":
    pass