'''
Date: 2025-09-23 23:06:56
LastEditTime: 2025-09-25 14:13:58
FilePath: /D2D-placer/placer/ops/draw_layout_result/draw_layout_result.py
Description: 
'''

# from ...configure import *
from torch.autograd import Function

import logging

logger = logging.getLogger(__name__)

import placer.ops.draw_layout_result.draw_layout_result_cpp as draw_layout_result_cpp


class DrawLayoutResultFunction(Function):

    @staticmethod
    def forward(file_path, result_dir):

        func = draw_layout_result_cpp.draw_layout_result
        input_args = ["3d-placer", file_path, result_dir]
        output = func(input_args)

        return output


class DrawLayoutResult(object):

    def __init__(self, file_path, result_dir):
        self.file_path = file_path
        self.result_dir = result_dir

    def __call__(self):
        return DrawLayoutResultFunction.forward(self.file_path,
                                                self.result_dir)


if __name__ == "__main__":
    pass
