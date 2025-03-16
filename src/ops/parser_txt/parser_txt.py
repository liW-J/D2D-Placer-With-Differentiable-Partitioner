'''
Author: JeanneWillis hi@jeannewillis.cn
Date: 2025-03-15 14:13:35
LastEditors: JeanneWillis hi@jeannewillis.cn
LastEditTime: 2025-03-16 22:34:27
FilePath: /D2D-placer/src/ops/parser_txt/parser_txt.py
Description: parser ICCAD 3D placement txt file
'''

# from ...configure import *
from torch.autograd import Function

import logging
logger = logging.getLogger(__name__)

import parser_txt_cpp

class ParserTxtFunction(Function):
    @staticmethod
    def forward(file_path, output_path):
        
        func = parser_txt_cpp.parser_txt
        input_args = ["3d-placer", file_path, output_path]
        output = func(input_args)
        
        return output
    
class ParserTxt(object):
    def __init__(self, file_path, output_path):
        self.file_path = file_path
        self.output_path = output_path
        
    def __call__(self):
        return ParserTxtFunction.forward(self.file_path, self.output_path)

if __name__ == "__main__":
    parser_txt = ParserTxt("/D2D-placer/benchmarks/iccad2022/case1.txt", "/D2D-placer/output/case1_output.txt")
    result = parser_txt()
    print("ok!")
