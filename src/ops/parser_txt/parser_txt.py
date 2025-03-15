'''
Author: JeanneWillis hi@jeannewillis.cn
Date: 2025-03-15 14:13:35
LastEditors: JeanneWillis hi@jeannewillis.cn
LastEditTime: 2025-03-15 14:47:23
FilePath: /D2D-placer/src/ops/parser_txt/parser_txt.py
Description: parser ICCAD 3D placement txt file
'''

from src import configure
from torch.autograd import Function

import logging
logger = logging.getLogger(__name__)

import parser_txt.parser_txt_cpp as parser_txt_cpp

class ParserTxtFunction(Function):
    @staticmethod
    def forward(ctx, file_path):
        
        func = parser_txt_cpp.parser_txt
        output = func(file_path)
        
        return output
    
class ParserTxt(object):
    def __init__(self, file_path):
        self.file_path = file_path
        
    def __call__(self):
        return ParserTxtFunction.forward(self.file_path)
      
  
