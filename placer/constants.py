'''
Author: JeanneWillis hi@jeannewillis.cn
Date: 2025-07-19 17:20:57
LastEditors: JeanneWillis hi@jeannewillis.cn
LastEditTime: 2025-07-20 01:40:01
FilePath: /D2D-placer/placer/contants.py
Description: 
'''
from enum import Enum, auto


class Format(Enum):
    ICCAD2022 = "iccad2022"
    ICCAD2023 = "iccad2023"
    
class Orient(Enum):
    N = "N"
    S = "S"
    E = "E"
    W = "W"
    FN = "FN"
    FS = "FS"
    FE = "FE"
    FW = "FW"

