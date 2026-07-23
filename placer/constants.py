'''
Date: 2025-07-19 17:20:57
LastEditTime: 2025-07-23 15:42:09
FilePath: /D2D-placer/placer/constants.py
Description: 
'''
from enum import Enum, auto


class Format(Enum):
    ICCAD2022 = "ICCAD2022"
    ICCAD2023 = "ICCAD2023"


class Orient(Enum):
    N = "R0"  # 0
    S = "R180"  # 180
    E = "R90"  # 90
    W = "R270"  # 270
    FN = "FN"  # flip 0
    FS = "FS"  # flip 180
    FE = "FE"  # flip 90
    FW = "FW"  # flip 270
