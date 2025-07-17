'''
Author: JeanneWillis hi@jeannewillis.cn
Date: 2025-07-17 13:11:20
LastEditors: JeanneWillis hi@jeannewillis.cn
LastEditTime: 2025-07-17 13:53:25
FilePath: /D2D-placer/placer/D2DplaceDB.py
Description: 
'''

import dreamplace.PlaceDB as PlaceDB

class D2DPlaceDB (object):
    """
    @brief placement database
    """
    def __init__(self, num_tiers):
        """
        initialization
        To avoid the usage of list, I flatten everything.
        """
        self.placedb_2d = None # raw placement database, a C++ object
        self.placedb_tier = [None] * num_tiers # python placement database interface

        # macro mask
        self.movable_macro_mask = None # movable macros in movables nodes
        self.movable_macro_angle = None # angle of movable macros
    
    @property
    def num_movable_nodes(self):
        """
        @return number of movable macro nodes
        """
        return self.movable_macro_mask.sum()


       