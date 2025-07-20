'''
Author: JeanneWillis hi@jeannewillis.cn
Date: 2025-07-19 17:57:28
LastEditors: JeanneWillis hi@jeannewillis.cn
LastEditTime: 2025-07-20 21:15:57
FilePath: /D2D-placer/placer/d2d_placer.py
Description: 
'''

import configure
import matplotlib

matplotlib.use('Agg')

import os
import sys
import time
import numpy as np
import logging
# for consistency between python2 and python3
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root_dir not in sys.path:
    sys.path.append(root_dir)
import dreamplace.BasicPlace as BasicPlace
from colorama import Fore, Style
from placer.ops.parser_txt.parser_txt import ParserTxt
from placer.op_wrapper import OpWrapper
from placer.d2d_params import D2DParams
from placer.tools.dreamplace_data import DreamplaceData
from placer.constants import Format, Orient
import torch


def printWelcome():
    welcome_msg = f"""
{Fore.BLUE}================================================================
                     D2D-Placer v1.0.0 (2025)                     
----------------------------------------------------------------{Style.RESET_ALL}
{Fore.GREEN}     A Die-to-Die Placement Research Framework      {Style.RESET_ALL}
{Fore.YELLOW}----------------------------------------------------------------
     Website  : https://github.com/liW-J/D2D-Placer                    
     Contact  : hi@jeannewillis.cn                          
     License  : Apache/MIT License                              
================================================================{Style.RESET_ALL}
"""
    print(welcome_msg)


class D2Dplacer:

    def __init__(self, input_params):
        self.params = D2DParams(input_params)
        self.num_tiers = self.params.flatten_2d.num_tiers
        self.place_data = DreamplaceData(self.num_tiers)

        self.format = Format.ICCAD2023

        self.pos_2d = None
        self.pos_tier = [None] * self.num_tiers
        self.pos_terminal = None

        # macro mask
        self.movable_macro_mask = None  # movable macros in movables nodes
        # self.movable_macro_angle = None  # angle of movable macro
        self.node_orient = None  # orient of nodes

        self.cut_net_mask = None
        self.num_terminal_NIs = None

        self.tier = None
        self.timer = None
        self.die_spec = None

        self.op_wrapper = None

    @property
    def num_movable_macro(self):
        """
        @return number of movable macro nodes
        """
        return self.movable_macro_mask.sum()

    def hpwl_d2d(self):
        hpwl_d2d = self.op_wrapper.d2d_op_collections.hpwl_d2d_op(
            self.pos_2d, self.cut_net_mask, self.tier, self.pos_terminal,
            self.num_terminal_NIs, self.place_data.placedb_terminal.node_names)
        logging.info("HPWL_D2D:%.6f " % (hpwl_d2d))

        return hpwl_d2d

    def parse_die_spec(self):
        if self.params.flatten_2d.txt_input:
            logging.info("parsing iccad txt input......")
            # parser iccad txt format to aux
            parser_txt = ParserTxt(self.params.flatten_2d.txt_input)
            self.die_spec = parser_txt()

    def init_basic_date(self):
        # control numpy multithreading
        os.environ["OMP_NUM_THREADS"] = "%d" % (
            self.params.flatten_2d.num_threads)

        self.place_data.placedb_2d, self.timer = DreamplaceData.database(
            self.params.flatten_2d)
        self.place_data.data_2d = BasicPlace.BasicPlace(
            self.params.flatten_2d, self.place_data.placedb_2d, self.timer)

        # save each tier's placedb for backup
        for i in range(self.num_tiers):
            self.place_data.placedb_tier[
                i], self.timer = DreamplaceData.database(
                    self.params.flattened_tier[i])
            self.place_data.data_tier[i] = BasicPlace.BasicPlace(
                self.params.flattened_tier[i], self.place_data.placedb_tier[i],
                self.timer)

        self.node_orient = [Orient.N.name
                            ] * self.place_data.placedb_2d.num_movable_nodes

    def init_op_wrapper(self):
        self.op_wrapper = OpWrapper(self.place_data.data_2d,
                                    self.place_data.placedb_2d,
                                    self.place_data.placedb_tier, self.params,
                                    self.place_data.data_tier, self.die_spec)

    def die_by_die_place(self, random_center_init_flag):
        for i in range(self.num_tiers):
            self.params.partition_tier[
                i].random_center_init_flag = random_center_init_flag

            # update placedb_tier & data_tier using new terminal_insert result
            self.place_data.placedb_tier[
                i], self.timer = DreamplaceData.database(
                    self.params.partition_tier[i])
            self.place_data.data_tier[i] = BasicPlace.BasicPlace(
                self.params.partition_tier[i], self.place_data.placedb_tier[i],
                self.timer)

            self.place_data.metrics_tier[i], self.pos_tier[
                i] = DreamplaceData.place(self.params.partition_tier[i],
                                          self.place_data.placedb_tier[i],
                                          self.timer)

        for i in range(self.num_tiers):
            logging.info("tier %d placement  HPWL:%.6f " %
                         (i, self.place_data.metrics_tier[i][-1].hpwl))

        self.op_wrapper.d2d_op_collections.pos_flattened_op(
            self.tier, self.pos_2d, self.pos_tier)

    def terminal_place(self):
        pass

    def flatten_2d_place(self):
        self.place_data.metrics_2d, self.pos_2d = DreamplaceData.place(
            self.params.flatten_2d, self.place_data.placedb_2d, self.timer)

    def partition(self):
        self.tier = self.op_wrapper.d2d_op_collections.hmetis_op(self.pos_2d)
        # tier = avg_cut(pin_pos_op(d2d_placer.pos_2d), node_size_x, node_size_y)
        # tier = multi_bipartition(d2d_placer.pos_2d)

        # bin-based partition
        # temporarily call tier result from file
        # tier = torch.load('placer/die_tensor.pt')
        # d2d_placer.tier = d2d_placer.tier.to(torch.int32)

        # return partition result but not receive now
        # pos_2d/2 beceuse of 3d-placer set flattened_die size as die_size*2
        # cut_net_mask = self.op_wrapper.d2d_op_collections.init_partition_op(
        #     self.tier, self.pos_2d / 2)
        self.cut_net_mask = self.op_wrapper.d2d_op_collections.terminal_insert_op(
            self.tier, self.pos_2d / 2, self.node_orient)
        self.num_terminal_NIs = int(self.cut_net_mask.sum().item())

    def terminal_insert(self):
        self.op_wrapper.d2d_op_collections.terminal_insert_op(
            self.tier, self.pos_2d, self.node_orient)

        # create terminal aux for collaborative optimization by tier[0]
        self.op_wrapper.d2d_op_collections.terminal_aux_op(
            self.tier, self.pos_2d)

        self.place_data.placedb_terminal, self.timer = DreamplaceData.database(
            self.params.terminal)
        self.place_data.metrics_terminal, self.pos_terminal = DreamplaceData.place(
            self.params.terminal, self.place_data.placedb_terminal, self.timer)

        self.op_wrapper.d2d_op_collections.terminal_legalize_op(
            self.tier, self.pos_2d, self.pos_terminal, self.num_terminal_NIs,
            self.place_data.placedb_terminal.node_names, self.node_orient)

    def refinement(self):
        self.tier = self.op_wrapper.d2d_op_collections.refinement_op(
            self.tier, self.pos_2d, self.pos_terminal, self.num_terminal_NIs,
            self.place_data.placedb_terminal.node_names)

        self.cut_net_mask = self.op_wrapper.d2d_op_collections.terminal_insert_op(
            self.tier, self.pos_2d, self.node_orient)
        self.num_terminal_NIs = int(self.cut_net_mask.sum().item())

        # create terminal aux for collaborative optimization by tier[0]
        self.op_wrapper.d2d_op_collections.terminal_aux_op(
            self.tier, self.pos_2d)

        self.place_data.placedb_terminal, self.timer = DreamplaceData.database(
            self.params.terminal)
        self.place_data.metrics_terminal, terminal_pos = DreamplaceData.place(
            self.params.terminal, self.place_data.placedb_terminal, self.timer)

        self.op_wrapper.d2d_op_collections.terminal_legalize_op(
            self.tier, self.pos_2d, terminal_pos, self.num_terminal_NIs,
            self.place_data.placedb_terminal.node_names, self.node_orient)

    def macro_rotation(self):
        # self.place_data.placedb_terminal.node_orient = np.array(self.place_data.placedb_terminal.node_orient, dtype=np.string_)
        # self.place_data.placedb_terminal.node_orient = np.array(self.place_data.placedb_terminal.node_orient, dtype=np.string_)
        pass

    def output(self):
        self.op_wrapper.d2d_op_collections.out_fmt_iccad_op(
            self.place_data.placedb_terminal, self.params.case_name,
            self.format, self.node_orient)

    def run(self):
        self.parse_die_spec()
        self.init_basic_date()
        self.flatten_2d_place()
        self.init_op_wrapper()
        self.partition()
        self.die_by_die_place(random_center_init_flag=True)
        self.terminal_insert()
        self.die_by_die_place(random_center_init_flag=False)
        self.macro_rotation()
        self.refinement()
        for i in range(self.num_tiers):
            self.params.partition_tier[i].global_place_flag = 0
        self.die_by_die_place(random_center_init_flag=False)
        self.hpwl_d2d()
        self.output()


if __name__ == "__main__":
    """
    @brief main function to invoke the entire placement flow.
    """
    logging.root.name = 'D2Dplacer'
    logging.basicConfig(level=logging.INFO,
                        format='[%(levelname)-7s] %(name)s - %(message)s',
                        stream=sys.stdout)

    d2d_placer = D2Dplacer(sys.argv[1])

    # placement begin
    printWelcome()
    tt = time.time()

    d2d_placer.run()

    # # load parameters
    # # parse input get flattened .aux
    # d2d_placer.parse_die_spec()

    # d2d_placer.init_basic_date()

    # # dreamplace for flattened 2d placement
    # logging.info("flattened 2d placement begin")
    # d2d_placer.flatten_2d_place()

    # # prepare for partitioning
    # d2d_placer.init_op_wrapper()

    # # partition
    # d2d_placer.partition()
    # breakpoint()

    # # if 2D result for init pos may casued no convergence
    # d2d_placer.die_by_die_place(random_center_init_flag=True)
    # logging.info("2d placement  HPWL:%.6f " %
    #              (d2d_placer.place_data.metrics_2d[-1].hpwl))
    # breakpoint()

    # # terminal insert
    # d2d_placer.terminal_insert()
    # d2d_placer.die_by_die_place(random_center_init_flag=False)

    # # refinement
    # d2d_placer.refinement()

    # d2d_placer.die_by_die_place(random_center_init_flag=False)
    # d2d_placer.hpwl_d2d()
    # logging.info("placement takes %.3f seconds" % (time.time() - tt))

    # d2d_placer.output()
    # breakpoint()
