'''
Author: JeanneWillis hi@jeannewillis.cn
Date: 2025-07-19 17:57:28
LastEditors: JeanneWillis hi@jeannewillis.cn
LastEditTime: 2025-08-12 00:48:47
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
from colorama import Fore, Style
from placer.ops.parser_txt.parser_txt import ParserTxt
from placer.op_wrapper import OpWrapper
from placer.d2d_params import D2DParams
from placer.tools.dreamplace_base import DreamplaceBaseCollection
from placer.tools.specpart_base import SpecPartBase
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


def init_log(result_root_dir):
    console_formatter = logging.Formatter(
        '[%(levelname)-7s] %(name)s - %(message)s')
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S')

    logging.root.name = 'placement'
    logging.basicConfig(level=logging.INFO,
                        format='[%(levelname)-7s] %(name)s - %(message)s',
                        stream=sys.stdout)

    d2d_logger = logging.getLogger('d2d-logger')
    d2d_logger.setLevel(logging.INFO)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(console_formatter)
    d2d_logger.addHandler(console_handler)

    os.makedirs(result_root_dir, exist_ok=True)
    log_filename = f"{result_root_dir}/d2d-placement.log"
    file_handler = logging.FileHandler(log_filename, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)

    file_handler.setFormatter(file_formatter)
    d2d_logger.addHandler(file_handler)

    return d2d_logger


class D2Dplacer:

    def __init__(self, input_params):
        self.params = D2DParams(input_params)
        self.num_tiers = self.params.flatten_2d.num_tiers
        self.dreamplace = DreamplaceBaseCollection(self.num_tiers)
        self.specpart = SpecPartBase(self.params.run_tmp_dir_root)
        self.op_wrapper = None

        self.format = Format.ICCAD2022

        # macro mask
        self.movable_macro_mask = None  # movable macros in movables nodes
        self.num_movable_macro = [None] * self.num_tiers
        # self.movable_macro_angle = None  # angle of movable macro
        self.node_orient = None  # orient of nodes

        self.cut_net_mask = None
        self.num_terminal_NIs = None

        self.tier = None
        self.timer = None
        self.die_spec = None

    def hpwl_d2d(self, logger=logging):
        if self.dreamplace.dp_terminal.pos is not None:
            hpwl_d2d = self.op_wrapper.d2d_op_collections.hpwl_d2d_op(
                self.dreamplace.dp_2d.pos, self.cut_net_mask, self.tier,
                self.dreamplace.dp_terminal.pos, self.num_terminal_NIs,
                self.dreamplace.dp_terminal.placedb.node_names)
        else:
            hpwl_d2d = self.op_wrapper.d2d_op_collections.hpwl_d2d_op(
                self.dreamplace.dp_2d.pos, self.cut_net_mask, self.tier)
        logger.info("HPWL_D2D:%.6f " % (hpwl_d2d))

        return hpwl_d2d

    def parse_die_spec(self):
        if self.params.flatten_2d.txt_input:
            logging.info("parsing iccad txt input......")
            # parser iccad txt format to aux
            parser_txt = ParserTxt(self.params.flatten_2d.txt_input)
            self.die_spec = parser_txt()

    def init_date(self):
        # control numpy multithreading
        os.environ["OMP_NUM_THREADS"] = "%d" % (
            self.params.flatten_2d.num_threads)

        self.dreamplace.init_all_basic_place(self.params, self.timer)
        self.node_orient = [Orient.N.name
                            ] * self.dreamplace.dp_2d.placedb.num_movable_nodes

        self.cut_net_mask = torch.zeros(self.dreamplace.dp_2d.placedb.num_nets,
                                        dtype=torch.int32)
        self.tier = torch.zeros(
            self.dreamplace.dp_2d.placedb.num_movable_nodes, dtype=torch.int32)

    def init_op_wrapper(self):
        self.op_wrapper = OpWrapper(self.dreamplace.dp_2d,
                                    self.dreamplace.dp_tier, self.params,
                                    self.die_spec)

    def die_by_die_place(self,
                         global_place_flag,
                         legalize_flag,
                         detailed_place_flag,
                         random_center_init_flag,
                         ntuplace_flag,
                         logger=logging):
        self.params.set_die_place_flags(
            global_place_flag=global_place_flag,
            legalize_flag=legalize_flag,
            detailed_place_flag=detailed_place_flag,
            random_center_init_flag=random_center_init_flag,
            ntuplace_flag=ntuplace_flag)
        self.dreamplace.reload_die_basic_place(self.params, self.timer)

        for i in range(self.num_tiers):
            self.dreamplace.dp_tier[i].place(self.params.partition_tier[i],
                                             self.timer)

        # for i in range(self.num_tiers):
        #     logger.info("tier %d placement  HPWL:%.6f " %
        #                  (i, self.dreamplace.metrics_tier[i][-1].hpwl))

        self.op_wrapper.d2d_op_collections.pos_flattened_op(
            self.tier, self.dreamplace.dp_2d.pos,
            [self.dreamplace.dp_tier[i].pos for i in range(self.num_tiers)])

    def flatten_2d_place(self):
        self.dreamplace.dp_2d.place(self.params.flatten_2d, self.timer)

    def partition(self):
        self.tier = self.op_wrapper.d2d_op_collections.hmetis_op(
            self.dreamplace.dp_2d.pos)
        # self.tier = self.op_wrapper.d2d_op_collections.avg_cut_op(self.dreamplace.dp_2d.pos)

        # temporarily call tier result from file
        # self.tier = torch.load('placer/die_tensor.pt')
        # self.tier = self.tier.to(torch.int32)

        self.specpart.flow(self.tier,
                           self.op_wrapper.d2d_op_collections.part_reader_op)

        # return partition result but not receive now
        # pos_2d/2 beceuse of 3d-placer set flattened_die size as die_size*2
        # self.cut_net_mask = self.op_wrapper.d2d_op_collections.init_partition_op(
        #     self.tier, self.pos_2d / 2, self.node_orient)
        self.cut_net_mask = self.op_wrapper.d2d_op_collections.terminal_insert_op(
            self.tier, self.dreamplace.dp_2d.pos / 2, self.node_orient)

        if self.format == Format.ICCAD2023:

            # update placedb_tier & data_tier using new terminal_insert result
            self.dreamplace.reload_die_basic_place(self.params, self.timer)

            self.tier = self.op_wrapper.d2d_op_collections.macro_balance_op(
                self.tier, self.node_orient)

            self.cut_net_mask = self.op_wrapper.d2d_op_collections.terminal_insert_op(
                self.tier, self.dreamplace.dp_2d.pos / 2, self.node_orient)

        self.num_terminal_NIs = int(self.cut_net_mask.sum().item())

    def terminal_insert(self):
        self.op_wrapper.d2d_op_collections.terminal_insert_op(
            self.tier, self.dreamplace.dp_2d.pos, self.node_orient)

        # create terminal aux for collaborative optimization by tier[0]
        self.op_wrapper.d2d_op_collections.terminal_aux_op(
            self.tier, self.dreamplace.dp_2d.pos)

        self.dreamplace.dp_terminal.database(self.params.terminal)
        self.dreamplace.dp_terminal.place(self.params.terminal, self.timer)

        self.op_wrapper.d2d_op_collections.terminal_legalize_op(
            self.tier, self.dreamplace.dp_2d.pos,
            self.dreamplace.dp_terminal.pos, self.num_terminal_NIs,
            self.dreamplace.dp_terminal.placedb.node_names, self.node_orient)

    def refinement(self):
        self.tier = self.op_wrapper.d2d_op_collections.refinement_op(
            self.tier, self.dreamplace.dp_2d.pos,
            self.dreamplace.dp_terminal.pos, self.num_terminal_NIs,
            self.dreamplace.dp_terminal.placedb.node_names)

        self.cut_net_mask = self.op_wrapper.d2d_op_collections.terminal_insert_op(
            self.tier, self.dreamplace.dp_2d.pos, self.node_orient)
        self.num_terminal_NIs = int(self.cut_net_mask.sum().item())

        # create terminal aux for collaborative optimization by tier[0]
        self.op_wrapper.d2d_op_collections.terminal_aux_op(
            self.tier, self.dreamplace.dp_2d.pos)

        self.dreamplace.dp_terminal.database(self.params.terminal)
        self.dreamplace.dp_terminal.place(self.params.terminal, self.timer)

        self.op_wrapper.d2d_op_collections.terminal_legalize_op(
            self.tier, self.dreamplace.dp_2d.pos,
            self.dreamplace.dp_terminal.pos, self.num_terminal_NIs,
            self.dreamplace.dp_terminal.placedb.node_names, self.node_orient)

    def macro_rotation(self):
        # self.dreamplace.placedb_terminal.node_orient = np.array(self.dreamplace.placedb_terminal.node_orient, dtype=np.string_)
        # self.dreamplace.placedb_terminal.node_orient = np.array(self.dreamplace.placedb_terminal.node_orient, dtype=np.string_)
        pass

    def output(self):
        self.op_wrapper.d2d_op_collections.out_fmt_iccad_op(
            self.dreamplace.dp_terminal.placedb, self.params.case_name,
            self.format, self.node_orient)


if __name__ == "__main__":
    """
    @brief main function to invoke the entire placement flow.
    """

    # placement begin
    printWelcome()
    tt = time.time()
    d2d_placer = D2Dplacer(sys.argv[1])
    d2d_logger = init_log(d2d_placer.params.result_dir_root)

    d2d_placer.parse_die_spec()
    d2d_placer.init_date()
    d2d_placer.flatten_2d_place()
    d2d_placer.init_op_wrapper()

    d2d_placer.partition()
    d2d_placer.die_by_die_place(global_place_flag=True,
                                legalize_flag=False,
                                detailed_place_flag=False,
                                random_center_init_flag=False,
                                ntuplace_flag=False)

    d2d_placer.terminal_insert()
    d2d_placer.die_by_die_place(global_place_flag=True,
                                legalize_flag=False,
                                detailed_place_flag=False,
                                random_center_init_flag=False,
                                ntuplace_flag=False)

    d2d_placer.macro_rotation()
    d2d_placer.refinement()
    d2d_placer.die_by_die_place(global_place_flag=False,
                                legalize_flag=False,
                                detailed_place_flag=False,
                                random_center_init_flag=False,
                                ntuplace_flag=True)
    d2d_placer.hpwl_d2d(d2d_logger)
    d2d_placer.output()
