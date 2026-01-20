'''
Author: JeanneWillis hi@jeannewillis.cn
Date: 2025-06-13 20:00:00
LastEditors: JeanneWillis hi@jeannewillis.cn
LastEditTime: 2026-01-20 02:03:44
FilePath: /D2D-placer/placer/d2d_params.py
Description: 
'''
import dreamplace.Params as Params
import time
import os
from configure import compile_configurations


class D2DParams:

    def __init__(self, json_path):
        self.tt_format = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
        self.case_name = os.path.splitext(os.path.basename(json_path))[0]
        self.run_tmp_dir_root = f"{compile_configurations['PLACER_RUNTMP_DIR']}/{self.case_name}"
        self.result_dir_root = f"{compile_configurations['PLACER_RESULT_DIR']}/{self.case_name}/{self.tt_format}"

        self.flatten_2d = Params.Params()
        self.terminal = Params.Params()

        self.flatten_2d.load(json_path)
        self.terminal.load(json_path)
        self.num_tiers = self.flatten_2d.num_tiers

        self.flatten_2d.aux_input = f"{self.run_tmp_dir_root}/flattened-2d/flattened-2d.aux"
        self.terminal.aux_input = f"{self.run_tmp_dir_root}/terminal/terminal.aux"

        self.flatten_2d.result_dir = self.result_dir_root
        self.terminal.result_dir = self.result_dir_root

        self.flattened_tier = [Params.Params() for _ in range(self.num_tiers)]
        self.partition_tier = [Params.Params() for _ in range(self.num_tiers)]

        for i in range(self.num_tiers):
            self.flattened_tier[i].load(json_path)
            self.partition_tier[i].load(json_path)
            self.flattened_tier[i].result_dir = self.result_dir_root
            self.partition_tier[i].result_dir = self.result_dir_root

            self.flattened_tier[
                i].aux_input = f"{self.run_tmp_dir_root}/flattened-2d/tier{i}.aux"
            self.partition_tier[
                i].aux_input = f"{self.run_tmp_dir_root}/partition/tier{i}.aux"

        # special params
        self.terminal.random_center_init_flag = True
        self.terminal.global_place_stages[0]["iteration"] = 500
        self.terminal.stop_overflow = 0.01

        self.terminal.global_place_flag = True
        self.terminal.legalize_flag = False
        self.terminal.detailed_place_flag = False
        self.terminal.ntuplace_flag = True

        self.flatten_2d.global_place_flag = True
        self.flatten_2d.legalize_flag = False
        self.flatten_2d.detailed_place_flag = False
        self.flatten_2d.ntuplace_flag = True
        # self.flatten_2d.target_density = 2.0

    def set_die_place_flags(self,
                            random_center_init_flag=False,
                            global_place_flag=False,
                            legalize_flag=False,
                            detailed_place_flag=False,
                            ntuplace_flag=False):
        """
        @brief Set flags for all tiers.
        @param random_center_init_flag: Whether to use random center initialization.
        @param global_place_flag: Whether to use global placement.
        @param legalize_flag: Whether to use legalization.
        @param detailed_place_flag: Whether to use detailed placement.
        @param ntuplace_flag: Whether to use NTUplace.
        """
        for i in range(self.num_tiers):
            self.partition_tier[
                i].random_center_init_flag = random_center_init_flag
            self.partition_tier[i].global_place_flag = global_place_flag
            self.partition_tier[i].legalize_flag = legalize_flag
            self.partition_tier[i].detailed_place_flag = detailed_place_flag
            self.partition_tier[i].ntuplace_flag = ntuplace_flag
