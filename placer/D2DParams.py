'''
Author: JeanneWillis hi@jeannewillis.cn
Date: 2025-06-13 20:00:00
LastEditors: JeanneWillis hi@jeannewillis.cn
LastEditTime: 2025-06-16 23:46:49
FilePath: /D2D-placer/placer/D2DParams.py
Description: 
'''
import dreamplace.Params as Params
import time
import os


class D2DParams:

    def __init__(self, json_path):
        self.tt_format = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
        self.case_name = os.path.splitext(os.path.basename(json_path))[0]
        
        self.flatten_2d = Params.Params()
        self.terminal = Params.Params()

        self.flatten_2d.load(json_path)
        self.terminal.load(json_path)

        self.flatten_2d.aux_input = f"run_tmp/{self.case_name}/flattened-2d/flattened-2d.aux"
        self.terminal.aux_input = f"run_tmp/{self.case_name}/terminal/terminal.aux"
        
        self.flatten_2d.result_dir = f"results/{self.case_name}/{self.tt_format}"
        self.terminal.result_dir = f"results/{self.case_name}/{self.tt_format}"
        

        self.flattened_tier = [
            Params.Params() for _ in range(self.flatten_2d.num_tiers)
        ]
        self.partition_tier = [
            Params.Params() for _ in range(self.flatten_2d.num_tiers)
        ]

        for i in range(self.flatten_2d.num_tiers):
            self.flattened_tier[i].load(json_path)
            self.partition_tier[i].load(json_path)
            self.flattened_tier[i].result_dir = f"results/{self.case_name}/{self.tt_format}"
            self.partition_tier[i].result_dir = f"results/{self.case_name}/{self.tt_format}"

            self.flattened_tier[
                i].aux_input = f"run_tmp/{self.case_name}/flattened-2d/tier{i}.aux"
            self.partition_tier[
                i].aux_input = f"run_tmp/{self.case_name}/partition/tier{i}.aux"

        # special params
        self.terminal.random_center_init_flag = 0
        self.terminal.detailed_place_flag = 1