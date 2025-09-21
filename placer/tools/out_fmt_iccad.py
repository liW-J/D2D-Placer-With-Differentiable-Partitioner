'''
Author: JeanneWillis hi@jeannewillis.cn
Date: 2025-04-14 01:03:42
LastEditors: JeanneWillis hi@jeannewillis.cn
LastEditTime: 2025-09-15 19:33:38
FilePath: /D2D-placer/placer/tools/out_fmt_iccad.py
Description: 
'''
import time
import logging
from placer.constants import Format, Orient


class OutfmtICCAD:

    def __init__(self, data_tier, params, die_spec):
        self.data_tier = data_tier
        self.params = params
        self.die_spec = die_spec
        self.terminal_size_x = die_spec.terminalSizeX
        self.terminal_size_y = die_spec.terminalSizeY
        self.terminal_spacing = die_spec.terminalSpacing

    def __call__(self, placedb_terminal, case_name, format, node_orient):
        """
            @brief write .txt file
            @param output_file .txt file
            """
        # read .gp.pl file
        content = ""
        output_path = self.params.result_dir
        output_file = output_path + f"/output.txt"
        num_tiers = self.params.num_tiers
        logging.info("output_file: %s" % (output_file))

        tt = time.time()
        logging.info("writing to %s" % (output_path))

        if (num_tiers == 2):
            # write movable nodes in each tier
            for tier_id in range(num_tiers):

                node_x, node_y = self.data_tier[tier_id].placedb.unscale_pl(
                    self.params.shift_factor, self.params.scale_factor)
                dieName = "TopDiePlacement" if tier_id == 0 else "BottomDiePlacement"

                num_movable_nodes = self.data_tier[
                    tier_id].placedb.num_movable_nodes
                rawdb = self.data_tier[tier_id].placedb.rawdb

                content += f"{dieName} {num_movable_nodes}\n"

                for node_id in range(num_movable_nodes):
                    if format == Format.ICCAD2022:
                        content += f"Inst {rawdb.nodeName(node_id)} {int(node_x[node_id])} {int(node_y[node_id])}\n"
                    elif format == Format.ICCAD2023:
                        content += f"Inst {rawdb.nodeName(node_id)} {int(node_x[node_id])} {int(node_y[node_id])} {Orient[node_orient[node_id]].value}\n"

            # write terminal nodes
            content += f"NumTerminals {placedb_terminal.num_movable_nodes}\n"

            rawdb_terminal = placedb_terminal.rawdb
            terminal_x, terminal_y = placedb_terminal.unscale_pl(
                self.params.shift_factor, self.params.scale_factor)
            for terminal_id in range(0, placedb_terminal.num_movable_nodes):
                # node_x, node_y of terminal must be same in each tier
                # rawdb here is tier[-1] for easier
                content += f"Terminal {rawdb_terminal.nodeName(terminal_id)} {int(terminal_x[terminal_id] + (self.terminal_size_x + self.terminal_spacing) / 2)} {int(terminal_y[terminal_id] + (self.terminal_size_y + self.terminal_spacing) / 2)}\n"
        else:
            logging.info("unsupported num_tiers: %d for iccad format" %
                         (num_tiers))

        with open(output_file, "w") as f:
            f.write(content)
        logging.info("output_iccad_fmt takes %.3f seconds" %
                     (time.time() - tt))
