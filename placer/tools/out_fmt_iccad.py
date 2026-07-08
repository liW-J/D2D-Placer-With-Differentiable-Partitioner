'''
Author: JeanneWillis hi@jeannewillis.cn
Date: 2025-04-14 01:03:42
LastEditors: JeanneWillis hi@jeannewillis.cn
LastEditTime: 2025-09-15 19:33:38
FilePath: /D2D-placer/placer/tools/out_fmt_iccad.py
Description: 
'''
import os
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

    def _unscale_tensor_pos(self, pos, placedb):
        num_nodes = placedb.num_nodes
        if pos is None:
            return placedb.unscale_pl(self.params.shift_factor,
                                     self.params.scale_factor)

        pos_np = pos.detach().cpu().numpy()
        node_x = pos_np[:num_nodes].copy()
        node_y = pos_np[num_nodes:num_nodes + num_nodes].copy()

        scale_factor = float(self.params.scale_factor)
        shift_x = self.params.shift_factor[0]
        shift_y = self.params.shift_factor[1]
        if shift_x != 0 or shift_y != 0 or scale_factor != 1.0:
            node_x = node_x / scale_factor + shift_x
            node_y = node_y / scale_factor + shift_y
        return node_x, node_y

    def _write_final_pl(self, placedb, subdir, prefix, pos):
        node_x, node_y = self._unscale_tensor_pos(pos, placedb)
        output_dir = os.path.join(self.params.result_dir, subdir)
        os.makedirs(output_dir, exist_ok=True)
        pl_file = os.path.join(output_dir, f"{prefix}.final.pl")
        placedb.write_pl(self.params, pl_file, node_x, node_y)
        return node_x, node_y

    def __call__(self,
                 placedb_terminal,
                 case_name,
                 format,
                 node_orient,
                 pos_terminal=None):
        """
            @brief write .txt file
            @param output_file .txt file
            """
        content = ""
        output_path = self.params.result_dir
        output_file = output_path + f"/output.txt"
        num_tiers = self.params.num_tiers
        logging.info("output_file: %s" % (output_file))

        tt = time.time()
        logging.info("writing to %s" % (output_path))
        os.makedirs(output_path, exist_ok=True)

        if (num_tiers == 2):
            # write movable nodes in each tier
            for tier_id in range(num_tiers):

                placedb_tier = self.data_tier[tier_id].placedb
                pos_tier = self.data_tier[
                    tier_id].basic_place.data_collections.pos[0]
                prefix = f"tier{tier_id}"
                node_x, node_y = self._write_final_pl(placedb_tier, prefix,
                                                      prefix, pos_tier)
                dieName = "TopDiePlacement" if tier_id == 0 else "BottomDiePlacement"

                num_movable_nodes = placedb_tier.num_movable_nodes
                rawdb = placedb_tier.rawdb

                content += f"{dieName} {num_movable_nodes}\n"

                for node_id in range(num_movable_nodes):
                    if format == Format.ICCAD2022:
                        content += f"Inst {rawdb.nodeName(node_id)} {int(node_x[node_id])} {int(node_y[node_id])}\n"
                    elif format == Format.ICCAD2023:
                        content += f"Inst {rawdb.nodeName(node_id)} {int(node_x[node_id])} {int(node_y[node_id])} {Orient[node_orient[node_id]].value}\n"

            # write terminal nodes
            content += f"NumTerminals {placedb_terminal.num_movable_nodes}\n"

            rawdb_terminal = placedb_terminal.rawdb
            terminal_x, terminal_y = self._write_final_pl(
                placedb_terminal, "terminal", "terminal", pos_terminal)
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
