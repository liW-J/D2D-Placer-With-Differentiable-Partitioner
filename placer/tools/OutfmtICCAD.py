'''
Author: JeanneWillis hi@jeannewillis.cn
Date: 2025-04-14 01:03:42
LastEditors: JeanneWillis hi@jeannewillis.cn
LastEditTime: 2025-04-14 02:28:45
FilePath: /D2D-placer/placer/tools/OutfmtICCAD.py
Description: 
'''
import time
import logging


class OutfmtICCAD:

    def __init__(self, placedb_tier, params):
        self.placedb_tier = placedb_tier
        self.params = params

    def output_iccad_fmt(self, params):
        """
            @brief write .txt file
            @param output_file .txt file
            """
        # read .gp.pl file
        # TODO: case_name
        content = ""
        output_path = "%s" % (params.result_dir)
        output_file = output_path + f"/case1/output.txt"
        num_tiers = self.params.num_tiers
        logging.info("output_file: %s" % (output_file))

        tt = time.time()
        logging.info("writing to %s" % (output_path))
        if (num_tiers == 2):
            for i in range(num_tiers):
                pl_file = output_path + f"/tier{i}/tier{i}.gp.pl"
                node_x, node_y = self.placedb_tier[i].unscale_pl(params.shift_factor,
                                                 params.scale_factor)
                dieName = "TopDiePlacement" if i == 0 else "BottomDiePlacement"

                self.placedb_tier[i].read_pl(self.params, pl_file)
                num_movable_nodes = self.placedb_tier[i].num_movable_nodes
                rawdb = self.placedb_tier[i].rawdb

                content += f"{dieName} {num_movable_nodes}\n"

                for node_id in range(num_movable_nodes):
                    content += f"Inst {rawdb.nodeName(node_id)} {node_x[node_id]} {node_y[node_id]}\n"

        else:
            logging.info("unsupported num_tiers: %d for iccad format" %
                         (num_tiers))

        with open(output_file, "w") as f:
            f.write(content)
        logging.info("output_iccad_fmt takes %.3f seconds" %
                     (time.time() - tt))
