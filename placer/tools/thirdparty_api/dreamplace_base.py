'''
Author: JeanneWillis hi@jeannewillis.cn
Date: 2025-07-17 14:31:37
LastEditors: JeanneWillis hi@jeannewillis.cn
LastEditTime: 2025-10-29 23:34:02
FilePath: /D2D-placer/placer/tools/dreamplace_base.py
Description: 
'''
import dreamplace.PlaceDB as PlaceDB
import dreamplace.Timer as Timer
import dreamplace.configure as configure
import dreamplace.NonLinearPlace as NonLinearPlace
import dreamplace.BasicPlace as BasicPlace
import time
import logging
import numpy as np
import os
import re


class DreamplaceBase:

    def __init__(self):
        self.placedb = PlaceDB.PlaceDB()
        self.basic_place = None
        self.metrics = None
        self.pos = None

    def database(self, params):
        """
        @brief Data collection for placement.
        @param params parameters
        @param placedb placement database
        """
        assert (not params.gpu) or configure.compile_configurations["CUDA_FOUND"] == 'TRUE', \
                "CANNOT enable GPU without CUDA compiled"

        np.random.seed(params.random_seed)
        # read database
        tt = time.time()
        self.placedb(params)
        logging.info("reading database takes %.2f seconds" %
                     (time.time() - tt))

    def init_basic_place(self, params, timer):
        self.database(params)
        self.basic_place = NonLinearPlace.NonLinearPlace(
            params, self.placedb, timer)

    def convert_terminal_ni_format(self, params):
        """
        ntuplace3 cannot recognize terminal_NI, so we need to convert it to terminal
        change size.x size.y to 0 0
        """
        nodes_files = []
        pl_files = []
        for root, dirs, files in os.walk(os.path.dirname(params.aux_input)):
            for file in files:
                if file.endswith('.nodes'):
                    nodes_files.append(os.path.join(root, file))
                if file.endswith('.pl'):
                    pl_files.append(os.path.join(root, file))

        logger = logging.getLogger(__name__)
        logger.info(f"find {len(nodes_files)} .nodes files to convert")

        for nodes_file in nodes_files:
            try:
                with open(nodes_file, 'r', encoding='utf-8') as f:
                    content = f.read()

                # replace terminal_NI to terminal, and change size.x size.y to 0 0
                # pattern: size.x size.y terminal_NI
                pattern = r'(\d+\.?\d*)\s+(\d+\.?\d*)\s+terminal_NI'
                replacement = '0 0 terminal'

                new_content = re.sub(pattern, replacement, content)

                if new_content != content:
                    with open(nodes_file, 'w', encoding='utf-8') as f:
                        f.write(new_content)

                    matches = re.findall(pattern, content)
                    logger.info(
                        f"{nodes_file} converted, replaced {len(matches)} terminal_NI"
                    )
                else:
                    logger.info(f"{nodes_file} no need to convert")

            except Exception as e:
                logger.error(f"error when converting {nodes_file}: {str(e)}")

        for pl_file in pl_files:
            try:
                with open(pl_file, 'r', encoding='utf-8') as f:
                    content = f.read()

                pattern = 'FIXED_NI'
                replacement = 'FIXED'
                new_content = re.sub(pattern, replacement, content)

                if new_content != content:
                    with open(pl_file, 'w', encoding='utf-8') as f:
                        f.write(new_content)

                    matches = re.findall(pattern, content)
                    logger.info(
                        f"{pl_file} converted, replaced {len(matches)} FIXED_NI"
                    )
                else:
                    logger.info(f"{pl_file} no need to convert")

            except Exception as e:
                logger.error(f"error when converting {pl_file}: {str(e)}")

    def place(self, params, timer):
        """
        @brief Top API to run the entire placement flow.
        @param params parameters
        """
        # solve placement
        tt = time.time()
        placer = NonLinearPlace.NonLinearPlace(params, self.placedb, timer)
        logging.info("non-linear placement initialization takes %.2f seconds" %
                     (time.time() - tt))
        global_place_stages = getattr(params, "global_place_stages", [])
        learning_rate_value = 0.01
        if global_place_stages:
            learning_rate_value = global_place_stages[0].get("learning_rate", learning_rate_value)
        metrics = placer(params, self.placedb, learning_rate_value)
        logging.info("non-linear placement takes %.2f seconds" %
                     (time.time() - tt))

        # write placement solution
        path = "%s/%s" % (params.result_dir, params.design_name())
        if not os.path.exists(path):
            os.system("mkdir -p %s" % (path))
        gp_out_file = os.path.join(
            path,
            "%s.gp.%s" % (params.design_name(), params.solution_file_suffix()))
        self.placedb.write(params, gp_out_file)

        # call external detailed placement
        # TODO: support more external placers, currently only support
        # 1. NTUplace3/NTUplace4h with Bookshelf format
        # 2. NTUplace_4dr with LEF/DEF format
        if params.ntuplace_flag and params.detailed_place_engine and os.path.exists(
                params.detailed_place_engine):
            logging.info("Use external detailed placement engine %s" %
                         (params.detailed_place_engine))
            self.convert_terminal_ni_format(params)

            if params.solution_file_suffix() == "pl" and any(
                    dp_engine in params.detailed_place_engine
                    for dp_engine in ['ntuplace3', 'ntuplace4h']):
                dp_out_file = gp_out_file.replace(".gp.pl", "")
                # add target density constraint if provided
                target_density_cmd = ""
                if params.target_density < 1.0 and not params.routability_opt_flag:
                    target_density_cmd = " -util %f" % (params.target_density)
                cmd = "%s -aux %s -loadpl %s %s -out %s -noglobal %s" % (
                    params.detailed_place_engine, params.aux_input,
                    gp_out_file, target_density_cmd, dp_out_file,
                    params.detailed_place_command)
                logging.info("%s" % (cmd))
                tt = time.time()
                os.system(cmd)
                logging.info("External detailed placement takes %.2f seconds" %
                             (time.time() - tt))

                if params.plot_flag:
                    # read solution and evaluate
                    self.placedb.read_pl(params, dp_out_file + ".ntup.pl")
                    iteration = len(metrics)
                    pos = placer.init_pos
                    pos[0:self.placedb.
                        num_physical_nodes] = self.placedb.node_x
                    pos[self.placedb.num_nodes:self.placedb.num_nodes +
                        self.placedb.num_physical_nodes] = self.placedb.node_y
                    # hpwl, density_overflow, max_density = placer.validate(
                    #     placedb, pos, iteration)
                    # logging.info(
                    #     "iteration %4d, HPWL %.3E, overflow %.3E, max density %.3E"
                    #     % (iteration, hpwl, density_overflow, max_density))
                    placer.plot(params, self.placedb, iteration, pos)
            elif 'ntuplace_4dr' in params.detailed_place_engine:
                dp_out_file = gp_out_file.replace(".gp.def", "")
                cmd = "%s" % (params.detailed_place_engine)
                for lef in params.lef_input:
                    if "tech.lef" in lef:
                        cmd += " -tech_lef %s" % (lef)
                    else:
                        cmd += " -cell_lef %s" % (lef)
                    benchmark_dir = os.path.dirname(lef)
                cmd += " -floorplan_def %s" % (gp_out_file)
                if (params.verilog_input):
                    cmd += " -verilog %s" % (params.verilog_input)
                cmd += " -out ntuplace_4dr_out"
                cmd += " -placement_constraints %s/placement.constraints" % (
                    # os.path.dirname(params.verilog_input))
                    benchmark_dir)
                cmd += " -noglobal %s ; " % (params.detailed_place_command)
                # cmd += " %s ; " % (params.detailed_place_command) ## test whole flow
                cmd += "mv ntuplace_4dr_out.fence.plt %s.fence.plt ; " % (
                    dp_out_file)
                cmd += "mv ntuplace_4dr_out.init.plt %s.init.plt ; " % (
                    dp_out_file)
                cmd += "mv ntuplace_4dr_out %s.ntup.def ; " % (dp_out_file)
                cmd += "mv ntuplace_4dr_out.ntup.overflow.plt %s.ntup.overflow.plt ; " % (
                    dp_out_file)
                cmd += "mv ntuplace_4dr_out.ntup.plt %s.ntup.plt ; " % (
                    dp_out_file)
                if os.path.exists("%s/dat" % (os.path.dirname(dp_out_file))):
                    cmd += "rm -r %s/dat ; " % (os.path.dirname(dp_out_file))
                cmd += "mv dat %s/ ; " % (os.path.dirname(dp_out_file))
                logging.info("%s" % (cmd))
                tt = time.time()
                os.system(cmd)
                logging.info("External detailed placement takes %.2f seconds" %
                             (time.time() - tt))

            else:
                logging.warning(
                    "External detailed placement only supports NTUplace3/NTUplace4dr API"
                )
        self.metrics = metrics
        self.pos = placer.pos[0]


class DreamplaceBaseCollection:

    def __init__(self, num_tiers):
        self.num_tiers = num_tiers
        self.dp_2d = DreamplaceBase()
        self.dp_tier = [DreamplaceBase() for i in range(num_tiers)]
        self.dp_terminal = DreamplaceBase()

    def init_all_basic_place(self, d2d_params, timer):
        self.dp_2d.init_basic_place(d2d_params.flatten_2d, timer)
        # save each tier's placedb for backup
        for i in range(self.num_tiers):
            self.dp_tier[i].init_basic_place(d2d_params.flattened_tier[i],
                                             timer)

    def reload_die_basic_place(self, d2d_params, timer):
        for i in range(self.num_tiers):
            # update placedb_tier & data_tier using new terminal_insert result
            self.dp_tier[i].init_basic_place(d2d_params.partition_tier[i],
                                             timer)
