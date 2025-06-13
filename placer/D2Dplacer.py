##
# @file   Placer.py
# @author Yibo Lin
# @date   Apr 2018
# @brief  Main file to run the entire placement flow.
#

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
import dreamplace.configure as configure
import dreamplace.PlaceDB as PlaceDB
import dreamplace.Params as Params
import dreamplace.Timer as Timer
import dreamplace.NonLinearPlace as NonLinearPlace
import dreamplace.BasicPlace as BasicPlace
from colorama import Fore, Style
from ops.parser_txt.parser_txt import ParserTxt
from placer.D2DOpWapper import D2DOpWapper

import torch


def database(params):
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
    placedb = PlaceDB.PlaceDB()
    placedb(params)

    # # Reference nangate generate cell power
    # power = Power()
    # power.generate_power_for_cells(placedb)

    logging.info("reading database takes %.2f seconds" % (time.time() - tt))

    # Read timing constraints provided in the benchmarks into out timing analysis
    # engine and then pass the timer into the placement core.
    timer = None
    if params.timing_opt_flag:
        tt = time.time()
        timer = Timer.Timer()
        timer(params, placedb)
        # This must be done to explicitly execute the parser builders.
        # The parsers in OpenTimer are all in lazy mode.
        timer.update_timing()
        logging.info("reading timer takes %.2f seconds" % (time.time() - tt))

    return placedb, timer


def place(params, placedb, timer):
    """
    @brief Top API to run the entire placement flow.
    @param params parameters
    """

    # solve placement
    tt = time.time()
    placer = NonLinearPlace.NonLinearPlace(params, placedb, timer)
    logging.info("non-linear placement initialization takes %.2f seconds" %
                 (time.time() - tt))
    metrics = placer(params, placedb)
    logging.info("non-linear placement takes %.2f seconds" %
                 (time.time() - tt))

    # write placement solution
    path = "%s/%s" % (params.result_dir, params.design_name())
    if not os.path.exists(path):
        os.system("mkdir -p %s" % (path))
    gp_out_file = os.path.join(
        path,
        "%s.gp.%s" % (params.design_name(), params.solution_file_suffix()))
    placedb.write(params, gp_out_file)

    # call external detailed placement
    # TODO: support more external placers, currently only support
    # 1. NTUplace3/NTUplace4h with Bookshelf format
    # 2. NTUplace_4dr with LEF/DEF format
    if params.detailed_place_engine and os.path.exists(
            params.detailed_place_engine):
        logging.info("Use external detailed placement engine %s" %
                     (params.detailed_place_engine))
        if params.solution_file_suffix() == "pl" and any(
                dp_engine in params.detailed_place_engine
                for dp_engine in ['ntuplace3', 'ntuplace4h']):
            dp_out_file = gp_out_file.replace(".gp.pl", "")
            # add target density constraint if provided
            target_density_cmd = ""
            if params.target_density < 1.0 and not params.routability_opt_flag:
                target_density_cmd = " -util %f" % (params.target_density)
            cmd = "%s -aux %s -loadpl %s %s -out %s -noglobal %s" % (
                params.detailed_place_engine, params.aux_input, gp_out_file,
                target_density_cmd, dp_out_file, params.detailed_place_command)
            logging.info("%s" % (cmd))
            tt = time.time()
            os.system(cmd)
            logging.info("External detailed placement takes %.2f seconds" %
                         (time.time() - tt))

            if params.plot_flag:
                # read solution and evaluate
                placedb.read_pl(params, dp_out_file + ".ntup.pl")
                iteration = len(metrics)
                pos = placer.init_pos
                pos[0:placedb.num_physical_nodes] = placedb.node_x
                pos[placedb.num_nodes:placedb.num_nodes +
                    placedb.num_physical_nodes] = placedb.node_y
                hpwl, density_overflow, max_density = placer.validate(
                    placedb, pos, iteration)
                logging.info(
                    "iteration %4d, HPWL %.3E, overflow %.3E, max density %.3E"
                    % (iteration, hpwl, density_overflow, max_density))
                placer.plot(params, placedb, iteration, pos)
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
    elif params.detailed_place_engine:
        logging.warning(
            "External detailed placement engine %s or aux file NOT found" %
            (params.detailed_place_engine))

    return metrics, placer.pos[0]


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


if __name__ == "__main__":
    """
    @brief main function to invoke the entire placement flow.
    """
    logging.root.name = 'D2Dplacer'
    logging.basicConfig(level=logging.INFO,
                        format='[%(levelname)-7s] %(name)s - %(message)s',
                        stream=sys.stdout)
    params = Params.Params()
    printWelcome()

    # load parameters
    params.load(sys.argv[1])
    logging.info("parameters = %s" % (params))

    # parse input get flattened .aux
    if params.txt_input:
        logging.info("parsing iccad txt input......")
        # parser iccad txt format to aux
        parser_txt = ParserTxt(params.txt_input)
        result = parser_txt()

    # control numpy multithreading
    os.environ["OMP_NUM_THREADS"] = "%d" % (params.num_threads)

    # placement begin
    tt = time.time()

    # TODO: set dir_path by case_name
    params.aux_input = "run_tmp/case_demo/flattened-2d/flattened-2d.aux"
    placedb_2d, timer = database(params)
    basic_data = BasicPlace.BasicPlace(params, placedb_2d, timer)

    # dreamplace for flattened 2d placement
    logging.info("flattened 2d placement begin")
    params.printWelcome()
    metrics_2d, pos_2d = place(params, placedb_2d, timer)

    # save each tier's placedb for backup
    placedb_tier = []
    tier_data = []
    for i in range(params.num_tiers):
        params.aux_input = f"run_tmp/case_demo/flattened-2d/tier{i}.aux"
        placedb, timer = database(params)
        placedb_tier.append(placedb)
        tier_data.append(BasicPlace.BasicPlace(params, placedb, timer))

    # prepare for partitioning
    d2d_op_wapper = D2DOpWapper(basic_data, placedb_2d, placedb_tier, params,
                                tier_data)

    # partition
    tier = d2d_op_wapper.d2d_op_collections.hmetis_op(pos_2d)
    # tier = avg_cut(pin_pos_op(pos_2d), node_size_x, node_size_y)
    # tier = multi_bipartition(pos_2d)
    # breakpoint()

    # bin-based partition
    # temporarily call tier result from file
    # tier = torch.load('placer/die_tensor.pt')
    tier = tier.to(torch.int32)

    # return partition result but not receive now
    # cut_net_mask = init_partition(tier,
    #                               pos_2d=pos_2d / 2)
    # pos_2d/2 beceuse of 3d-placer set flattened_die size as die_size*2
    cut_net_mask = d2d_op_wapper.d2d_op_collections.terminal_insert_op(
        tier,
        d2d_op_wapper.d2d_op_collections.pin_pos_op(pos_2d) / 2,
        pos_2d=pos_2d / 2)
    num_terminal_NIs = int(cut_net_mask.sum().item())
    breakpoint()
    # 2D result for init pos may casued no convergence
    # params.random_center_init_flag = 0  # 2D pos gives initial placement result
    metrics_tier = []
    pos_tier = []
    for i in range(params.num_tiers):
        params.aux_input = f"run_tmp/case_demo/partition/tier{i}.aux"
        placedb_tier[i], timer = database(params)
        params.printWelcome()
        metrics, pos = place(params, placedb_tier[i], timer)
        metrics_tier.append(metrics)
        pos_tier.append(pos)

    logging.info("2d placement  HPWL:%.6f " % (metrics_2d[-1].hpwl))
    for i in range(params.num_tiers):
        logging.info("tier %d placement  HPWL:%.6f " %
                     (i, metrics_tier[i][-1].hpwl))

    logging.info("placement takes %.3f seconds" % (time.time() - tt))
    breakpoint()
    d2d_op_wapper.d2d_op_collections.pos_flattened_op(tier, pos_2d, pos_tier)

    d2d_op_wapper.d2d_op_collections.terminal_insert_op(
        tier,
        d2d_op_wapper.d2d_op_collections.pin_pos_op(pos_2d),
        pos_2d=pos_2d)

    # update placedb_tier & tier_data using new terminal_insert result
    for i in range(params.num_tiers):
        params.aux_input = f"run_tmp/case_demo/partition/tier{i}.aux"
        placedb_tier[i], timer = database(params)
        tier_data[i] = BasicPlace.BasicPlace(params, placedb_tier[i], timer)

    # create terminal aux for collaborative optimization by tier[0]
    d2d_op_wapper.d2d_op_collections.terminal_aux_op(
        tier, d2d_op_wapper.d2d_op_collections.pin_pos_op(pos_2d), pos_2d)

    params.random_center_init_flag = 0
    params.aux_input = f"run_tmp/case_demo/terminal/terminal.aux"
    placedb_terminal, timer = database(params)
    params.printWelcome()
    terminal_metrics, terminal_pos = place(params, placedb_terminal, timer)
    # breakpoint()

    d2d_op_wapper.d2d_op_collections.terminal_legalize_op(
        tier, d2d_op_wapper.d2d_op_collections.pin_pos_op(pos_2d),
        terminal_pos, num_terminal_NIs, pos_2d, placedb_terminal.node_names)

    # breakpoint()

    for i in range(params.num_tiers):
        params.aux_input = f"run_tmp/case_demo/partition/tier{i}.aux"
        placedb_tier[i], timer = database(params)
        params.printWelcome()
        metrics, pos = place(params, placedb_tier[i], timer)
        metrics_tier[i] = metrics
        pos_tier[i] = pos
        terminal_legalize_flag = False

    for i in range(params.num_tiers):
        logging.info("tier %d placement  HPWL:%.6f " %
                     (i, metrics_tier[i][-1].hpwl))

    logging.info("placement takes %.3f seconds" % (time.time() - tt))

    d2d_op_wapper.d2d_op_collections.out_fmt_iccad_op("case_demo")

    # breakpoint()
