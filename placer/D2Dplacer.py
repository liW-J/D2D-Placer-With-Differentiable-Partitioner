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
from placer.ops.parser_txt.parser_txt import ParserTxt
from placer.D2DOpWapper import D2DOpWapper
from placer.D2DParams import D2DParams

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
    d2d_params = D2DParams(sys.argv[1])
    num_tiers = d2d_params.flatten_2d.num_tiers
    printWelcome()

    # load parameters
    # parse input get flattened .aux
    if d2d_params.flatten_2d.txt_input:
        logging.info("parsing iccad txt input......")
        # parser iccad txt format to aux
        parser_txt = ParserTxt(d2d_params.flatten_2d.txt_input)
        die_spec = parser_txt()

    # control numpy multithreading
    os.environ["OMP_NUM_THREADS"] = "%d" % (d2d_params.flatten_2d.num_threads)

    # placement begin
    tt = time.time()

    placedb_2d, timer = database(d2d_params.flatten_2d)
    basic_data = BasicPlace.BasicPlace(d2d_params.flatten_2d, placedb_2d,
                                       timer)

    # dreamplace for flattened 2d placement
    logging.info("flattened 2d placement begin")
    d2d_params.flatten_2d.printWelcome()
    metrics_2d, pos_2d = place(d2d_params.flatten_2d, placedb_2d, timer)

    # save each tier's placedb for backup
    placedb_tier = []
    tier_data = []
    for i in range(num_tiers):
        placedb, timer = database(d2d_params.flattened_tier[i])
        placedb_tier.append(placedb)
        tier_data.append(
            BasicPlace.BasicPlace(d2d_params.flattened_tier[i], placedb,
                                  timer))

    # prepare for partitioning
    d2d_op_wapper = D2DOpWapper(basic_data, placedb_2d, placedb_tier,
                                d2d_params, tier_data, die_spec)

    # partition
    tier = d2d_op_wapper.d2d_op_collections.hmetis_op(pos_2d)
    # tier = avg_cut(pin_pos_op(pos_2d), node_size_x, node_size_y)
    # tier = multi_bipartition(pos_2d)

    # bin-based partition
    # temporarily call tier result from file
    # tier = torch.load('placer/die_tensor.pt')
    tier = tier.to(torch.int32)

    # return partition result but not receive now
    # cut_net_mask = d2d_op_wapper.d2d_op_collections.init_partition_op(
    #     tier, pos_2d / 2)
    # pos_2d/2 beceuse of 3d-placer set flattened_die size as die_size*2
    cut_net_mask = d2d_op_wapper.d2d_op_collections.terminal_insert_op(
        tier, pos_2d / 2)
    num_terminal_NIs = int(cut_net_mask.sum().item())
    breakpoint()

    # if 2D result for init pos may casued no convergence
    metrics_tier = []
    pos_tier = []
    for i in range(num_tiers):
        placedb_tier[i], timer = database(d2d_params.partition_tier[i])
        d2d_params.partition_tier[i].printWelcome()
        metrics, pos = place(d2d_params.partition_tier[i], placedb_tier[i],
                             timer)
        metrics_tier.append(metrics)
        pos_tier.append(pos)

    logging.info("2d placement  HPWL:%.6f " % (metrics_2d[-1].hpwl))
    for i in range(num_tiers):
        logging.info("tier %d placement  HPWL:%.6f " %
                     (i, metrics_tier[i][-1].hpwl))

    logging.info("placement takes %.3f seconds" % (time.time() - tt))
    breakpoint()
    d2d_op_wapper.d2d_op_collections.pos_flattened_op(tier, pos_2d, pos_tier)

    d2d_op_wapper.d2d_op_collections.terminal_insert_op(tier, pos_2d)

    # update placedb_tier & tier_data using new terminal_insert result
    for i in range(num_tiers):
        placedb_tier[i], timer = database(d2d_params.partition_tier[i])
        tier_data[i] = BasicPlace.BasicPlace(d2d_params.partition_tier[i],
                                             placedb_tier[i], timer)

    # create terminal aux for collaborative optimization by tier[0]
    d2d_op_wapper.d2d_op_collections.terminal_aux_op(tier, pos_2d)

    placedb_terminal, timer = database(d2d_params.terminal)
    d2d_params.terminal.printWelcome()
    terminal_metrics, terminal_pos = place(d2d_params.terminal,
                                           placedb_terminal, timer)

    d2d_op_wapper.d2d_op_collections.terminal_legalize_op(
        tier, pos_2d, terminal_pos, num_terminal_NIs,
        placedb_terminal.node_names)

    # breakpoint()

    for i in range(num_tiers):
        d2d_params.partition_tier[i].random_center_init_flag = 0
        placedb_tier[i], timer = database(d2d_params.partition_tier[i])
        d2d_params.partition_tier[i].printWelcome()
        metrics, pos = place(d2d_params.partition_tier[i], placedb_tier[i],
                             timer)
        metrics_tier[i] = metrics
        pos_tier[i] = pos
        terminal_legalize_flag = False

    for i in range(num_tiers):
        logging.info("tier %d placement  HPWL:%.6f " %
                     (i, metrics_tier[i][-1].hpwl))

    d2d_op_wapper.d2d_op_collections.pos_flattened_op(tier, pos_2d, pos_tier)

    hpwl_d2d = d2d_op_wapper.d2d_op_collections.hpwl_d2d_op(
        pos_2d, cut_net_mask, tier, terminal_pos, num_terminal_NIs,
        placedb_terminal.node_names)
    logging.info("HPWL_D2D:%.6f " % (hpwl_d2d))

    logging.info("placement takes %.3f seconds" % (time.time() - tt))

    d2d_op_wapper.d2d_op_collections.out_fmt_iccad_op(placedb_terminal,
                                                      d2d_params.case_name)

    # breakpoint()
