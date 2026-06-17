'''
Author: JeanneWillis hi@jeannewillis.cn
Date: 2025-07-19 17:57:28
LastEditors: JeanneWillis hi@jeannewillis.cn
LastEditTime: 2026-03-11 19:30:56
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
from placer.ops.draw_block.draw_block import DrawBlock

from placer.op_wrapper import OpWrapper
from placer.d2d_params import D2DParams
from placer.tools.thirdparty_api.dreamplace_base import DreamplaceBaseCollection
from placer.tools.thirdparty_api.specpart_base import SpecPartBase
from placer.tools.thirdparty_api.tritonpart_base import TritonPartBase

from placer.constants import Format, Orient
import torch
import matplotlib.pyplot as plt

from placer.tools.d2d_result_analyzer.d2d_result_analyzer import D2DResultAnalyzer
from placer.tools.diff_d2d_wirelength import D2DCoPlace
from placer.tools.openroad_3d_export import export_openroad_3d_inputs
import dreamplace.NesterovAcceleratedGradientOptimizer as NesterovOpt


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
            terminal_names = self.dreamplace.dp_terminal.placedb.node_names[
                :self.num_terminal_NIs]
            hpwl_d2d = self.op_wrapper.d2d_op_collections.hpwl_d2d_op(
                self.dreamplace.dp_2d.pos, self.cut_net_mask, self.tier,
                self.dreamplace.dp_terminal.pos, self.num_terminal_NIs,
                terminal_names)
        else:
            hpwl_d2d = self.op_wrapper.d2d_op_collections.hpwl_d2d_op(
                self.dreamplace.dp_2d.pos, self.cut_net_mask, self.tier)
        if all(dp.pos is not None for dp in self.dreamplace.dp_tier):
            tier_hpwl = sum(
                float(dp.basic_place.op_collections.hpwl_op(dp.pos))
                for dp in self.dreamplace.dp_tier)
            ratio = float(hpwl_d2d) / tier_hpwl if tier_hpwl > 0 else 0.0
            logger.info("HPWL_D2D_CHECK: tier_hpwl_sum=%.6f ratio=%.6f",
                        tier_hpwl, ratio)
        logger.info("HPWL_D2D:%.6f " % (hpwl_d2d))

        return hpwl_d2d

    def init_spec(self):
        if self.params.flatten_2d.txt_input:
            logging.info("parsing iccad txt input......")
            # parser iccad txt format to die_spec
            parser_txt = ParserTxt(self.params.flatten_2d.txt_input)
            self.die_spec = parser_txt()

        # control numpy multithreading
        os.environ["OMP_NUM_THREADS"] = "%d" % (
            self.params.flatten_2d.num_threads)

        self.dreamplace.init_all_basic_place(self.params, self.timer)
        logging.info(
            "Thread config effective: torch.get_num_threads()=%d, "
            "OMP_NUM_THREADS=%s, params.num_threads=%s, flatten_2d.num_threads=%s",
            torch.get_num_threads(),
            os.environ.get("OMP_NUM_THREADS", "unset"),
            str(getattr(self.params, "num_threads", "unset")),
            str(getattr(self.params.flatten_2d, "num_threads", "unset")))
        self.node_orient = [Orient.N.name
                            ] * self.dreamplace.dp_2d.placedb.num_movable_nodes

        self.cut_net_mask = torch.zeros(self.dreamplace.dp_2d.placedb.num_nets,
                                        dtype=torch.int32)
        self.tier = torch.zeros(
            self.dreamplace.dp_2d.placedb.num_movable_nodes, dtype=torch.int32)

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

        self.op_wrapper.d2d_op_collections.pos_flattened_op(
            self.tier, self.dreamplace.dp_2d.pos,
            [self.dreamplace.dp_tier[i].pos for i in range(self.num_tiers)])

        return self.hpwl_d2d(logger)

    def die_terminal_co_place(self,
                              global_place_flag=True,
                              legalize_flag=False,
                              detailed_place_flag=False,
                              random_center_init_flag=True,
                              ntuplace_flag=False,
                              logger=logging):
        """
        Coplace-style 2.5D global placement:
        simultaneously optimize top tier cells, bottom tier cells, and D2D
        terminal_NIs under a single Nesterov optimizer.

        Each placedb is wrapped by a dreamplace ``PlaceObj``, so its
        ``obj_and_grad_fn`` already integrates wirelength + density penalty
        with the diagonal preconditioner; the outer loop also drives each
        layer's adaptive ``update_density_weight_op`` and ``update_gamma_op``.

        A single ``mov_node_pos_all = [top_cell | bot_cell | terminal]`` is
        shared by the three placedbs; per-layer preconditioned grads are
        scattered back to it before the Nesterov step.

        Prerequisites: ``partition()`` and ``terminal_insert()`` must have
        been called so that ``dp_tier[0/1]`` and ``dp_terminal`` are properly
        built and contain their per-tier net topology with D2D terminal_NIs
        in place.
        """
        
        self.params.set_die_place_flags(
            global_place_flag=global_place_flag,
            legalize_flag=legalize_flag,
            detailed_place_flag=detailed_place_flag,
            random_center_init_flag=False,
            ntuplace_flag=ntuplace_flag)
        self.dreamplace.reload_die_basic_place(self.params, self.timer)

        co = D2DCoPlace(self)
        mov_node_pos_all = co.pack_initial()

        # ----- Per-layer PlaceObj initialization (density_weight + gamma) -----
        co.initialize_models(mov_node_pos_all)

        target_overflow = float(self.params.co_place_stop_overflow)
        max_iter = int(self.params.co_place_iteration)

        obj_and_grad_fn = co.make_obj_and_grad_fn()

        # ----- Estimate initial learning rate via dreamplace's heuristic. -----
        with torch.enable_grad():
            obj0, g0 = obj_and_grad_fn(mov_node_pos_all)
            x_minus = mov_node_pos_all.detach().clone() \
                - self.params.co_place_lr * g0.detach()
            x_minus.requires_grad_(True)
            obj1, g1 = obj_and_grad_fn(x_minus)
            num = (mov_node_pos_all.detach() - x_minus.detach()).norm(p=2)
            den = (g0.detach() - g1.detach()).norm(p=2)
            init_lr = (num / den).item() if den.item() > 1e-12 else \
                self.params.co_place_lr
        logger.info("co-place init learning rate %.3E", init_lr)

        # mov_node_pos_all has been written into via .data; reset its grad.
        if mov_node_pos_all.grad is not None:
            mov_node_pos_all.grad.zero_()

        optimizer = NesterovOpt.NesterovAcceleratedGradientOptimizer(
            [mov_node_pos_all],
            lr=init_lr,
            obj_and_grad_fn=obj_and_grad_fn,
            constraint_fn=co.constraint_fn,
            use_bb=True)

        # ----- Main loop: nesterov + adaptive density_weight + gamma -----
        # Track best solution by *normalized* overflow (sum of three layers).
        best_overflow = float("inf")
        best_pos = None
        log_freq = 20
        plot_freq = int(self.params.co_place_plot_freq)

        # Initial layout (iter 0) before any optimization step.
        if plot_freq > 0:
            co.plot(mov_node_pos_all, 0)

        prev_metrics = None
        for it in range(max_iter):
            # 1) Eval cur per-layer metrics (hpwl + normalized overflow)
            cur_metrics = co.evaluate_metrics(mov_node_pos_all, it)
            cur_top, cur_bot, cur_term = cur_metrics
            ov_top = float(cur_top.overflow.mean().item())
            ov_bot = float(cur_bot.overflow.mean().item())
            ov_term = float(cur_term.overflow.mean().item())
            ov_sum = ov_top + ov_bot + ov_term
            # Terminal overflow can legitimately be 0 from the start when
            # terminals are pre-placed at intersection centers; exclude it from
            # the convergence check so die cells are still fully optimized.
            ov_max = max(ov_top, ov_bot, ov_term)
            ov_die_max = max(ov_top, ov_bot)

            # Best-solution book-keeping (use sum to avoid one layer dominating).
            if ov_sum < best_overflow:
                best_overflow = ov_sum
                best_pos = mov_node_pos_all.detach().clone()

            if it % log_freq == 0 or it == max_iter - 1:
                d2d_hpwl = float(co.evaluate_d2d_hpwl(mov_node_pos_all))
                logger.info(
                    "co-place iter %4d | overflow (top, bot, term) = "
                    "(%.4f, %.4f, %.4f) | dens_w (%.3E, %.3E, %.3E) | "
                    "gamma (%.3E, %.3E, %.3E) | local hpwl "
                    "(%.3E, %.3E, %.3E) | d2d_hpwl %.3E", it,
                    ov_top, ov_bot, ov_term,
                    co.model_top.density_weight.mean().item(),
                    co.model_bot.density_weight.mean().item(),
                    co.model_term.density_weight.mean().item(),
                    co.model_top.gamma.item(), co.model_bot.gamma.item(),
                    co.model_term.gamma.item(),
                    float(cur_top.hpwl.item()), float(cur_bot.hpwl.item()),
                    float(cur_term.hpwl.item()), d2d_hpwl)

            if plot_freq > 0 and ((it + 1) % 500 == 0
                                  or it == max_iter - 1):
                co.plot(mov_node_pos_all, it + 1)

            # Early stop when die layers are converged (terminal overflow excluded:
            # it can be 0 from the start when pre-placed at intersection centers).
            if ov_die_max < target_overflow:
                logger.info(
                    "co-place early stop at iter %d (die max overflow %.4f < %.4f)",
                    it, ov_die_max, target_overflow)
                if plot_freq > 0:
                    co.plot(mov_node_pos_all, it + 1)
                break

            # 2) Nesterov line-search step (calls obj_and_grad_fn internally).
            optimizer.step()

            # 3) Adaptive density-weight update (HPWL-based, per layer).
            if prev_metrics is not None:
                co.update_density_weights(cur_metrics, prev_metrics, it + 1)

            # 4) Gamma annealing (per layer, based on its own overflow).
            co.update_gammas(it, cur_metrics)

            prev_metrics = cur_metrics

        # ----- Roll back to best solution if available. -----
        if best_pos is not None:
            mov_node_pos_all.data.copy_(best_pos)
            logger.info("co-place: rolled back to best overflow %.4f",
                        best_overflow)
            if plot_freq > 0:
                co.plot(mov_node_pos_all, max_iter + 1)

        # Write final positions back to placedbs.
        co.writeback(mov_node_pos_all)

        # Immediately free the three PlaceObj GPU models: they are the bulk of
        # the ~80 GB co-place footprint and are no longer needed after writeback.
        # Without explicit deletion the tensors stay in PyTorch's allocator cache
        # and leave no room for the terminal placement in refinement().
        del co.model_top, co.model_bot, co.model_term
        del optimizer, mov_node_pos_all, best_pos

        # Persist co-place tier positions to partition pl files so that the
        # subsequent die_by_die_place(global_place_flag=False, ntuplace=True)
        # step starts from the co-place result instead of the raw partition
        # positions.  (reload_die_basic_place re-reads the partition pl files
        # from disk; without this write-back the co-place result would be
        # silently discarded.)
        for i in range(self.num_tiers):
            dp_t = self.dreamplace.dp_tier[i]
            pos_t = dp_t.basic_place.data_collections.pos[0]
            n_phys = dp_t.placedb.num_physical_nodes
            N_t = dp_t.placedb.num_nodes
            node_x_out = pos_t[:n_phys].detach().cpu().numpy()
            node_y_out = pos_t[N_t:N_t + n_phys].detach().cpu().numpy()
            partition_pl = self.params.partition_tier[i].aux_input.replace(
                ".aux", ".pl")
            dp_t.placedb.write_pl(self.params.partition_tier[i], partition_pl,
                                  node_x_out, node_y_out)
            logger.info(
                "co-place: wrote tier%d co-place positions to %s", i,
                partition_pl)

        # Persist terminal layer positions (movable D2D terminals only).
        dp_term = self.dreamplace.dp_terminal
        pos_term = dp_term.basic_place.data_collections.pos[0]
        n_phys_term = dp_term.placedb.num_physical_nodes
        N_term = dp_term.placedb.num_nodes
        term_x_out = pos_term[:n_phys_term].detach().cpu().numpy()
        term_y_out = pos_term[N_term:N_term + n_phys_term].detach().cpu().numpy()
        terminal_pl = self.params.terminal.aux_input.replace(".aux", ".pl")
        dp_term.placedb.write_pl(self.params.terminal, terminal_pl, term_x_out,
                                 term_y_out)
        logger.info("co-place: wrote terminal co-place positions to %s",
                    terminal_pl)

        # Sync dp_2d.pos so HPWL_D2D / downstream stages see the latest result.
        self.op_wrapper.d2d_op_collections.pos_flattened_op(
            self.tier, self.dreamplace.dp_2d.pos,
            [self.dreamplace.dp_tier[i].pos for i in range(self.num_tiers)])

        return self.hpwl_d2d(logger)

    def flatten_2d_place(self):
        self.dreamplace.dp_2d.place(self.params.flatten_2d, self.timer)

    def partition(self, logger=logging):

        # temporarily call tier result from file
        self.tier = self.op_wrapper.d2d_op_collections.partition_flow_op(
            partitioner="hmetis", logger=logger)
        torch.save(self.tier, self.params.result_dir_root + "/tier.pt")

        # return partition result but not receive now
        # pos_2d/2 beceuse of 3d-placer set flattened_die size as die_size*2
        # self.cut_net_mask = self.op_wrapper.d2d_op_collections.init_partition_op(
        #     self.tier, self.dreamplace.dp_2d.pos / 2, self.node_orient)
        self.cut_net_mask = self.op_wrapper.d2d_op_collections.terminal_insert_op(
            self.tier,  self.dreamplace.dp_2d.pos * 2**0.5 /2 - (2**0.5 - 1) * self.die_spec.dieSizeX / 2, self.node_orient)

        if self.format == Format.ICCAD2023:
            # update placedb_tier & data_tier using new terminal_insert result
            self.dreamplace.reload_die_basic_place(self.params, self.timer)

            self.tier = self.op_wrapper.d2d_op_collections.macro_balance_op(
                self.tier, self.node_orient)
            self.cut_net_mask = self.op_wrapper.d2d_op_collections.terminal_insert_op(
                self.tier, self.dreamplace.dp_2d.pos / 2, self.node_orient)

        self.num_terminal_NIs = int(self.cut_net_mask.sum().item())
        filename = self.params.result_dir_root + "/final-partition-block.png"
        self.op_wrapper.d2d_op_collections.draw_block_op(
            self.dreamplace.dp_2d.pos, filename, self.tier)

    def export_openroad_3d_inputs(self, logger=logging):
        return export_openroad_3d_inputs(self.params,
                                         self.dreamplace.dp_2d.placedb,
                                         self.dreamplace.dp_2d.pos, self.tier,
                                         logger)

    def terminal_insert(self):
        # create terminal aux for collaborative optimization by tier[0]
        self.op_wrapper.d2d_op_collections.terminal_insert_aux_op(
            self.tier, self.dreamplace.dp_2d.pos)

        self.dreamplace.dp_terminal.init_basic_place(self.params.terminal,
                                                     self.timer)
        # self.dreamplace.dp_terminal.database(self.params.terminal)
        self.dreamplace.dp_terminal.place(self.params.terminal, self.timer)

        self.op_wrapper.d2d_op_collections.terminal_legalize_op(
            self.tier, self.dreamplace.dp_2d.pos,
            self.dreamplace.dp_terminal.pos, self.num_terminal_NIs,
            self.dreamplace.dp_terminal.placedb.node_names, self.node_orient)

    def terminal_legalize(self):
        # create terminal aux for collaborative optimization by tier[0]
        self.op_wrapper.d2d_op_collections.terminal_legalize_aux_op(
            self.tier, self.dreamplace.dp_2d.pos,
            self.dreamplace.dp_terminal.pos, self.num_terminal_NIs,
            self.dreamplace.dp_terminal.placedb.node_names)

        self.params.terminal.global_place_flag = False
        self.dreamplace.dp_terminal.init_basic_place(self.params.terminal,
                                                     self.timer)
        # self.dreamplace.dp_terminal.database(self.params.terminal)
        self.dreamplace.dp_terminal.place(self.params.terminal, self.timer)
        self.params.terminal.global_place_flag = True

        self.op_wrapper.d2d_op_collections.terminal_legalize_op(
            self.tier, self.dreamplace.dp_2d.pos,
            self.dreamplace.dp_terminal.pos, self.num_terminal_NIs,
            self.dreamplace.dp_terminal.placedb.node_names, self.node_orient)

    def refinement(self):
        terminal_names = self.dreamplace.dp_terminal.placedb.node_names[
            :self.num_terminal_NIs]
        self.tier = self.op_wrapper.d2d_op_collections.bin_based_fm_op(
            self.tier, self.dreamplace.dp_2d.pos,
            self.dreamplace.dp_terminal.pos, self.num_terminal_NIs,
            terminal_names)

        self.cut_net_mask = self.op_wrapper.d2d_op_collections.terminal_insert_op(
            self.tier, self.dreamplace.dp_2d.pos, self.node_orient)
        self.num_terminal_NIs = int(self.cut_net_mask.sum().item())

        # create terminal aux for collaborative optimization by tier[0]
        self.op_wrapper.d2d_op_collections.terminal_insert_aux_op(
            self.tier, self.dreamplace.dp_2d.pos)

        self.dreamplace.dp_terminal.init_basic_place(self.params.terminal,
                                                     self.timer)
        # self.dreamplace.dp_terminal.database(self.params.terminal)
        self.dreamplace.dp_terminal.place(self.params.terminal, self.timer)

        self.op_wrapper.d2d_op_collections.terminal_legalize_op(
            self.tier, self.dreamplace.dp_2d.pos,
            self.dreamplace.dp_terminal.pos, self.num_terminal_NIs,
            self.dreamplace.dp_terminal.placedb.node_names, self.node_orient)

        torch.save(self.tier,
                   self.params.result_dir_root + "/tier-refinement.pt")

    def output(self):
        self.op_wrapper.d2d_op_collections.out_fmt_iccad_op(
            self.dreamplace.dp_terminal.placedb, self.params.case_name,
            self.format, self.node_orient)

    def draw_d2d_layout(self):
        top_layout_filename = self.params.result_dir_root + "/top-layout.png"
        draw_top_block = DrawBlock(self.dreamplace.dp_tier[0].placedb)(
            self.dreamplace.dp_tier[0].pos, top_layout_filename,
            torch.zeros(self.dreamplace.dp_tier[0].placedb.num_movable_nodes))

        bottom_layout_filename = self.params.result_dir_root + "/bottom-layout.png"
        draw_bottom_block = DrawBlock(self.dreamplace.dp_tier[1].placedb)(
            self.dreamplace.dp_tier[1].pos, bottom_layout_filename,
            torch.zeros(self.dreamplace.dp_tier[1].placedb.num_movable_nodes))

        terminal_layout_filename = self.params.result_dir_root + "/terminal-layout.png"
        # track
        terminal_placedb = self.dreamplace.dp_terminal.placedb
        terminal_placedb.num_terminal_NIs += self.num_terminal_NIs
        terminal_placedb.node_size_x[:self.
                                     num_terminal_NIs] -= self.die_spec.terminalSpacing
        terminal_placedb.node_size_y[:self.
                                     num_terminal_NIs] -= self.die_spec.terminalSpacing

        terminal_placedb.node_size_x[self.num_terminal_NIs:] = 0
        terminal_placedb.node_size_y[self.num_terminal_NIs:] = 0
        draw_terminal_block = DrawBlock(terminal_placedb)(
            self.dreamplace.dp_terminal.pos +
            self.die_spec.terminalSpacing / 2, terminal_layout_filename,
            torch.zeros(self.num_terminal_NIs))

    def analyze_results(self):

        def analyze_placer_output(benchmark_file: str,
                                  result_dir: str,
                                  logger=logging):

            # Construct file paths
            analyzer = D2DResultAnalyzer(benchmark_file=benchmark_file,
                                         result_dir=result_dir,
                                         logger=logger)
            return analyzer.run_comprehensive_analysis()

        analyze_placer_output(self.params.flatten_2d.txt_input,
                              self.params.result_dir_root)

        self.op_wrapper.d2d_op_collections.draw_layout_result_op()
        self.draw_d2d_layout()


if __name__ == "__main__":
    """
    @brief main function to invoke the entire placement flow.
    """

    # placement begin
    printWelcome()
    d2d_placer = D2Dplacer(sys.argv[1])
    d2d_logger = init_log(d2d_placer.params.result_dir_root)

    d2d_placer.init_spec()
    d2d_placer.flatten_2d_place()

    d2d_placer.partition(d2d_logger)
    # breakpoint()

    if d2d_placer.params.co_place_flag:
        d2d_logger.info("=== Running coplace-style 2.5D global placement ===")
        # Warm-start the tier placedbs with a quick GP so that co-place begins
        # from a spread-out (not random-center) configuration.
        d2d_placer.die_by_die_place(global_place_flag=True,
                                    legalize_flag=False,
                                    detailed_place_flag=False,
                                    random_center_init_flag=True,
                                    ntuplace_flag=False)
        d2d_placer.terminal_insert()
        # Co-place: jointly optimise top cells, bottom cells, and terminals.
        # Writes the final positions back to dp_tier[i] (in memory) and to
        # the partition/tier{i}.pl files on disk.
        d2d_placer.die_terminal_co_place(global_place_flag=True,
                                         legalize_flag=False,
                                         detailed_place_flag=False,
                                         random_center_init_flag=False,
                                         ntuplace_flag=False,
                                         logger=d2d_logger)
        # Release co-place model GPU cache before refinement to avoid OOM:
        # die_terminal_co_place runs 3 simultaneous PlaceObj models which
        # exhaust nearly all VRAM; PyTorch retains the cache after the function
        # returns, so we must explicitly free it here.
        import gc; gc.collect()
        torch.cuda.empty_cache()
        d2d_logger.info("GPU cache cleared before refinement: %.1f GiB free",
                        torch.cuda.mem_get_info()[0] / 1024**3)
        # FM refinement: updates tier assignment and re-inserts terminals.
        # terminal_insert_op inside refinement() re-writes partition pl files
        # from dp_2d.pos (synced from co-place result), so NTUplace3 below
        # will start from the co-place positions rather than a fresh GP.
        d2d_placer.refinement()
        # Legalization + detailed placement from co-place positions.
        # global_place_flag=False skips the GP re-run;
        # random_center_init_flag=False preserves what is in partition pl files.
        d2d_placer.die_by_die_place(global_place_flag=False,
                                    legalize_flag=False,
                                    detailed_place_flag=False,
                                    random_center_init_flag=False,
                                    ntuplace_flag=True,
                                    logger=d2d_logger)
    else:
        d2d_placer.die_by_die_place(global_place_flag=True,
                                    legalize_flag=False,
                                    detailed_place_flag=False,
                                    random_center_init_flag=True,
                                    ntuplace_flag=False)

        d2d_placer.terminal_insert()
        d2d_placer.die_by_die_place(global_place_flag=True,
                                    legalize_flag=False,
                                    detailed_place_flag=False,
                                    random_center_init_flag=True,
                                    ntuplace_flag=False)

        # d2d_placer.refinement()
        d2d_placer.die_by_die_place(global_place_flag=True,
                                    legalize_flag=False,
                                    detailed_place_flag=False,
                                    random_center_init_flag=True,
                                    ntuplace_flag=True,
                                    logger=d2d_logger)

    d2d_placer.hpwl_d2d(d2d_logger)
    d2d_placer.export_openroad_3d_inputs(d2d_logger)
    d2d_placer.output()
    d2d_placer.analyze_results()
