'''
Date: 2025-06-13 15:35:55
LastEditTime: 2025-10-21 00:05:08
FilePath: /D2D-placer/placer/op_wrapper.py
Description:
'''
import configure
from placer.ops.multi_bipartition.multi_bipartition import MultiBipartition
from placer.ops.partition_aux.partition_aux import PartitionAux
from placer.ops.avg_cut.avg_cut import AvgCut
from placer.ops.terminal_aux.terminal_aux import TerminalAux
from placer.ops.refinement.refinement import Refinement
from placer.ops.hpwl_d2d.hpwl_d2d import HPWLD2D
from placer.ops.macro_balance.macro_balance import MacroBalance
from placer.ops.bin_based_fm.bin_based_fm import BinBasedFM
from placer.ops.draw_layout_result.draw_layout_result import DrawLayoutResult
from placer.ops.draw_block.draw_block import DrawBlock

from placer.tools.out_fmt_iccad import OutfmtICCAD
from placer.tools.pos_flattened import PosFlattened
from placer.tools.graph_cutsize import GraphCutsize

from placer.tools.partitioner_manager import PartitionerManager
from dreamplace.ops.pin_pos.pin_pos import PinPos

from placer.tools.thirdparty_api.tritonpart_base import TritonPartBase
from placer.tools.thirdparty_api.differentiable_3d_partitioner_base import \
    Differentiable3DPartitionerBase

import torch
import numpy as np
import logging
import os


class D2DOpCollection(object):

    def __init__(self, hmetis_op, multi_bipartition_op, init_partition_op,
                 out_fmt_iccad_op, pos_flattened_op, terminal_insert_op,
                 pin_pos_op, pin_pos_tier_op, pin_pos_terminal_op,
                 terminal_legalize_op, avg_cut_op, terminal_insert_aux_op,
                 terminal_legalize_aux_op, refinement_op, hpwl_d2d_op,
                 macro_balance_op, parts_reader_op, hgr_generator_op,
                 bin_based_fm_op, draw_layout_result_op, draw_block_op,
                 partition_flow_op):
        self.hmetis_op = hmetis_op
        self.multi_bipartition_op = multi_bipartition_op
        self.init_partition_op = init_partition_op
        self.out_fmt_iccad_op = out_fmt_iccad_op
        self.pos_flattened_op = pos_flattened_op
        self.terminal_insert_op = terminal_insert_op
        self.pin_pos_op = pin_pos_op
        self.pin_pos_tier_op = pin_pos_tier_op
        self.pin_pos_terminal_op = pin_pos_terminal_op
        self.terminal_legalize_op = terminal_legalize_op
        self.avg_cut_op = avg_cut_op
        self.terminal_insert_aux_op = terminal_insert_aux_op
        self.terminal_legalize_aux_op = terminal_legalize_aux_op
        self.refinement_op = refinement_op
        self.hpwl_d2d_op = hpwl_d2d_op
        self.macro_balance_op = macro_balance_op
        self.parts_reader_op = parts_reader_op
        self.hgr_generator_op = hgr_generator_op
        self.bin_based_fm_op = bin_based_fm_op
        self.draw_layout_result_op = draw_layout_result_op
        self.draw_block_op = draw_block_op
        self.partition_flow_op = partition_flow_op


class OpWrapper(object):

    def __init__(self, data_2d, data_tier, d2d_params, die_spec):
        self.data_2d = data_2d
        self.data_tier = data_tier
        self.d2d_params = d2d_params
        self.die_spec = die_spec

        self.data_collections_2d = data_2d.basic_place.data_collections
        self.placedb_2d = data_2d.placedb
        self.placedb_tier = [data.placedb for data in data_tier]
        self.params = d2d_params.flatten_2d
        self.case_name = d2d_params.case_name
        self.num_tiers = d2d_params.flatten_2d.num_tiers
        self.data_collections_tier = [
            data.basic_place.data_collections for data in data_tier
        ]

        self.node_size_x = torch.stack([
            data_collection.node_size_x[:self.placedb_2d.num_movable_nodes]
            for data_collection in self.data_collections_tier
        ])
        self.node_size_y = torch.stack([
            data_collection.node_size_y[:self.placedb_2d.num_movable_nodes]
            for data_collection in self.data_collections_tier
        ])

        self.pin_offset_x = torch.stack([
            data_collections.pin_offset_x
            for data_collections in self.data_collections_tier
        ])
        self.pin_offset_y = torch.stack([
            data_collections.pin_offset_y
            for data_collections in self.data_collections_tier
        ])
        self.d2d_hpwl_net_weights = torch.ones_like(
            self.data_collections_2d.net_weights)
        logging.info(
            "D2D HPWL/FM net weights: using unity weights; flattened original "
            "range=(%.3E, %.3E)",
            float(self.data_collections_2d.net_weights.min().item()),
            float(self.data_collections_2d.net_weights.max().item()))

        self.top_die_max_util = self.die_spec.topDieMaxUtil / 100
        self.bottom_die_max_util = self.die_spec.bottomDieMaxUtil / 100

        # TXT/3d-placer describes a two-tier die with die_spec.  LEF/DEF,
        # however, enters DREAMPlace in DBU and is then shifted/scaled (usually
        # by 1/site_width).  The node sizes below already come from the scaled
        # PlaceDB, so pairing them with the raw die_spec dimensions produces a
        # Bookshelf core that is larger by scale_factor**-2 and consequently
        # trillions of bogus filler nodes.  Keep all LEF/DEF geometry in the
        # same, scaled coordinate system as node_size_{x,y}.
        if self.d2d_params.is_txt_input and self.num_tiers == 2:
            self.die_size_x = float(self.die_spec.dieSizeX)
            self.die_size_y = float(self.die_spec.dieSizeY)
            die_size_source = "TXT die_spec"
        else:
            tier_widths = np.asarray([
                float(data.placedb.xh) - float(data.placedb.xl)
                for data in self.data_tier
            ])
            tier_heights = np.asarray([
                float(data.placedb.yh) - float(data.placedb.yl)
                for data in self.data_tier
            ])
            if (tier_widths.size == 0 or tier_heights.size == 0
                    or not np.all(np.isfinite(tier_widths))
                    or not np.all(np.isfinite(tier_heights))
                    or np.any(tier_widths <= 0) or np.any(tier_heights <= 0)):
                raise ValueError(
                    "invalid scaled tier core dimensions: widths=%s heights=%s"
                    % (tier_widths.tolist(), tier_heights.tolist()))
            self.die_size_x = float(np.mean(tier_widths))
            self.die_size_y = float(np.mean(tier_heights))
            die_size_source = "scaled DREAMPlace tier core"

        logging.info(
            "D2D geometry: die_size=(%.6g, %.6g), source=%s, input_format=%s",
            self.die_size_x, self.die_size_y, die_size_source,
            self.d2d_params.input_format)

        self.row_height = [data.placedb.row_height for data in self.data_tier]
        self.partition_aux_dir = os.path.dirname(
            self.d2d_params.partition_tier[0].aux_input)
        self.terminal_aux_dir = os.path.dirname(
            self.d2d_params.terminal.aux_input)
        self._logged_d2d_pin_pos_builder = False

        self.hmetis_op = self.build_hmetis()
        self.multi_bipartition_op = self.build_multi_bipartition()
        self.init_partition_op = self.build_init_partition()
        self.out_fmt_iccad_op = self.build_out_fmt_iccad()
        self.pos_flattened_op = self.build_pos_flattened()
        self.terminal_insert_op = self.build_terminal_insert()
        self.pin_pos_op = self.build_pin_pos()
        self.pin_pos_tier_op = self.build_pin_pos_tier()
        self.pin_pos_terminal_op = self.build_pin_pos_terminal()
        self.terminal_legalize_op = self.build_terminal_legalize()
        self.avg_cut_op = self.build_avg_cut()
        self.terminal_insert_aux_op = self.build_terminal_insert_aux()
        self.terminal_legalize_aux_op = self.build_terminal_legalize_aux()
        self.refinement_op = self.build_refinement()
        self.hpwl_d2d_op = self.build_hpwl_d2d()
        self.macro_balance_op = self.build_macro_balance()
        self.parts_reader_op = self.build_parts_reader()
        self.hgr_generator_op = self.build_hgr_generator()
        self.bin_based_fm_op = self.build_bin_based_fm()
        self.draw_layout_result_op = self.build_draw_layout_result()
        self.draw_block_op = self.build_draw_block()
        self.partition_flow_op = self.build_partition_flow()

        self.d2d_op_collections = D2DOpCollection(
            hmetis_op=self.hmetis_op,
            multi_bipartition_op=self.multi_bipartition_op,
            init_partition_op=self.init_partition_op,
            out_fmt_iccad_op=self.out_fmt_iccad_op,
            pos_flattened_op=self.pos_flattened_op,
            terminal_insert_op=self.terminal_insert_op,
            pin_pos_op=self.pin_pos_op,
            pin_pos_tier_op=self.pin_pos_tier_op,
            pin_pos_terminal_op=self.pin_pos_terminal_op,
            terminal_legalize_op=self.terminal_legalize_op,
            avg_cut_op=self.avg_cut_op,
            terminal_insert_aux_op=self.terminal_insert_aux_op,
            terminal_legalize_aux_op=self.terminal_legalize_aux_op,
            refinement_op=self.refinement_op,
            hpwl_d2d_op=self.hpwl_d2d_op,
            macro_balance_op=self.macro_balance_op,
            parts_reader_op=self.parts_reader_op,
            hgr_generator_op=self.hgr_generator_op,
            bin_based_fm_op=self.bin_based_fm_op,
            draw_layout_result_op=self.draw_layout_result_op,
            draw_block_op=self.draw_block_op,
            partition_flow_op=self.partition_flow_op)

    def build_hmetis(self):

        def hmetis_op():

            self.hgr_generator_op(self.case_name)
            hg = f"{self.d2d_params.run_tmp_dir_root}/{self.case_name}.hgr"
            cmd = f"bin/hmetis {hg} 2 2 10 1 1 0 1 0"
            os.system(cmd)

            return self.parts_reader_op(self.case_name)

        return hmetis_op

    def build_multi_bipartition(self):

        multi_bipartition_op = MultiBipartition(
            self.data_collections_2d.flat_net2pin_map,
            self.data_collections_2d.flat_net2pin_start_map,
            self.data_collections_2d.pin2node_map,
            self.data_collections_2d.net_weights,
            self.data_collections_2d.net_mask_all,
            self.placedb_2d.num_movable_nodes,
            self.data_collections_2d.flat_node2pin_map,
            self.data_collections_2d.flat_node2pin_start_map,
            self.data_collections_2d.pin2net_map)

        return multi_bipartition_op

    def build_init_partition(self):

        init_partition_op = PartitionAux(
            self.data_collections_2d.flat_net2pin_map,
            self.data_collections_2d.flat_net2pin_start_map,
            self.data_collections_2d.pin2node_map,
            self.data_collections_2d.net_weights,
            self.placedb_2d.num_movable_nodes,
            self.placedb_2d.node_names,
            self.placedb_2d.net_names,
            self.node_size_x,
            self.node_size_y,
            self.pin_offset_x,
            self.pin_offset_y,
            self.die_size_x,
            self.die_size_y,
            self.row_height,
            self.die_spec.terminalSizeX,
            self.die_spec.terminalSizeY,
            self.die_spec.terminalSpacing,
            terminal_instert_flag=False,
            terminal_legalize_flag=False,
            case_name=self.case_name,
            output_dir=self.partition_aux_dir)

        def build_init_partition_op(tier, pos_2d, node_orient):
            pin_pos = self.build_d2d_pin_pos(pos_2d)

            return init_partition_op(tier, node_orient, pin_pos, pos_2d)

        return build_init_partition_op

    def build_out_fmt_iccad(self):

        out_fmt_iccad_op = OutfmtICCAD(self.data_tier, self.params,
                                       self.die_spec)

        return out_fmt_iccad_op

    def build_pos_flattened(self):

        pos_flattened_op = PosFlattened(self.params, self.data_2d,
                                        self.data_tier)

        return pos_flattened_op

    def build_terminal_insert(self):

        terminal_insert_op = PartitionAux(
            self.data_collections_2d.flat_net2pin_map,
            self.data_collections_2d.flat_net2pin_start_map,
            self.data_collections_2d.pin2node_map,
            self.data_collections_2d.net_weights,
            self.placedb_2d.num_movable_nodes,
            self.placedb_2d.node_names,
            self.placedb_2d.net_names,
            self.node_size_x,
            self.node_size_y,
            self.pin_offset_x,
            self.pin_offset_y,
            self.die_size_x,
            self.die_size_y,
            self.row_height,
            self.die_spec.terminalSizeX,
            self.die_spec.terminalSizeY,
            self.die_spec.terminalSpacing,
            terminal_instert_flag=True,
            terminal_legalize_flag=False,
            case_name=self.case_name,
            output_dir=self.partition_aux_dir)

        def build_terminal_insert_op(tier, pos_2d, node_orient):
            pin_pos = self.build_d2d_pin_pos(pos_2d)

            return terminal_insert_op(tier,
                                      node_orient,
                                      pin_pos,
                                      pos_2d=pos_2d)

        return build_terminal_insert_op

    def build_pin_pos(self):

        pin_pos_op = PinPos(
            pin_offset_x=self.data_collections_2d.pin_offset_x,
            pin_offset_y=self.data_collections_2d.pin_offset_y,
            pin2node_map=self.data_collections_2d.pin2node_map,
            flat_node2pin_map=self.data_collections_2d.flat_node2pin_map,
            flat_node2pin_start_map=self.data_collections_2d.
            flat_node2pin_start_map,
            num_physical_nodes=self.placedb_2d.num_physical_nodes,
            algorithm="node-by-node")

        return pin_pos_op

    def build_d2d_pin_pos(self, pos_2d):
        """Build tiered pin positions in flattened-2D pin order."""
        if not self._logged_d2d_pin_pos_builder:
            logging.info("D2D pin_pos: using direct flattened pin builder")
            self._logged_d2d_pin_pos_builder = True
        device = pos_2d.device
        dtype = pos_2d.dtype
        num_nodes = pos_2d.numel() // 2
        pin2node = self.data_collections_2d.pin2node_map.to(
            device=device, dtype=torch.long)
        num_pins = pin2node.numel()

        node_x = pos_2d[:num_nodes]
        node_y = pos_2d[num_nodes:]
        base_x = node_x.index_select(0, pin2node).unsqueeze(0)
        base_y = node_y.index_select(0, pin2node).unsqueeze(0)
        pin_offset_x = self.pin_offset_x[:, :num_pins].to(device=device,
                                                          dtype=dtype)
        pin_offset_y = self.pin_offset_y[:, :num_pins].to(device=device,
                                                          dtype=dtype)
        pin_pos_x = base_x + pin_offset_x
        pin_pos_y = base_y + pin_offset_y
        return torch.cat([pin_pos_x, pin_pos_y], dim=0).contiguous()

    def build_pin_pos_terminal(self):

        def build_pin_pos_terminal_op(data_collections_terminal,
                                      num_terminal_NIs):
            pin_pos_terminal_op = PinPos(
                pin_offset_x=data_collections_terminal.pin_offset_x,
                pin_offset_y=data_collections_terminal.pin_offset_y,
                pin2node_map=data_collections_terminal.pin2node_map,
                flat_node2pin_map=data_collections_terminal.flat_node2pin_map,
                flat_node2pin_start_map=data_collections_terminal.
                flat_node2pin_start_map,
                num_physical_nodes=num_terminal_NIs,
                algorithm="node-by-node")

            return pin_pos_terminal_op

        return build_pin_pos_terminal_op

    def build_pin_pos_tier(self):

        pin_pos_tier_op = []

        for tier_id in range(self.num_tiers):
            pin_pos_tier_op.append(
                PinPos(pin_offset_x=self.data_collections_tier[tier_id].
                       pin_offset_x,
                       pin_offset_y=self.data_collections_tier[tier_id].
                       pin_offset_y,
                       pin2node_map=self.data_collections_tier[tier_id].
                       pin2node_map,
                       flat_node2pin_map=self.data_collections_tier[tier_id].
                       flat_node2pin_map,
                       flat_node2pin_start_map=self.
                       data_collections_tier[tier_id].flat_node2pin_start_map,
                       num_physical_nodes=self.placedb_tier[tier_id].
                       num_physical_nodes,
                       algorithm="node-by-node"))

        return pin_pos_tier_op

    def build_terminal_legalize(self):

        terminal_legalize_op = PartitionAux(
            self.data_collections_2d.flat_net2pin_map,
            self.data_collections_2d.flat_net2pin_start_map,
            self.data_collections_2d.pin2node_map,
            self.data_collections_2d.net_weights,
            self.placedb_2d.num_movable_nodes,
            self.placedb_2d.node_names,
            self.placedb_2d.net_names,
            self.node_size_x,
            self.node_size_y,
            self.pin_offset_x,
            self.pin_offset_y,
            self.die_size_x,
            self.die_size_y,
            self.row_height,
            self.die_spec.terminalSizeX,
            self.die_spec.terminalSizeY,
            self.die_spec.terminalSpacing,
            terminal_instert_flag=True,
            terminal_legalize_flag=True,
            case_name=self.case_name,
            output_dir=self.partition_aux_dir)

        def build_terminal_legalize_op(tier, pos_2d, pos_terminal,
                                       num_terminal_NIs, terminal_names,
                                       node_orient):
            pin_pos = self.build_d2d_pin_pos(pos_2d)
            return terminal_legalize_op(tier, node_orient, pin_pos,
                                        pos_terminal, num_terminal_NIs, pos_2d,
                                        terminal_names)

        return build_terminal_legalize_op

    def build_avg_cut(self):

        avg_cut_op = AvgCut(self.data_collections_2d.flat_net2pin_map,
                            self.data_collections_2d.flat_net2pin_start_map,
                            self.data_collections_2d.pin2node_map,
                            self.data_collections_2d.net_weights,
                            self.placedb_2d.num_movable_nodes)

        def build_avg_cut_op(pos_2d):
            return avg_cut_op(self.pin_pos_op(pos_2d), self.node_size_x,
                              self.node_size_y)

        return build_avg_cut_op

    def build_terminal_insert_aux(self):

        terminal_insert_aux_op = TerminalAux(
            self.data_collections_2d.flat_net2pin_map,
            self.data_collections_2d.flat_net2pin_start_map,
            self.data_collections_2d.pin2node_map,
            self.data_collections_2d.net_weights,
            self.placedb_2d.num_movable_nodes,
            self.placedb_2d.node_names,
            self.placedb_2d.net_names,
            self.node_size_x,
            self.node_size_y,
            self.pin_offset_x,
            self.pin_offset_y,
            self.die_size_x,
            self.die_size_y,
            self.row_height,
            self.die_spec.terminalSizeX,
            self.die_spec.terminalSizeY,
            self.die_spec.terminalSpacing,
            self.case_name,
            terminal_legalize_flag=False,
            output_dir=self.terminal_aux_dir)

        def build_terminal_insert_aux_op(tier, pos_2d):
            pin_pos = self.build_d2d_pin_pos(pos_2d)
            return terminal_insert_aux_op(tier, pin_pos, pos_2d)

        return build_terminal_insert_aux_op

    def build_terminal_legalize_aux(self):

        terminal_legalize_aux_op = TerminalAux(
            self.data_collections_2d.flat_net2pin_map,
            self.data_collections_2d.flat_net2pin_start_map,
            self.data_collections_2d.pin2node_map,
            self.data_collections_2d.net_weights,
            self.placedb_2d.num_movable_nodes,
            self.placedb_2d.node_names,
            self.placedb_2d.net_names,
            self.node_size_x,
            self.node_size_y,
            self.pin_offset_x,
            self.pin_offset_y,
            self.die_size_x,
            self.die_size_y,
            self.row_height,
            self.die_spec.terminalSizeX,
            self.die_spec.terminalSizeY,
            self.die_spec.terminalSpacing,
            self.case_name,
            terminal_legalize_flag=True,
            output_dir=self.terminal_aux_dir)

        def build_terminal_legalize_aux_op(tier, pos_2d, pos_terminal,
                                           num_terminal_NIs, terminal_names):
            pin_pos = self.build_d2d_pin_pos(pos_2d)

            return terminal_legalize_aux_op(tier, pin_pos, pos_2d,
                                            pos_terminal, num_terminal_NIs,
                                            terminal_names)

        return build_terminal_legalize_aux_op

    def build_refinement(self):

        refinement_op = Refinement(
            self.data_collections_2d.flat_net2pin_map,
            self.data_collections_2d.flat_net2pin_start_map,
            self.data_collections_2d.pin2node_map,
            self.data_collections_2d.net_weights,
            self.placedb_2d.num_movable_nodes, self.placedb_2d.node_names,
            self.placedb_2d.net_names, self.node_size_x, self.node_size_y,
            self.pin_offset_x, self.pin_offset_y, self.die_size_x,
            self.die_size_y, self.row_height, self.die_spec.terminalSizeX,
            self.die_spec.terminalSizeY, self.die_spec.terminalSpacing,
            self.case_name)

        def build_refinement_op(tier, pos_2d, pos_terminal, num_terminal_NIs,
                                terminal_names):
            pin_pos = self.build_d2d_pin_pos(pos_2d)
            return refinement_op(tier, pin_pos, pos_2d, pos_terminal,
                                 num_terminal_NIs, terminal_names)

        return build_refinement_op

    def build_bin_based_fm(self):

        bin_based_fm_op = BinBasedFM(
            self.data_collections_2d.flat_net2pin_map,
            self.data_collections_2d.flat_net2pin_start_map,
            self.data_collections_2d.pin2node_map,
            self.d2d_hpwl_net_weights,
            self.placedb_2d.num_movable_nodes, self.placedb_2d.node_names,
            self.placedb_2d.net_names, self.node_size_x, self.node_size_y,
            self.pin_offset_x, self.pin_offset_y, self.die_size_x,
            self.die_size_y, self.row_height, self.die_spec.terminalSizeX,
            self.die_spec.terminalSizeY, self.die_spec.terminalSpacing,
            self.case_name, self.top_die_max_util, self.bottom_die_max_util,
            self.placedb_2d.num_nodes, self.placedb_2d.num_bins_x//2,
            self.placedb_2d.num_bins_y//2, self.placedb_tier[0].xl, self.placedb_tier[0].yl,
            self.placedb_tier[0].xh, self.placedb_tier[0].yh)

        def build_bin_based_fm_op(tier, pos_2d, pos_terminal, num_terminal_NIs,
                                  terminal_names):
            pin_pos = self.build_d2d_pin_pos(pos_2d)
            return bin_based_fm_op(tier, pin_pos, pos_2d, pos_terminal,
                                   num_terminal_NIs, terminal_names)

        return build_bin_based_fm_op

    def build_hpwl_d2d(self):

        hpwl_d2d_op = HPWLD2D(self.data_collections_2d.flat_net2pin_map,
                              self.data_collections_2d.flat_net2pin_start_map,
                              self.data_collections_2d.pin2node_map,
                              self.d2d_hpwl_net_weights,
                              self.die_spec.terminalSizeX,
                              self.die_spec.terminalSizeY,
                              self.die_spec.terminalSpacing,
                              self.placedb_2d.net_names, self.num_tiers)

        def build_hpwl_d2d_op(pos_2d,
                              cut_net_mask,
                              tier,
                              pos_terminal=torch.empty(0),
                              num_terminal_NIs=0,
                              terminal_names=np.array([], dtype=np.bytes_)):
            pin_pos = self.build_d2d_pin_pos(pos_2d)

            return hpwl_d2d_op(pin_pos, cut_net_mask, tier, pos_terminal,
                               num_terminal_NIs, terminal_names)

        return build_hpwl_d2d_op

    def build_macro_balance(self):
        macro_balance_op = MacroBalance(
            self.data_collections_2d.flat_net2pin_map,
            self.data_collections_2d.flat_net2pin_start_map,
            self.data_collections_2d.pin2node_map,
            self.data_collections_2d.net_weights,
            self.placedb_2d.num_movable_nodes, self.placedb_2d.node_names,
            self.placedb_2d.net_names, self.node_size_x, self.node_size_y,
            self.pin_offset_x, self.pin_offset_y, self.die_size_x,
            self.die_size_y, self.row_height, self.die_spec.terminalSizeX,
            self.die_spec.terminalSizeY, self.die_spec.terminalSpacing,
            self.case_name)

        def build_macro_balance_op(tier, node_orient):

            return macro_balance_op(
                tier, node_orient, self.data_collections_2d.movable_macro_mask)

        return build_macro_balance_op

    def build_hgr_generator(self):
        partitioner_manager = PartitionerManager(
            self.data_collections_2d, self.placedb_2d,
            self.d2d_params.run_tmp_dir_root)
        return partitioner_manager.hgr_generator

    def build_parts_reader(self):
        partitioner_manager = PartitionerManager(
            self.data_collections_2d, self.placedb_2d,
            self.d2d_params.run_tmp_dir_root)
        return partitioner_manager.parts_reader

    def build_draw_layout_result(self):
        if not getattr(self.params, "txt_input", ""):
            return lambda: None
        draw_layout_result_op = DrawLayoutResult(self.params.txt_input,
                                                 self.params.result_dir)
        return draw_layout_result_op

    def build_draw_block(self):
        return DrawBlock(self.placedb_2d)

    def build_partition_flow(self):

        def build_partition_flow_op(partitioner="tritonpart", logger=logging):
            logger.info(f"building partition flow for {partitioner}")
            if partitioner == "tritonpart":
                tritonpart = TritonPartBase(self.d2d_params)
                tier = tritonpart.flow(hgr_generator_op=self.hgr_generator_op,
                                       parts_reader_op=self.parts_reader_op)

            elif partitioner == "bin-based-tritonpart":
                tritonpart = TritonPartBase(self.d2d_params)
                tier = tritonpart.flow(
                    hgr_generator_op=self.hgr_generator_op,
                    parts_reader_op=self.parts_reader_op,
                    pos=self.data_2d.pos,
                    num_movable_nodes=self.placedb_2d.num_movable_nodes)

            elif partitioner == "specpart":
                # SpecPart imports juliacall; keep it out of non-specpart flows
                # because juliacall and torch have fragile import-order behavior.
                from placer.tools.thirdparty_api.specpart_base import SpecPartBase
                specpart = SpecPartBase(self.d2d_params)
                tier = specpart.flow(hgr_generator_op=self.hgr_generator_op,
                                     parts_reader_op=self.parts_reader_op)

            elif Differentiable3DPartitionerBase.is_partition_name(
                    partitioner):
                d3d_partitioner = Differentiable3DPartitionerBase(
                    self.d2d_params, data_2d=self.data_2d, logger=logger)
                tier = d3d_partitioner.flow(
                    hgr_generator_op=self.hgr_generator_op,
                    parts_reader_op=self.parts_reader_op)

            else:
                logger.info(
                    f"no partitioner specified, using default partitioner: Hmetis"
                )
                tier = self.hmetis_op()

            hgr_path = (self.d2d_params.run_tmp_dir_root + "/" +
                        self.case_name + ".hgr")
            part_path = hgr_path + ".part.2"
            new_clique_cut, new_cutnet = GraphCutsize.calculate_from_files(
                hgr_path, part_path)
            logger.info("clique graph cutsize: %d, hyperedge cutsize: %d" %
                        (new_clique_cut, new_cutnet))

            return tier

        return build_partition_flow_op
