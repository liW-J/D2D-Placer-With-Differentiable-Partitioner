'''
Author: JeanneWillis hi@jeannewillis.cn
Date: 2025-06-13 15:35:55
LastEditors: JeanneWillis hi@jeannewillis.cn
LastEditTime: 2025-09-21 00:31:44
FilePath: /D2D-placer/placer/op_wrapper.py
Description:
'''
import configure
from placer.ops.hmetis.hmetis import Hmetis
from placer.ops.multi_bipartition.multi_bipartition import MultiBipartition
from placer.ops.partition_aux.partition_aux import PartitionAux
from placer.ops.avg_cut.avg_cut import AvgCut
from placer.ops.terminal_aux.terminal_aux import TerminalAux
from placer.ops.refinement.refinement import Refinement
from placer.ops.hpwl_d2d.hpwl_d2d import HPWLD2D
from placer.ops.macro_balance.macro_balance import MacroBalance
from placer.ops.parts_reader.parts_reader import PartsReader

from placer.tools.out_fmt_iccad import OutfmtICCAD
from placer.tools.pos_flattened import PosFlattened
from placer.tools.partitioner_manager import PartitionerManager
from dreamplace.ops.pin_pos.pin_pos import PinPos

import torch
import numpy as np


class D2DOpCollection(object):

    def __init__(self, hmetis_op, multi_bipartition_op, init_partition_op,
                 out_fmt_iccad_op, pos_flattened_op, terminal_insert_op,
                 pin_pos_op, pin_pos_tier_op, terminal_legalize_op, avg_cut_op,
                 terminal_insert_aux_op, terminal_legaliza_aux_op,
                 refinement_op, hpwl_d2d_op, macro_balance_op, parts_reader_op,
                 hgr_generator_op):
        self.hmetis_op = hmetis_op
        self.multi_bipartition_op = multi_bipartition_op
        self.init_partition_op = init_partition_op
        self.out_fmt_iccad_op = out_fmt_iccad_op
        self.pos_flattened_op = pos_flattened_op
        self.terminal_insert_op = terminal_insert_op
        self.pin_pos_op = pin_pos_op
        self.pin_pos_tier_op = pin_pos_tier_op
        self.terminal_legalize_op = terminal_legalize_op
        self.avg_cut_op = avg_cut_op
        self.terminal_insert_aux_op = terminal_insert_aux_op
        self.terminal_legaliza_aux_op = terminal_legaliza_aux_op
        self.refinement_op = refinement_op
        self.hpwl_d2d_op = hpwl_d2d_op
        self.macro_balance_op = macro_balance_op
        self.parts_reader_op = parts_reader_op
        self.hgr_generator_op = hgr_generator_op


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

        # 3d-placer set flattened_die size as die_size*2
        if self.num_tiers == 2:
            self.die_size_x = self.die_spec.dieSizeX
            self.die_size_y = self.die_spec.dieSizeY
        else:
            self.die_size_x = np.mean([
                data.placedb.xh for data in self.data_tier
            ]) - np.mean([data.placedb.xl for data in self.data_tier])
            self.die_size_y = np.mean([
                data.placedb.yh for data in self.data_tier
            ]) - np.mean([data.placedb.yl for data in self.data_tier])

        self.row_height = [data.placedb.row_height for data in self.data_tier]

        self.hmetis_op = self.build_hmetis()
        self.multi_bipartition_op = self.build_multi_bipartition()
        self.init_partition_op = self.build_init_partition()
        self.out_fmt_iccad_op = self.build_out_fmt_iccad()
        self.pos_flattened_op = self.build_pos_flattened()
        self.terminal_insert_op = self.build_terminal_insert()
        self.pin_pos_op = self.build_pin_pos()
        self.pin_pos_tier_op = self.build_pin_pos_tier()
        self.terminal_legalize_op = self.build_terminal_legalize()
        self.avg_cut_op = self.build_avg_cut()
        self.terminal_insert_aux_op = self.build_terminal_insert_aux()
        self.terminal_legaliza_aux_op = self.build_terminal_legaliza_aux()
        self.refinement_op = self.build_refinement()
        self.hpwl_d2d_op = self.build_hpwl_d2d()
        self.macro_balance_op = self.build_macro_balance()
        self.parts_reader_op = self.build_parts_reader()
        self.hgr_generator_op = self.build_hgr_generator()

        self.d2d_op_collections = D2DOpCollection(
            hmetis_op=self.hmetis_op,
            multi_bipartition_op=self.multi_bipartition_op,
            init_partition_op=self.init_partition_op,
            out_fmt_iccad_op=self.out_fmt_iccad_op,
            pos_flattened_op=self.pos_flattened_op,
            terminal_insert_op=self.terminal_insert_op,
            pin_pos_op=self.pin_pos_op,
            pin_pos_tier_op=self.pin_pos_tier_op,
            terminal_legalize_op=self.terminal_legalize_op,
            avg_cut_op=self.avg_cut_op,
            terminal_insert_aux_op=self.terminal_insert_aux_op,
            terminal_legaliza_aux_op=self.terminal_legaliza_aux_op,
            refinement_op=self.refinement_op,
            hpwl_d2d_op=self.hpwl_d2d_op,
            macro_balance_op=self.macro_balance_op,
            parts_reader_op=self.parts_reader_op,
            hgr_generator_op=self.hgr_generator_op)

    def build_hmetis(self):

        hmetis_op = Hmetis(self.data_collections_2d.flat_net2pin_map,
                           self.data_collections_2d.flat_net2pin_start_map,
                           self.data_collections_2d.pin2node_map,
                           self.data_collections_2d.net_weights,
                           self.data_collections_2d.net_mask_all,
                           self.placedb_2d.num_movable_nodes, self.case_name)

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
            case_name=self.case_name)

        def build_init_partition_op(tier, pos_2d, node_orient):
            pin_pos_x = torch.stack([
                self.pin_pos_tier_op[tier_id](pos_2d)
                [:self.data_collections_2d.pin2node_map.numel()]
                for tier_id in range(self.num_tiers)
            ])
            pin_pos_y = torch.stack([
                self.pin_pos_tier_op[tier_id](pos_2d)
                [self.data_collections_2d.pin2node_map.numel():]
                for tier_id in range(self.num_tiers)
            ])
            pin_pos = torch.cat([pin_pos_x, pin_pos_y], dim=0)

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
            case_name=self.case_name)

        def build_terminal_insert_op(tier, pos_2d, node_orient):
            pin_pos_x = torch.stack([
                self.pin_pos_tier_op[tier_id](pos_2d)
                [:self.data_collections_2d.pin2node_map.numel()]
                for tier_id in range(self.num_tiers)
            ])
            pin_pos_y = torch.stack([
                self.pin_pos_tier_op[tier_id](pos_2d)
                [self.data_collections_2d.pin2node_map.numel():]
                for tier_id in range(self.num_tiers)
            ])
            pin_pos = torch.cat([pin_pos_x, pin_pos_y], dim=0)

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
            case_name=self.case_name)

        def build_terminal_legalize_op(tier, pos_2d, pos_terminal,
                                       num_terminal_NIs, terminal_names,
                                       node_orient):
            pin_pos_x = torch.stack([
                self.pin_pos_tier_op[tier_id](pos_2d)
                [:self.data_collections_2d.pin2node_map.numel()]
                for tier_id in range(self.num_tiers)
            ])
            pin_pos_y = torch.stack([
                self.pin_pos_tier_op[tier_id](pos_2d)
                [self.data_collections_2d.pin2node_map.numel():]
                for tier_id in range(self.num_tiers)
            ])
            pin_pos = torch.cat([pin_pos_x, pin_pos_y], dim=0)
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
            terminal_legalize_flag=False)

        def build_terminal_insert_aux_op(tier, pos_2d):
            pin_pos_x = torch.stack([
                self.pin_pos_tier_op[tier_id](pos_2d)
                [:self.data_collections_2d.pin2node_map.numel()]
                for tier_id in range(self.num_tiers)
            ])
            pin_pos_y = torch.stack([
                self.pin_pos_tier_op[tier_id](pos_2d)
                [self.data_collections_2d.pin2node_map.numel():]
                for tier_id in range(self.num_tiers)
            ])
            pin_pos = torch.cat([pin_pos_x, pin_pos_y], dim=0)
            return terminal_insert_aux_op(tier, pin_pos, pos_2d)

        return build_terminal_insert_aux_op

    def build_terminal_legaliza_aux(self):

        terminal_legaliza_aux_op = TerminalAux(
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
            terminal_legalize_flag=True)

        def build_terminal_legaliza_aux_op(tier, pos_2d, pos_terminal,
                                           num_terminal_NIs, terminal_names):
            pin_pos_x = torch.stack([
                self.pin_pos_tier_op[tier_id](pos_2d)
                [:self.data_collections_2d.pin2node_map.numel()]
                for tier_id in range(self.num_tiers)
            ])
            pin_pos_y = torch.stack([
                self.pin_pos_tier_op[tier_id](pos_2d)
                [self.data_collections_2d.pin2node_map.numel():]
                for tier_id in range(self.num_tiers)
            ])
            pin_pos = torch.cat([pin_pos_x, pin_pos_y], dim=0)
            return terminal_legaliza_aux_op(tier, pin_pos, pos_2d, pos_terminal,
                                       num_terminal_NIs, terminal_names)

        return build_terminal_legaliza_aux_op

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
            pin_pos_x = torch.stack([
                self.pin_pos_tier_op[tier_id](pos_2d)
                [:self.data_collections_2d.pin2node_map.numel()]
                for tier_id in range(self.num_tiers)
            ])
            pin_pos_y = torch.stack([
                self.pin_pos_tier_op[tier_id](pos_2d)
                [self.data_collections_2d.pin2node_map.numel():]
                for tier_id in range(self.num_tiers)
            ])
            pin_pos = torch.cat([pin_pos_x, pin_pos_y], dim=0)
            return refinement_op(tier, pin_pos, pos_2d, pos_terminal,
                                 num_terminal_NIs, terminal_names)

        return build_refinement_op

    def build_hpwl_d2d(self):

        hpwl_d2d_op = HPWLD2D(self.data_collections_2d.flat_net2pin_map,
                              self.data_collections_2d.flat_net2pin_start_map,
                              self.data_collections_2d.pin2node_map,
                              self.data_collections_2d.net_weights,
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
            pin_pos_x = torch.stack([
                self.pin_pos_tier_op[tier_id](pos_2d)
                [:self.data_collections_2d.pin2node_map.numel()]
                for tier_id in range(self.num_tiers)
            ])
            pin_pos_y = torch.stack([
                self.pin_pos_tier_op[tier_id](pos_2d)
                [self.data_collections_2d.pin2node_map.numel():]
                for tier_id in range(self.num_tiers)
            ])
            pin_pos = torch.cat([pin_pos_x, pin_pos_y], dim=0)

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
