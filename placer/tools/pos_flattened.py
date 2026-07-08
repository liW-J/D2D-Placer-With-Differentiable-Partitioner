'''
Author: JeanneWillis hi@jeannewillis.cn
Date: 2025-04-14 03:21:27
LastEditors: JeanneWillis hi@jeannewillis.cn
LastEditTime: 2025-09-14 21:37:03
FilePath: /D2D-placer/placer/tools/pos_flattened.py
Description: 
'''
import time
import logging
import torch


class PosFlattened:

    BOOKSHELF_RESERVED = {
        "terminal", "ucla", "netdegree", "scl", "nodes", "nets",
        "pl", "wts", "shapes", "route", "aux", "fixed",
        "fixed_ni", "placed", "unplaced", "o", "i", "b",
        "n", "s", "w", "e", "fn", "fs", "fw", "fe",
    }

    def __init__(self, params, data_2d, data_tier):
        self.params = params
        self.data_2d = data_2d
        self.data_tier = data_tier
        self._node_alias_to_2d_id = None

    @staticmethod
    def _decode_name(name):
        return name.decode('utf-8') if isinstance(name, bytes) else str(name)

    @classmethod
    def _is_bookshelf_string_char(cls, char):
        return char.isalnum() or char in "_,.$-[]/"

    @classmethod
    def _bookshelf_name(cls, raw, prefix, index, force_prefix=False):
        changed = (force_prefix or not raw or not raw[0].isalpha() or
                   raw.lower() in cls.BOOKSHELF_RESERVED)
        body_chars = []
        for char in raw:
            if cls._is_bookshelf_string_char(char):
                body_chars.append(char)
            else:
                body_chars.append('_')
                changed = True
        body = ''.join(body_chars) or "anon"
        if not changed:
            return raw
        return "%s%d_%s" % (prefix, index, body)

    @classmethod
    def _bookshelf_node_name(cls, raw, node_id):
        return cls._bookshelf_name(raw, "X", node_id)

    def _build_2d_node_alias_map(self):
        if self._node_alias_to_2d_id is not None:
            return self._node_alias_to_2d_id

        placedb = self.data_2d.placedb
        raw_name2id = placedb.node_name2id_map
        alias_to_id = {}

        for node_id, raw_name in enumerate(placedb.node_names):
            key = self._decode_name(raw_name)
            mapped_id = raw_name2id.get(key)
            if mapped_id is None:
                mapped_id = raw_name2id.get(raw_name, node_id)
            mapped_id = int(mapped_id)
            alias_to_id[key] = mapped_id
            alias_to_id[self._bookshelf_node_name(key, mapped_id)] = mapped_id

        for raw_name, node_id in raw_name2id.items():
            key = self._decode_name(raw_name)
            node_id = int(node_id)
            alias_to_id[key] = node_id
            alias_to_id[self._bookshelf_node_name(key, node_id)] = node_id

        self._node_alias_to_2d_id = alias_to_id
        return alias_to_id

    def map_tier_to_2d(self, node_names):
        """
        @brief Map tier movable node names to flattened-2D placedb node ids.

        NOTE: do NOT assume node "Ck" has 2D node id k-1. dreamplace's
        PlaceDB reorders movable nodes internally (e.g. large movable
        macros are handled separately), so the only safe mapping is the
        2D placedb's node_name2id_map plus the same Bookshelf-safe aliases
        used when partition_aux writes tier Bookshelf files.
        """
        name2id = self._build_2d_node_alias_map()
        ids = []
        for name in node_names:
            key = self._decode_name(name)
            idx = name2id.get(key)
            if idx is None:
                raise KeyError(
                    "pos_flattened: tier node %s not found in flattened-2D "
                    "placedb" % key)
            ids.append(idx)
        return torch.tensor(ids, dtype=torch.int64)

    def __call__(self, tier, pos_2d, pos_tier):
        """
            @brief 
            @param 
            """
        num_tiers = self.params.num_tiers
        num_nodes_2d = len(pos_2d) // 2

        tt = time.time()
        for i in range(num_tiers):
            num_movable = self.data_tier[i].placedb.num_movable_nodes
            node_x = pos_tier[i][:num_movable]
            node_y = pos_tier[i][len(pos_tier[i]) // 2:len(pos_tier[i]) // 2 +
                                 num_movable]
            ids_2d = self.map_tier_to_2d(
                self.data_tier[i].placedb.node_names[:num_movable]).to(
                    pos_2d.device)
            pos_2d.data[ids_2d] = node_x
            pos_2d.data[num_nodes_2d + ids_2d] = node_y

        logging.info("pos_flattened takes %.3f seconds" % (time.time() - tt))

        return pos_2d
