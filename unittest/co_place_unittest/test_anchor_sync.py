"""CPU regression tests for HBT anchors during joint placement.

Run with ``python3 -m unittest discover -s unittest/co_place_unittest -v``.
Only DREAMPlace's compiled dependencies and placement models are replaced;
mapping, position assembly, objective dispatch, metrics and writeback use the
production D2DCoPlace implementation and real PyTorch tensors.
"""

import importlib.util
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch


def _load_coplace():
    names = (
        "dreamplace", "dreamplace.PlaceObj", "dreamplace.EvalMetrics",
        "dreamplace.ops", "dreamplace.ops.hpwl", "dreamplace.ops.hpwl.hpwl",
    )
    modules = {name: ModuleType(name) for name in names}
    for name, module in modules.items():
        module.__path__ = []
        if "." in name:
            parent, attribute = name.rsplit(".", 1)
            setattr(modules[parent], attribute, module)
    path = (Path(__file__).resolve().parents[2] / "placer" / "tools" /
            "diff_d2d_wirelength.py")
    spec = importlib.util.spec_from_file_location("coplace_anchor_test", path)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, modules):
        spec.loader.exec_module(module)
    return module


coplace = _load_coplace()


def _terminal_hpwl(pos):
    # One two-pin net per anchor, with zero pin offsets.
    n = pos.numel() // 2
    return ((pos[0] - pos[2:6]).abs().sum() +
            (pos[n] - pos[n + 2:n + 6]).abs().sum())


class _Model:
    def __init__(self, terminal=False):
        self.terminal = terminal
        self.gamma = torch.ones(1)
        self.density_weight = torch.ones(1)

    def obj_and_grad_fn(self, pos):
        if self.terminal:
            n = pos.numel() // 2
            # Smooth two-pin cost with DREAMPlace's fixed-pin gradient mask:
            # anchor coordinates affect HBT forces but are not independent
            # movable variables of the terminal layer.
            obj = ((pos[0] - pos[2:6].detach()).square().sum() +
                   (pos[n] - pos[n + 2:n + 6].detach()).square().sum())
        else:
            obj = pos.sum() * 0
        obj.backward()
        return obj, pos.grad


class _Metrics:
    def __init__(self, iteration):
        self.iteration = iteration

    def evaluate(self, placedb, ops, pos, data):
        self.pos = pos.clone()
        self.hpwl = ops["hpwl"](pos)
        self.overflow = ops["overflow"](pos)[0]


def _database(names, n_mov, n_fixed, n_ni, xs, ys, terminal=False):
    pos = torch.tensor(xs + ys, dtype=torch.float64, requires_grad=True)
    placedb = SimpleNamespace(
        node_names=names,
        node2pin_map=[np.array([i]) for i in range(len(names))],
        pin_offset_x=np.array([2 if terminal else 1
                               for _ in names], dtype=float),
        pin_offset_y=np.array([2 if terminal else 1
                               for _ in names], dtype=float),
        num_movable_nodes=n_mov,
        num_terminals=n_fixed,
        num_terminal_NIs=n_ni,
        num_physical_nodes=len(names),
        num_nodes=len(xs),
    )
    ops = SimpleNamespace(
        hpwl_op=_terminal_hpwl if terminal else lambda p: p.new_zeros(()),
        density_overflow_op=lambda p: (p.new_zeros(1), p.new_zeros(1)),
        move_boundary_op=lambda p: None,
        draw_place_op=lambda p, filename: None,
    )
    return SimpleNamespace(
        placedb=placedb,
        basic_place=SimpleNamespace(
            data_collections=SimpleNamespace(pos=[pos]), op_collections=ops),
        pos=pos,
    )


def _placer():
    # Neither the anchor order nor bytes/str representation matches the tiers.
    top = _database([b"top_b", "top_a", b"bond"], 2, 0, 1,
                    [0, 20, 10, 30], [100, 120, 110, 130])
    bot = _database(["bot_b", b"bot_a", "bond"], 2, 0, 1,
                    [40, 60, 10, 50], [140, 160, 110, 150])
    term = _database(
        ["bond", "real_fixed_pad", "bot_a", "top_b", b"top_a", b"bot_b"],
        1, 1, 4,
        [10, 700, -900, -901, -902, -903, 70],
        [110, 800, -800, -801, -802, -803, 170], terminal=True)
    return SimpleNamespace(
        params=SimpleNamespace(
            partition_tier=[SimpleNamespace(), SimpleNamespace()],
            terminal=SimpleNamespace(), result_dir_root="unused"),
        dreamplace=SimpleNamespace(dp_tier=[top, bot], dp_terminal=term),
        num_terminal_NIs=1,
        die_spec=SimpleNamespace(terminalSpacing=2),
    )


def _helper(placer=None):
    def build_models(co):
        co.model_top = _Model()
        co.model_bot = _Model()
        co.model_term = _Model(terminal=True)

    with patch.object(coplace.D2DCoPlace, "_install_compact_topology"), \
            patch.object(coplace.D2DCoPlace, "_build_placeobjs", build_models), \
            patch.object(coplace.D2DCoPlace, "log_memory"):
        return coplace.D2DCoPlace(placer or _placer())


class AnchorSyncTest(unittest.TestCase):
    def test_names_map_current_cells_without_moving_true_fixed_nodes(self):
        co = _helper()
        p = co.pack_initial()
        original_base = co.dp_term.pos.detach().clone()
        top, bot, term = co._build_all_pos(p)
        expected = torch.tensor(
            [10, 700, 60, 0, 20, 40, 70,
             110, 800, 160, 100, 120, 140, 170], dtype=p.dtype)
        torch.testing.assert_close(term, expected)
        torch.testing.assert_close(co.dp_term.pos, original_base)
        # The existing HBT -> tier synchronization still includes spacing/2.
        self.assertEqual(top[2].item(), 11)
        self.assertEqual(bot[2].item(), 11)

        with torch.no_grad():
            p[0] = 333  # top_b x
            p[co._half + co._off_bot_mov + 1] = 444  # bot_a y
        _, _, updated = co._build_all_pos(p)
        self.assertEqual(updated[3].item(), 333)
        self.assertEqual(updated[co.n_term_total + 2].item(), 444)
        self.assertEqual(updated[1].item(), 700)
        self.assertEqual(updated[co.n_term_total + 1].item(), 800)

    def test_objective_uses_line_search_candidate_and_updates_hbt_force(self):
        for tier in ("top", "bot"):
            with self.subTest(tier=tier):
                co = _helper()
                p = co.pack_initial()
                with torch.no_grad():
                    p.zero_()
                    p[co._off_term_mov] = 10
                    p[co._half + co._off_term_mov] = 10
                objective = co.make_obj_and_grad_fn()
                old_base = co.dp_term.pos.detach().clone()
                obj, grad = objective(p)
                self.assertEqual(obj.item(), 800)
                self.assertEqual(grad[co._off_term_mov].item(), 80)

                candidate = p.detach().clone().requires_grad_(True)
                start = getattr(co, "_off_%s_mov" % tier)
                with torch.no_grad():
                    candidate[start:start + 2] = 40
                    candidate[co._half + start:co._half + start + 2] = 40
                obj, grad = objective(candidate)
                self.assertEqual(obj.item(), 4000)
                self.assertEqual(grad[co._off_term_mov].item(), -80)
                self.assertEqual(grad[co._half + co._off_term_mov].item(), -80)
                # Candidate evaluation must not mutate the base or accepted p.
                torch.testing.assert_close(co.dp_term.pos, old_base)
                torch.testing.assert_close(p[:4], p.new_zeros(4))

    def test_metrics_plot_and_writeback_use_the_same_current_anchors(self):
        co = _helper()
        p = co.pack_initial()
        _, _, expected = co._build_all_pos(p)
        with patch.object(coplace.EvalMetrics, "EvalMetrics", _Metrics,
                          create=True):
            metrics = co.evaluate_metrics(p, 2)
        torch.testing.assert_close(metrics[2].pos, expected)
        self.assertEqual(metrics[2].hpwl.item(), 200)

        # Exercise all position-building consumers affected by the new args.
        co.overflow(p)
        co.constraint_fn(p)
        drawn = []
        co.dp_term.basic_place.op_collections.draw_place_op = (
            lambda pos, filename: drawn.append(pos.clone()))
        with tempfile.TemporaryDirectory() as temp_dir:
            co.params.result_dir_root = temp_dir
            co.plot(p, 2)
        self.assertEqual(len(drawn), 1)
        torch.testing.assert_close(drawn[0], expected)

        co.writeback(p)
        torch.testing.assert_close(co.dp_term.pos, expected)
        torch.testing.assert_close(
            co.dp_term.basic_place.data_collections.pos[0], expected)

    def test_unknown_anchor_fails_instead_of_silently_staying_stale(self):
        placer = _placer()
        placer.dreamplace.dp_terminal.placedb.node_names[2] = "missing_cell"
        with self.assertRaisesRegex(ValueError, "missing_cell"):
            _helper(placer)

    def test_no_cross_tier_anchors_preserves_fixed_nodes_and_fillers(self):
        placer = _placer()
        placer.num_terminal_NIs = 0
        placer.dreamplace.dp_tier = [
            _database(["top_a"], 1, 0, 0, [1, 2], [3, 4]),
            _database(["bot_a"], 1, 0, 0, [5, 6], [7, 8]),
        ]
        placer.dreamplace.dp_terminal = _database(
            ["real_fixed_pad"], 0, 1, 0, [700, 70], [800, 170])
        co = _helper(placer)
        p = co.pack_initial()
        _, _, term = co._build_all_pos(p)
        torch.testing.assert_close(term, co.dp_term.pos)
        with torch.no_grad():
            p[co._off_term_fil] = 123
        _, _, term = co._build_all_pos(p)
        torch.testing.assert_close(
            term, p.new_tensor([700, 123, 800, 170]))

    def test_ambiguous_or_duplicate_anchor_fails(self):
        placer = _placer()
        placer.dreamplace.dp_tier[1].placedb.node_names[0] = "top_b"
        with self.assertRaises(ValueError):
            _helper(placer)
        placer = _placer()
        placer.dreamplace.dp_terminal.placedb.node_names[2] = "top_b"
        with self.assertRaises(ValueError):
            _helper(placer)


if __name__ == "__main__":
    unittest.main()
