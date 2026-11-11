"""Compiled AUX -> Bookshelf reader -> PinPos integration regression.

Run in a built D2D installation (CPU operators are sufficient)::

    python /path/to/repo/unittest/co_place_unittest/pin_views_integration.py \
        --runtime /path/to/install --device cuda:0

Use --device cpu for CPU-only validation. CUDA mode uses the same CPU AUX
exporters as the production GPU flow, then assembles positions and invokes
the compiled PinPos operator on CUDA.

The runtime must contain the current partition_aux/terminal_aux extensions and
diff_d2d_wirelength.py. No compiled operators or pin-coordinate conversions are
mocked. Tests use N-oriented cells, as in the co-placement global-placement
path; orientation conversion during legalization is outside this regression.
"""

import argparse
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace


def _bootstrap(runtime):
    runtime = Path(runtime).resolve()
    for path in (runtime, runtime / "placer", runtime / "thirdparty/DREAMPlace",
                 runtime / "thirdparty/DREAMPlace/dreamplace", runtime / "vendor"):
        sys.path.insert(0, str(path))


def _fixture(torch):
    f = lambda values: torch.tensor(values, dtype=torch.float64)
    i = lambda values: torch.tensor(values, dtype=torch.int32)
    return SimpleNamespace(
        tier=i([0, 1, 0, 1]),
        # Every net is cut, and traversal slots differ from original pin IDs.
        flat_netpin=i([6, 3, 7, 0, 1, 4, 2, 5]),
        netpin_start=i([0, 3, 6, 8]),
        pin2node=i([3, 0, 2, 1, 1, 3, 0, 2]),
        weights=f([1, 1, 1]),
        node_names=["cell_a", "cell_b", "cell_c", "cell_d"],
        net_names=["cut_alpha", "cut_beta", "cut_gamma"],
        # Per-technology arrays retain the original full node/pin indexing.
        width=f([11, 13, 15, 17, 21, 23, 25, 27]),
        height=f([9, 11, 13, 15, 19, 21, 23, 25]),
        # Pin 6 is at a cell corner: ceil(half_size) incorrectly reads back -1.
        offset_x=f([1, 3, 5, 7, 9, 2, 0, 6,
                    11, 13, 15, 17, 19, 12, 14, 16]),
        offset_y=f([2, 4, 6, 8, 10, 3, 0, 7,
                    12, 14, 16, 18, 20, 13, 15, 17]),
        pos=f([20, 80, 120, 160, 30, 70, 90, 150]),
    )


def _export(case, directory, dimensions, torch, partition_cpp, terminal_cpp):
    w, h, spacing = dimensions
    n, pins = len(case.node_names), case.pin2node.numel()
    owners = case.pin2node.long()
    pin_x = case.pos[:n][owners].repeat(2) + case.offset_x
    pin_y = case.pos[n:][owners].repeat(2) + case.offset_y
    pin_pos = torch.cat((pin_x, pin_y))
    common = (
        case.tier, case.flat_netpin, case.netpin_start, case.pin2node,
        case.weights, n, case.width, case.height, case.offset_x, case.offset_y,
        300.0, 300.0, [1, 1], w, h, spacing, pin_pos,
    )
    partition_dir, terminal_dir = directory / "partition", directory / "terminal"
    partition_dir.mkdir(parents=True)
    terminal_dir.mkdir(parents=True)
    empty = torch.empty(0, dtype=case.pos.dtype)
    partition_cut = partition_cpp.partition_aux(
        *common, True, False, empty, 0, case.node_names, case.net_names, [],
        case.pos, str(partition_dir), ["N"] * n)
    terminal_cut = terminal_cpp.terminal_aux(
        *common, case.node_names, case.net_names, case.pos, str(terminal_dir),
        False, empty, 0, [])
    assert partition_cut.tolist() == [1, 1, 1]
    assert terminal_cut.tolist() == [1, 1, 1]
    return ([partition_dir / ("tier%d.aux" % tier) for tier in range(2)],
            terminal_dir / "terminal.aux")


def _read(path, place_io, torch):
    raw = place_io.PlaceIOFunction.read(SimpleNamespace(
        aux_input=str(path), sort_nets_by_degree=False))
    db = place_io.PlaceIOFunction.pydb(raw)
    # Preserve actual reader output; add only the counters needed by co-place.
    names = (
        "node_names", "net_names", "node2pin_map", "pin2node_map",
        "pin2net_map", "pin_offset_x", "pin_offset_y", "flat_node2pin_map",
        "flat_node2pin_start_map", "node_size_x", "node_size_y",
    )
    view = SimpleNamespace(**{name: getattr(db, name) for name in names})
    view.num_nodes = int(db.num_nodes)
    view.num_terminals = int(db.num_terminals)
    view.num_terminal_NIs = int(db.num_terminal_NIs)
    view.num_physical_nodes = view.num_nodes
    view.num_movable_nodes = (view.num_nodes - view.num_terminals -
                              view.num_terminal_NIs)
    pos = torch.tensor(list(db.node_x) + list(db.node_y), dtype=torch.float64)
    return SimpleNamespace(
        raw=raw, placedb=view, pos=pos,
        basic_place=SimpleNamespace(data_collections=SimpleNamespace(pos=[pos])))


def _coplace_fixture(top, bot, term, helper):
    # Avoid constructing density FFTs: exercise real mapping and assembly with
    # the three real Bookshelf databases. There are no synthetic pin offsets.
    co = helper.__new__(helper)
    co.dp_top, co.dp_bot, co.dp_term = top, bot, term
    for name, dp in (("top", top), ("bot", bot), ("term", term)):
        setattr(co, "n_%s_mov" % name, dp.placedb.num_movable_nodes)
        setattr(co, "n_%s_phys" % name, dp.placedb.num_physical_nodes)
        setattr(co, "n_%s_total" % name, dp.placedb.num_nodes)
        setattr(co, "n_%s_fil" % name, 0)
    co.n_term = co.n_term_mov
    for name, dp in (("top", top), ("bot", bot)):
        lo = dp.placedb.num_movable_nodes + dp.placedb.num_terminals
        setattr(co, "%s_term_lo" % name, lo)
        setattr(co, "%s_term_hi" % name, lo + dp.placedb.num_terminal_NIs)
    co._off_top_mov = 0
    co._off_bot_mov = co.n_top_mov
    co._off_term_mov = co.n_top_mov + co.n_bot_mov
    co._half = co._off_term_mov + co.n_term
    co._off_top_fil = co._off_bot_fil = co._off_term_fil = co._half
    co._term_anchor_maps = co._build_term_anchor_maps()
    co._top_term_shift, co._bot_term_shift = co._build_terminal_shifts()
    return co


def _name(value):
    return value.decode() if isinstance(value, bytes) else str(value)


def _world_pins(dp, pos, torch, pin_pos):
    db = dp.placedb
    op = pin_pos.PinPos(
        pin_offset_x=torch.tensor(db.pin_offset_x, dtype=pos.dtype, device=pos.device),
        pin_offset_y=torch.tensor(db.pin_offset_y, dtype=pos.dtype, device=pos.device),
        pin2node_map=torch.tensor(db.pin2node_map, dtype=torch.int32, device=pos.device),
        flat_node2pin_map=torch.tensor(db.flat_node2pin_map, dtype=torch.int32, device=pos.device),
        flat_node2pin_start_map=torch.tensor(db.flat_node2pin_start_map,
                                           dtype=torch.int32, device=pos.device),
        num_physical_nodes=db.num_physical_nodes, algorithm="node-by-node")
    result = op(pos).detach().reshape(2, -1).T.cpu().numpy()
    return {
        (_name(db.net_names[net]), _name(db.node_names[node])): result[pin]
        for pin, (net, node) in enumerate(zip(db.pin2net_map, db.pin2node_map))
    }


def _check_views(case, views, positions, torch, np, pin_pos, deltas=None):
    tables = [_world_pins(dp, pos, torch, pin_pos)
              for dp, pos in zip(views, positions)]
    deltas = deltas or {}
    cell_pins = hbt_pins = 0
    for net_id, net_name in enumerate(case.net_names):
        lo, hi = case.netpin_start[net_id:net_id + 2].tolist()
        for slot in range(lo, hi):
            pin = int(case.flat_netpin[slot])
            node = int(case.pin2node[pin])
            tier = int(case.tier[node])
            cell = case.node_names[node]
            offset = tier * case.pin2node.numel() + pin
            delta = deltas.get(cell, (0, 0))
            expected = np.array([
                float(case.pos[node] + case.offset_x[offset]) + delta[0],
                float(case.pos[len(case.node_names) + node] +
                      case.offset_y[offset]) + delta[1]])
            tier_pin = tables[tier][(net_name, cell)]
            anchor_pin = tables[2][("%s_T%d" % (net_name, tier), cell)]
            np.testing.assert_allclose(tier_pin, expected, atol=1e-9, rtol=0,
                                       err_msg="tier pin: %s/%s" % (net_name, cell))
            np.testing.assert_allclose(anchor_pin, expected, atol=1e-9, rtol=0,
                                       err_msg="anchor pin: %s/%s" % (net_name, cell))
            cell_pins += 1
        for tier in range(2):
            np.testing.assert_allclose(
                tables[tier][(net_name, net_name)],
                tables[2][("%s_T%d" % (net_name, tier), net_name)],
                atol=1e-9, rtol=0, err_msg="HBT center: %s/tier%d" % (net_name, tier))
            hbt_pins += 1
    return cell_pins, hbt_pins


def run(runtime, device="cpu"):
    _bootstrap(runtime)
    import numpy as np
    import torch
    if not hasattr(np, "string_"):
        np.string_ = np.bytes_
    from placer.ops.partition_aux import partition_aux_cpp
    from placer.ops.terminal_aux import terminal_aux_cpp
    from placer.tools.diff_d2d_wirelength import D2DCoPlace
    from dreamplace.ops.place_io import place_io
    from dreamplace.ops.pin_pos import pin_pos

    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    torch.set_num_threads(1)
    case = _fixture(torch)
    checked = []
    with tempfile.TemporaryDirectory(prefix="coplace_pin_views_") as directory:
        for dimensions in ((6, 8, 4), (5, 7, 3)):
            path = Path(directory) / ("hbt_%d_%d_spacing_%d" % dimensions)
            tiers, terminal = _export(case, path, dimensions, torch,
                                      partition_aux_cpp, terminal_aux_cpp)
            views = [_read(aux, place_io, torch) for aux in tiers + [terminal]]
            for dp in views:
                dp.pos = dp.pos.to(device)
                dp.basic_place.data_collections.pos[0] = dp.pos
            if dimensions == (6, 8, 4):
                _check_views(case, views, [dp.pos for dp in views], torch, np,
                             pin_pos)
            co = _coplace_fixture(*views, D2DCoPlace)
            p = co.pack_initial()
            positions = co._build_all_pos(p)
            counts = _check_views(case, views, positions, torch, np, pin_pos)
            # Distinct fractional translations ensure anchor mappings cannot
            # accidentally succeed by assuming identical node order.
            deltas = {"cell_a": (7.25, -3.5), "cell_b": (-5.5, 9.25),
                      "cell_c": (2.75, 4.5), "cell_d": (6.5, -2.25)}
            with torch.no_grad():
                for tier, dp in enumerate(views[:2]):
                    start = co._off_top_mov if tier == 0 else co._off_bot_mov
                    for node, value in enumerate(dp.placedb.node_names[
                            :dp.placedb.num_movable_nodes]):
                        dx, dy = deltas[_name(value)]
                        p[start + node] += dx
                        p[co._half + start + node] += dy
                p[co._off_term_mov:co._off_term_mov + co.n_term] += 13.125
                p[co._half + co._off_term_mov:
                  co._half + co._off_term_mov + co.n_term] -= 6.25
            translated = co._build_all_pos(p)
            _check_views(case, views, translated, torch, np, pin_pos, deltas)
            co.writeback(p)
            _check_views(case, views, [dp.pos for dp in views], torch, np,
                         pin_pos, deltas)
            checked.append((dimensions, counts))
    print("PIN_VIEWS_INTEGRATION_PASS", device, checked, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", default=".",
                        help="D2D installation with current compiled AUX operators")
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda:0"))
    args = parser.parse_args()
    run(args.runtime, args.device)
