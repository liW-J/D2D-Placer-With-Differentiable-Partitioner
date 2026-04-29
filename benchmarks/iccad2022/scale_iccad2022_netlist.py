#!/usr/bin/env python3
"""Scale ICCAD2022-format netlist by replicating a base design.

This script:
1) Replicates the original instance/netlist structure by `scale` times.
2) Creates random cross-replica nets to connect replicated sub-netlists.
3) Rewrites `NumInstances` and `NumNets` to keep the file format valid.
"""

from __future__ import annotations

import argparse
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class Instance:
    name: str
    libcell: str


@dataclass(frozen=True)
class Net:
    name: str
    pins: Tuple[Tuple[str, str], ...]  # (inst_name, pin_name)


def parse_instance(line: str) -> Instance:
    toks = line.split()
    if len(toks) != 3 or toks[0] != "Inst":
        raise ValueError(f"Invalid Inst line: {line}")
    return Instance(name=toks[1], libcell=toks[2])


def parse_net_header(line: str) -> Tuple[str, int]:
    toks = line.split()
    if len(toks) != 3 or toks[0] != "Net":
        raise ValueError(f"Invalid Net line: {line}")
    degree = int(toks[2])
    if degree <= 0:
        raise ValueError(f"Net degree must be positive: {line}")
    return toks[1], degree


def parse_pin(line: str) -> Tuple[str, str]:
    toks = line.split()
    if len(toks) != 2 or toks[0] != "Pin":
        raise ValueError(f"Invalid Pin line: {line}")
    if "/" not in toks[1]:
        raise ValueError(f"Invalid pin token: {line}")
    inst_name, pin_name = toks[1].split("/", 1)
    if not inst_name or not pin_name:
        raise ValueError(f"Invalid pin token: {line}")
    return inst_name, pin_name


def find_unique_line(lines: Sequence[str], prefix: str) -> int:
    idxs = [i for i, line in enumerate(lines) if line.strip().startswith(prefix)]
    if len(idxs) != 1:
        raise ValueError(f"Expected exactly one `{prefix}` line, found {len(idxs)}")
    return idxs[0]


def best_tiling_factors(scale: int) -> Tuple[int, int]:
    """Return balanced (cols, rows) with cols * rows >= scale."""
    if scale < 1:
        raise ValueError("scale must be >= 1")
    # Prefer near-square tiling to keep die aspect ratio close to original.
    best_cols = 1
    best_rows = scale
    best_aspect_dist = float("inf")
    best_area = float("inf")

    max_cols = int(math.ceil(math.sqrt(scale) * 2))
    for cols in range(1, max_cols + 1):
        rows = int(math.ceil(scale / cols))
        area = cols * rows
        aspect_dist = abs(math.log(cols / rows))
        if (
            aspect_dist < best_aspect_dist
            or (aspect_dist == best_aspect_dist and area < best_area)
            or (
                aspect_dist == best_aspect_dist
                and area == best_area
                and abs(cols - rows) < abs(best_cols - best_rows)
            )
        ):
            best_cols, best_rows = cols, rows
            best_aspect_dist = aspect_dist
            best_area = area

    return best_cols, best_rows


def parse_diesize(line: str) -> Tuple[int, int, int, int]:
    toks = line.split()
    if len(toks) != 5 or toks[0] != "DieSize":
        raise ValueError(f"Invalid DieSize line: {line}")
    x1, y1, x2, y2 = map(int, toks[1:])
    return x1, y1, x2, y2


def parse_dierows(line: str, row_key: str) -> Tuple[int, int, int, int, int]:
    toks = line.split()
    if len(toks) != 6 or toks[0] != row_key:
        raise ValueError(f"Invalid {row_key} line: {line}")
    x, y, row_w, row_h, row_cnt = map(int, toks[1:])
    return x, y, row_w, row_h, row_cnt


def scale_floorplan_header(
    header_lines: Sequence[str],
    scale: int,
    tile_cols: Optional[int],
    tile_rows: Optional[int],
) -> List[str]:
    if (tile_cols is None) != (tile_rows is None):
        raise ValueError("--tile-cols and --tile-rows must be set together")

    if tile_cols is None:
        cols, rows = best_tiling_factors(scale)
    else:
        cols, rows = tile_cols, tile_rows
        if cols < 1 or rows < 1:
            raise ValueError("--tile-cols and --tile-rows must be >= 1")

    new_header = list(header_lines)

    idx_die = find_unique_line(new_header, "DieSize")
    x1, y1, x2, y2 = parse_diesize(new_header[idx_die])
    die_w = x2 - x1
    die_h = y2 - y1
    if die_w <= 0 or die_h <= 0:
        raise ValueError(f"Invalid DieSize geometry: {new_header[idx_die]}")
    new_header[idx_die] = f"DieSize {x1} {y1} {x1 + die_w * cols} {y1 + die_h * rows}"

    idx_top_rows = find_unique_line(new_header, "TopDieRows")
    tx, ty, t_row_w, t_row_h, t_row_cnt = parse_dierows(new_header[idx_top_rows], "TopDieRows")
    new_header[idx_top_rows] = (
        f"TopDieRows {tx} {ty} {t_row_w * cols} {t_row_h} {t_row_cnt * rows}"
    )

    idx_bottom_rows = find_unique_line(new_header, "BottomDieRows")
    bx, by, b_row_w, b_row_h, b_row_cnt = parse_dierows(
        new_header[idx_bottom_rows], "BottomDieRows"
    )
    new_header[idx_bottom_rows] = (
        f"BottomDieRows {bx} {by} {b_row_w * cols} {b_row_h} {b_row_cnt * rows}"
    )

    return new_header


def parse_input(path: Path) -> Tuple[List[str], List[Instance], List[Net]]:
    raw_lines = path.read_text(encoding="utf-8").splitlines()
    lines = [line.rstrip() for line in raw_lines]

    idx_num_inst = find_unique_line(lines, "NumInstances")
    idx_num_nets = find_unique_line(lines, "NumNets")
    if idx_num_nets <= idx_num_inst:
        raise ValueError("`NumNets` must appear after `NumInstances`")

    num_inst_tokens = lines[idx_num_inst].split()
    if len(num_inst_tokens) != 2 or num_inst_tokens[0] != "NumInstances":
        raise ValueError(f"Invalid NumInstances line: {lines[idx_num_inst]}")
    declared_num_instances = int(num_inst_tokens[1])

    header_lines = lines[:idx_num_inst]

    inst_lines = [line for line in lines[idx_num_inst + 1 : idx_num_nets] if line.strip()]
    instances = [parse_instance(line) for line in inst_lines]
    if len(instances) != declared_num_instances:
        raise ValueError(
            f"NumInstances mismatch: declared {declared_num_instances}, parsed {len(instances)}"
        )

    num_net_tokens = lines[idx_num_nets].split()
    if len(num_net_tokens) != 2 or num_net_tokens[0] != "NumNets":
        raise ValueError(f"Invalid NumNets line: {lines[idx_num_nets]}")
    declared_num_nets = int(num_net_tokens[1])

    nets: List[Net] = []
    i = idx_num_nets + 1
    while i < len(lines):
        if not lines[i].strip():
            i += 1
            continue
        net_name, degree = parse_net_header(lines[i])
        if i + degree >= len(lines) + 1:
            raise ValueError(f"Net `{net_name}` truncated at end of file")
        pins: List[Tuple[str, str]] = []
        for j in range(i + 1, i + 1 + degree):
            pins.append(parse_pin(lines[j]))
        nets.append(Net(name=net_name, pins=tuple(pins)))
        i += 1 + degree

    if len(nets) != declared_num_nets:
        raise ValueError(f"NumNets mismatch: declared {declared_num_nets}, parsed {len(nets)}")

    return header_lines, instances, nets


def get_unique_pin_candidates(nets: Sequence[Net]) -> List[Tuple[str, str]]:
    seen = set()
    unique_pins = []
    for net in nets:
        for pin in net.pins:
            if pin not in seen:
                seen.add(pin)
                unique_pins.append(pin)
    if not unique_pins:
        raise ValueError("No pins found in original netlist")
    return unique_pins


def scaled_inst_name(inst_name: str, replica_idx: int) -> str:
    return f"{inst_name}_R{replica_idx}"


def build_scaled_design(
    instances: Sequence[Instance],
    nets: Sequence[Net],
    scale: int,
    num_cross_nets: int,
    min_cross_degree: int,
    max_cross_degree: int,
    rng: random.Random,
) -> Tuple[List[Instance], List[Net]]:
    if scale < 1:
        raise ValueError("scale must be >= 1")
    if min_cross_degree < 2:
        raise ValueError("min_cross_degree must be >= 2")
    if max_cross_degree < min_cross_degree:
        raise ValueError("max_cross_degree must be >= min_cross_degree")
    if scale == 1 and num_cross_nets > 0:
        raise ValueError("cross nets require scale >= 2")

    scaled_instances: List[Instance] = []
    name_maps: List[Dict[str, str]] = []

    for r in range(scale):
        m: Dict[str, str] = {}
        for inst in instances:
            new_name = scaled_inst_name(inst.name, r)
            m[inst.name] = new_name
            scaled_instances.append(Instance(name=new_name, libcell=inst.libcell))
        name_maps.append(m)

    scaled_nets: List[Net] = []
    net_idx = 1
    for r in range(scale):
        for net in nets:
            mapped_pins = tuple((name_maps[r][inst], pin) for inst, pin in net.pins)
            scaled_nets.append(Net(name=f"N{net_idx}", pins=mapped_pins))
            net_idx += 1

    if num_cross_nets > 0:
        base_pin_candidates = get_unique_pin_candidates(nets)

        for _ in range(num_cross_nets):
            degree = rng.randint(min_cross_degree, max_cross_degree)
            used = set()
            new_pins: List[Tuple[str, str]] = []

            r1, r2 = rng.sample(range(scale), 2)
            replica_sequence = [r1, r2]
            while len(replica_sequence) < degree:
                replica_sequence.append(rng.randrange(scale))

            for replica in replica_sequence:
                for _try in range(40):
                    inst, pin = rng.choice(base_pin_candidates)
                    mapped = (name_maps[replica][inst], pin)
                    if mapped not in used:
                        used.add(mapped)
                        new_pins.append(mapped)
                        break
                else:
                    raise RuntimeError("Failed to sample unique pins for cross net")

            scaled_nets.append(Net(name=f"N{net_idx}", pins=tuple(new_pins)))
            net_idx += 1

    return scaled_instances, scaled_nets


def write_output(
    path: Path,
    header_lines: Sequence[str],
    instances: Sequence[Instance],
    nets: Sequence[Net],
) -> None:
    out_lines: List[str] = []
    out_lines.extend(header_lines)
    out_lines.append(f"NumInstances {len(instances)}")
    for inst in instances:
        out_lines.append(f"Inst {inst.name} {inst.libcell}")
    out_lines.append("")
    out_lines.append(f"NumNets {len(nets)}")
    for net in nets:
        out_lines.append(f"Net {net.name} {len(net.pins)}")
        for inst_name, pin_name in net.pins:
            out_lines.append(f"Pin {inst_name}/{pin_name}")
    out_lines.append("")
    path.write_text("\n".join(out_lines), encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Scale ICCAD2022-format netlist and add random cross-replica nets."
    )
    parser.add_argument("--input", required=True, type=Path, help="Input .txt netlist file.")
    parser.add_argument("--output", required=True, type=Path, help="Output scaled .txt file.")
    parser.add_argument(
        "--scale",
        required=True,
        type=int,
        help="Replication factor (e.g. 10 or 100).",
    )
    parser.add_argument(
        "--seed",
        default=2026,
        type=int,
        help="Random seed for reproducible cross-net generation.",
    )
    parser.add_argument(
        "--cross-ratio",
        default=0.05,
        type=float,
        help="Cross-net count ratio vs. original NumNets, applied after scaling decision.",
    )
    parser.add_argument(
        "--cross-nets",
        default=None,
        type=int,
        help="Explicit number of additional cross-replica nets (overrides --cross-ratio).",
    )
    parser.add_argument(
        "--min-cross-degree",
        default=2,
        type=int,
        help="Minimum degree for generated cross-replica nets.",
    )
    parser.add_argument(
        "--max-cross-degree",
        default=3,
        type=int,
        help="Maximum degree for generated cross-replica nets.",
    )
    parser.add_argument(
        "--tile-cols",
        default=None,
        type=int,
        help="Optional floorplan tiling columns. No strict product constraint with scale.",
    )
    parser.add_argument(
        "--tile-rows",
        default=None,
        type=int,
        help="Optional floorplan tiling rows. No strict product constraint with scale.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.scale < 1:
        raise ValueError("--scale must be >= 1")
    if args.cross_ratio < 0.0:
        raise ValueError("--cross-ratio must be >= 0")
    if args.cross_nets is not None and args.cross_nets < 0:
        raise ValueError("--cross-nets must be >= 0")

    header, instances, nets = parse_input(args.input)
    scaled_header = scale_floorplan_header(
        header_lines=header,
        scale=args.scale,
        tile_cols=args.tile_cols,
        tile_rows=args.tile_rows,
    )

    if args.cross_nets is None:
        # Cross-net volume scales with the number of replicated boundaries.
        cross_nets = int(round(len(nets) * args.cross_ratio * max(args.scale - 1, 0)))
    else:
        cross_nets = args.cross_nets

    rng = random.Random(args.seed)
    scaled_instances, scaled_nets = build_scaled_design(
        instances=instances,
        nets=nets,
        scale=args.scale,
        num_cross_nets=cross_nets,
        min_cross_degree=args.min_cross_degree,
        max_cross_degree=args.max_cross_degree,
        rng=rng,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_output(args.output, scaled_header, scaled_instances, scaled_nets)

    print(f"[OK] input={args.input}")
    print(f"[OK] output={args.output}")
    print(f"[OK] scale={args.scale}")
    print(f"[OK] original_instances={len(instances)}, scaled_instances={len(scaled_instances)}")
    print(f"[OK] original_nets={len(nets)}, cross_nets={cross_nets}, total_nets={len(scaled_nets)}")


if __name__ == "__main__":
    main()
