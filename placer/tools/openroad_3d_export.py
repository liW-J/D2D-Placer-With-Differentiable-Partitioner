'''
OpenROAD/Innovus DEF and Verilog exporter for the D2D placement result.
'''

import json
import logging
import re
from pathlib import Path


INVALID_COORD = -2147483648
TIER_SUFFIX = {
    0: "bottom",
    1: "upper",
}
SUFFIX_RE = re.compile(r"_(?:upper|bottom)$")


def strip_tier_suffix(name):
    return SUFFIX_RE.sub("", name)


def with_tier_suffix(name, suffix):
    return f"{strip_tier_suffix(name)}_{suffix}"


def normalize_name(name):
    if isinstance(name, bytes):
        return name.decode("utf-8")
    return str(name)


def resolve_path(path_value, repo_root):
    if not path_value:
        return None

    path = Path(path_value)
    candidates = [path]
    if not path.is_absolute():
        candidates.extend([Path.cwd() / path, repo_root / path])

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return candidates[-1].resolve()


def companion_path(params, attr_name, txt_path, suffix):
    explicit = getattr(params.flatten_2d, attr_name, None)
    if explicit:
        return resolve_path(explicit, Path(__file__).resolve().parents[2])
    return txt_path.with_name(f"{txt_path.stem}{suffix}")


def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def parse_txt_instances(txt_path):
    inst_to_libcell = {}
    diearea = None

    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if not parts:
                continue
            if parts[0] == "Inst" and len(parts) >= 3:
                inst_to_libcell[parts[1]] = parts[2]
            elif parts[0] == "DieSize" and len(parts) >= 5:
                diearea = tuple(int(float(v)) for v in parts[1:5])

    if not inst_to_libcell:
        raise ValueError(f"No Inst records found in {txt_path}")
    return inst_to_libcell, diearea


def parse_def_diearea(def_path):
    pattern = re.compile(
        r"DIEAREA\s+\(\s*(-?\d+)\s+(-?\d+)\s*\)\s+\(\s*(-?\d+)\s+(-?\d+)\s*\)\s*;"
    )
    with open(def_path, "r", encoding="utf-8") as f:
        for line in f:
            match = pattern.search(line)
            if match:
                return tuple(int(v) for v in match.groups())
    return None


def def_escape_name(verilog_name):
    if verilog_name.startswith("\\"):
        verilog_name = verilog_name[1:]
    return verilog_name.replace("\\", "\\\\").replace("[", "\\[").replace(
        "]", "\\]")


def verilog_instance_from_line(line):
    stripped = line.lstrip()
    if not stripped or stripped.startswith(("//", "/*", "*")):
        return None

    indent_len = len(line) - len(stripped)
    master_match = re.match(r"(?P<master>\S+)(?P<space>\s+)(?P<rest>.*)$",
                            stripped)
    if not master_match:
        return None

    rest = master_match.group("rest")
    if rest.startswith("\\"):
        inst_match = re.match(r"(?P<inst>\\\S+)(?P<space>\s+)(?P<tail>\(.*)$",
                              rest)
    else:
        inst_match = re.match(r"(?P<inst>[^\s(]+)(?P<space>\s*)(?P<tail>\(.*)$",
                              rest)
    if not inst_match:
        return None

    return {
        "indent": line[:indent_len],
        "master": master_match.group("master"),
        "space_after_master": master_match.group("space"),
        "inst": inst_match.group("inst"),
        "space_after_inst": inst_match.group("space"),
        "tail": inst_match.group("tail"),
    }


def write_verilog(verilog_path, out_path, inst_data):
    by_verilog_name = {data["verilog_name"]: data for data in inst_data}
    replaced = set()
    out_lines = []

    with open(verilog_path, "r", encoding="utf-8") as f:
        for line in f:
            parsed = verilog_instance_from_line(line)
            if parsed and parsed["inst"] in by_verilog_name:
                data = by_verilog_name[parsed["inst"]]
                line = (f"{parsed['indent']}{data['master']}"
                        f"{parsed['space_after_master']}{parsed['inst']}"
                        f"{parsed['space_after_inst']}{parsed['tail']}")
                replaced.add(parsed["inst"])
            out_lines.append(line)

    missing = sorted(set(by_verilog_name) - replaced)
    if missing:
        sample = ", ".join(missing[:5])
        raise ValueError(
            f"Failed to update {len(missing)} Verilog instances in {verilog_path}: {sample}"
        )

    out_path.write_text("".join(out_lines), encoding="utf-8")


def component_line(data):
    return (f"- {data['def_name']} {data['master']} + PLACED "
            f"( {data['x']} {data['y']} ) {data['orient']}\n")


def update_component_start_line(line, inst_by_def_name):
    match = re.match(r"(?P<prefix>\s*-\s+)(?P<name>\S+)(?P<sep>\s+)"
                     r"(?P<master>\S+)(?P<rest>.*)$", line)
    if not match:
        return line, None

    data = inst_by_def_name.get(match.group("name"))
    if data is None:
        return line, None

    rest = match.group("rest").rstrip("\n")
    placement = f"+ PLACED ( {data['x']} {data['y']} ) {data['orient']}"
    if re.search(r"\+\s+(?:PLACED|FIXED|COVER|UNPLACED)\b", rest):
        rest = re.sub(
            r"\+\s+(?:PLACED|FIXED|COVER)\s+\(\s*-?\d+\s+-?\d+\s*\)\s+\S+|\+\s+UNPLACED",
            placement, rest, count=1)
    else:
        rest = f"{rest} {placement}"

    return (f"{match.group('prefix')}{match.group('name')}"
            f"{match.group('sep')}{data['master']}{rest}\n"), data["def_name"]


def write_generated_components(lines, out_path, inst_data):
    insert_at = None
    for idx, line in enumerate(lines):
        if re.match(r"\s*PINS\s+\d+\s*;", line):
            insert_at = idx
            break
    if insert_at is None:
        for idx, line in enumerate(lines):
            if re.match(r"\s*SPECIALNETS\s+\d+\s*;", line):
                insert_at = idx
                break
    if insert_at is None:
        raise ValueError("Cannot find PINS or SPECIALNETS section for COMPONENTS insertion")

    component_lines = [f"COMPONENTS {len(inst_data)} ;\n"]
    for data in inst_data:
        component_lines.append(component_line(data))
        component_lines.append(" ;\n")
    component_lines.append("END COMPONENTS\n\n")

    out_path.write_text("".join(lines[:insert_at] + component_lines +
                                lines[insert_at:]),
                        encoding="utf-8")


def write_def(def_path, out_path, inst_data):
    lines = def_path.read_text(encoding="utf-8").splitlines(keepends=True)
    component_start = None
    component_end = None

    for idx, line in enumerate(lines):
        if component_start is None and re.match(r"\s*COMPONENTS\s+\d+\s*;",
                                                line):
            component_start = idx
        elif component_start is not None and re.match(r"\s*END COMPONENTS\b",
                                                      line):
            component_end = idx
            break

    if component_start is None:
        write_generated_components(lines, out_path, inst_data)
        return {"mode": "generated", "updated_components": len(inst_data)}

    if component_end is None:
        raise ValueError(f"Malformed DEF COMPONENTS section in {def_path}")

    inst_by_def_name = {data["def_name"]: data for data in inst_data}
    updated = set()
    out_lines = lines[:]

    for idx in range(component_start + 1, component_end):
        out_lines[idx], name = update_component_start_line(
            out_lines[idx], inst_by_def_name)
        if name is not None:
            updated.add(name)

    missing = sorted(set(inst_by_def_name) - updated)
    if missing:
        sample = ", ".join(missing[:5])
        raise ValueError(
            f"Failed to update {len(missing)} DEF components in {def_path}: {sample}"
        )

    out_path.write_text("".join(out_lines), encoding="utf-8")
    return {"mode": "updated", "updated_components": len(updated)}


def extract_position_table(placedb, pos, tier, instance_map, libcell_map,
                           inst_to_libcell, suffix_by_tier):
    num_movable = int(placedb.num_movable_nodes)
    num_nodes = int(placedb.num_nodes)
    node_names = [
        normalize_name(node_name)
        for node_name in placedb.node_names[:num_movable]
    ]

    if len(tier) != num_movable:
        raise ValueError(
            f"tier size {len(tier)} does not match movable nodes {num_movable}")
    if len(inst_to_libcell) != num_movable:
        raise ValueError(
            f"txt instance count {len(inst_to_libcell)} does not match movable nodes {num_movable}"
        )

    if hasattr(pos, "detach"):
        pos_values = pos.detach().cpu().numpy()
    else:
        pos_values = pos

    if hasattr(tier, "detach"):
        tier_values = tier.detach().cpu().numpy()
    else:
        tier_values = tier

    inst_data = []
    seen = set()
    for idx, node_name in enumerate(node_names):
        if node_name not in instance_map:
            raise ValueError(f"Missing instance map entry for {node_name}")
        if node_name not in inst_to_libcell:
            raise ValueError(f"Missing txt Inst entry for {node_name}")

        tier_id = int(tier_values[idx])
        if tier_id not in suffix_by_tier:
            raise ValueError(f"Unsupported tier id {tier_id} for {node_name}")

        libcell_key = inst_to_libcell[node_name]
        if libcell_key not in libcell_map:
            raise ValueError(f"Missing libcell map entry for {libcell_key}")

        x = int(round(float(pos_values[idx])))
        y = int(round(float(pos_values[num_nodes + idx])))
        if x <= INVALID_COORD or y <= INVALID_COORD:
            raise ValueError(f"Invalid placement coordinate for {node_name}: {x}, {y}")

        real_name = instance_map[node_name]
        data = {
            "bookshelf_name": node_name,
            "verilog_name": real_name,
            "def_name": def_escape_name(real_name),
            "master": with_tier_suffix(libcell_map[libcell_key],
                                       suffix_by_tier[tier_id]),
            "tier": tier_id,
            "tier_suffix": suffix_by_tier[tier_id],
            "x": x,
            "y": y,
            "orient": "N",
        }
        inst_data.append(data)
        seen.add(node_name)

    missing = sorted(set(inst_to_libcell) - seen)
    if missing:
        sample = ", ".join(missing[:5])
        raise ValueError(f"txt instances missing from placedb: {sample}")
    return inst_data


def validate_diearea(inst_data, diearea):
    if diearea is None:
        return
    xl, yl, xh, yh = diearea
    outside = [
        data for data in inst_data
        if data["x"] < xl or data["x"] > xh or data["y"] < yl or data["y"] > yh
    ]
    if outside:
        sample = ", ".join(f"{d['bookshelf_name']}=({d['x']},{d['y']})"
                           for d in outside[:5])
        raise ValueError(
            f"{len(outside)} placement coordinates are outside DEF diearea {diearea}: {sample}"
        )


def validate_exported_files(verilog_out, def_out, inst_data):
    expected_v = {d["verilog_name"]: d["master"] for d in inst_data}
    found_v = set()
    with open(verilog_out, "r", encoding="utf-8") as f:
        for line in f:
            parsed = verilog_instance_from_line(line)
            if parsed and expected_v.get(parsed["inst"]) == parsed["master"]:
                found_v.add(parsed["inst"])

    missing_v = [
        d["verilog_name"] for d in inst_data if d["verilog_name"] not in found_v
    ]
    if missing_v:
        sample = ", ".join(missing_v[:5])
        raise ValueError(f"Exported Verilog is missing updated instances: {sample}")

    expected_def = {d["def_name"]: d["master"] for d in inst_data}
    found_def = set()
    with open(def_out, "r", encoding="utf-8") as f:
        for line in f:
            match = re.match(r"\s*-\s+(\S+)\s+(\S+)\b", line)
            if match and expected_def.get(match.group(1)) == match.group(2):
                found_def.add(match.group(1))

    missing_def = [
        d["def_name"] for d in inst_data if d["def_name"] not in found_def
    ]
    if missing_def:
        sample = ", ".join(missing_def[:5])
        raise ValueError(f"Exported DEF is missing updated components: {sample}")


def export_openroad_3d_inputs(params, placedb, pos, tier, logger=logging):
    repo_root = Path(__file__).resolve().parents[2]
    txt_path = resolve_path(getattr(params.flatten_2d, "txt_input", None),
                            repo_root)
    if txt_path is None or not txt_path.exists():
        logger.info("OpenROAD 3D export skipped: txt_input is not available")
        return None

    verilog_path = companion_path(params, "openroad_3d_verilog_input",
                                  txt_path, ".v")
    def_path = companion_path(params, "openroad_3d_def_input", txt_path, ".def")
    map_path = companion_path(params, "openroad_3d_map_input", txt_path,
                              "_map.json")
    required = [verilog_path, def_path, map_path]
    if not all(path.exists() for path in required):
        logger.info(
            "OpenROAD 3D export skipped: companion .v/.def/_map.json files were not found for %s",
            txt_path)
        return None

    output_dir = Path(params.result_dir_root) / "openroad_3d"
    output_dir.mkdir(parents=True, exist_ok=True)
    verilog_out = output_dir / verilog_path.name
    def_out = output_dir / def_path.name
    manifest_out = output_dir / "openroad_3d_manifest.json"

    map_data = read_json(map_path)
    instance_map = map_data.get("instance", {})
    libcell_map = map_data.get("libcell", {})
    inst_to_libcell, txt_diearea = parse_txt_instances(txt_path)
    def_diearea = parse_def_diearea(def_path) or txt_diearea

    suffix_by_tier = dict(TIER_SUFFIX)
    inst_data = extract_position_table(placedb, pos, tier, instance_map,
                                       libcell_map, inst_to_libcell,
                                       suffix_by_tier)
    validate_diearea(inst_data, def_diearea)

    write_verilog(verilog_path, verilog_out, inst_data)
    def_result = write_def(def_path, def_out, inst_data)
    validate_exported_files(verilog_out, def_out, inst_data)

    tier_counts = {}
    for data in inst_data:
        tier_counts[str(data["tier"])] = tier_counts.get(str(data["tier"]),
                                                         0) + 1

    manifest = {
        "txt_input": str(txt_path),
        "verilog_input": str(verilog_path),
        "def_input": str(def_path),
        "map_input": str(map_path),
        "verilog_output": str(verilog_out),
        "def_output": str(def_out),
        "component_mode": def_result["mode"],
        "num_instances": len(inst_data),
        "tier_suffix": {str(k): v for k, v in suffix_by_tier.items()},
        "tier_counts": tier_counts,
        "diearea": list(def_diearea) if def_diearea is not None else None,
        "coordinate_range": {
            "x_min": min(data["x"] for data in inst_data),
            "x_max": max(data["x"] for data in inst_data),
            "y_min": min(data["y"] for data in inst_data),
            "y_max": max(data["y"] for data in inst_data),
        },
    }
    manifest_out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    logger.info("OpenROAD 3D export wrote %s and %s", def_out, verilog_out)
    return manifest
