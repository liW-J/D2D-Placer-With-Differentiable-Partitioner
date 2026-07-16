"""
Adapter for the Differentiable-3D-Partitioner submodule.

D2D owns only the partitioner selection. The differentiable partitioner owns its
own training config, and D2D always feeds it the generated Bookshelf design from
run_tmp/<case>/flattened-2d.
"""
from placer.configure import compile_configurations

import copy
import json
import logging
from pathlib import Path
import sys

import torch

logger = logging.getLogger(__name__)


class Differentiable3DPartitionerBase:
    PARTITIONER_NAMES = {
        "differentiable-3d-partitioner",
        "differentiable_3d_partitioner",
        "differentiable3dpartitioner",
        "differentiable3d",
        "d3d-partitioner",
        "d3d_partitioner",
        "d3d",
    }

    def __init__(self, params, data_2d=None, logger=logging):
        super().__init__()
        self.params = params
        self.data_2d = data_2d
        self.logger = logger
        self.case_name = params.case_name
        self.run_tmp_dir = Path(params.run_tmp_dir_root)
        self.result_dir = Path(params.result_dir_root)
        self.part_file = self.run_tmp_dir / f"{self.case_name}.hgr.part.2"
        aux_input = getattr(params.flatten_2d, "aux_input", "")
        self.aux_input = Path(aux_input) if aux_input else None
        self.project_root = self._resolve_project_root()
        self.work_dir = self.run_tmp_dir / "differentiable-3d-partitioner"
        self.config_path = self.work_dir / "d2d_partitioner_config.yaml"
        self.dreamplace_config_path = (
            self.work_dir / "d2d_bookshelf_input.json")

    @classmethod
    def is_partition_name(cls, partitioner):
        return str(partitioner).lower() in cls.PARTITIONER_NAMES

    def flow(self, hgr_generator_op, parts_reader_op):
        hgr_generator_op(self.case_name)
        binary_assignment = self.partitioning()
        expected_nodes = self._expected_num_movable_nodes()
        self._write_hmetis_style_part_file(binary_assignment, expected_nodes)
        return parts_reader_op(self.case_name)

    def partitioning(self):
        self.work_dir.mkdir(parents=True, exist_ok=True)
        config_path, result_path = self._prepare_flow_config()
        return self._partition_from_bookshelf(config_path, result_path)

    def _partition_from_bookshelf(self, config_path, result_path):
        if self.aux_input is not None and not self.aux_input.exists():
            raise FileNotFoundError(
                f"Bookshelf aux file not found: {self.aux_input}")

        self._prepare_dreamplace_config()
        Differentiable3DPartitionerFlow, DreamplaceParser = (
            self._load_external_api(load_parser=True))

        parser = DreamplaceParser()
        if self.aux_input is not None:
            self.logger.info(
                "Differentiable-3D-Partitioner: reading Bookshelf input %s",
                self.aux_input)
        else:
            self.logger.info(
                "Differentiable-3D-Partitioner: reading LEF/DEF input %s",
                getattr(self.params.flatten_2d, "def_input", ""))
        parser.parse_design(str(self.dreamplace_config_path))
        design = self._build_movable_only_design(parser)

        flow = Differentiable3DPartitionerFlow(
            num_nodes=design["num_nodes"],
            num_nets=design["num_nets"],
            num_pins=design["num_pins"],
            node_pos=parser.node_pos,
            pin_pos=design["pin_pos"],
            flat_net2pin_map=design["flat_net2pin_map"],
            flat_net2pin_start_map=design["flat_net2pin_start_map"],
            pin2node_map=design["pin2node_map"],
            node_size_x=parser.node_size_x,
            node_size_y=parser.node_size_y,
            dreamplace_basic=parser.dreamplace_basic,
            config_path=str(config_path),
            die_xl=parser.die_xl,
            die_yl=parser.die_yl,
            die_xh=parser.die_xh,
            die_yh=parser.die_yh)
        return self._run_flow_and_load_assignment(flow, result_path)

    def _run_flow_and_load_assignment(self, flow, result_path):
        self.logger.info("Differentiable-3D-Partitioner: start flow")
        metrics = flow.run()
        self.logger.info("Differentiable-3D-Partitioner metrics: %s", metrics)

        if not result_path.exists():
            raise RuntimeError(
                "Differentiable-3D-Partitioner did not write binary "
                f"assignment: {result_path}")
        return torch.load(str(result_path), map_location="cpu")

    def _build_movable_only_design(self, parser):
        placedb = getattr(parser, "placedb", None)
        num_movable_nodes = int(
            getattr(placedb, "num_movable_nodes", parser.num_nodes))
        if num_movable_nodes <= 0:
            raise RuntimeError(
                "Differentiable-3D-Partitioner received a design with no "
                "movable nodes")

        pin2node_cpu = parser.pin2node_map.detach().cpu().long()
        flat_net2pin_cpu = parser.flat_net2pin_map.detach().cpu().long()
        start_cpu = parser.flat_net2pin_start_map.detach().cpu().long()
        original_num_pins = int(pin2node_cpu.numel())
        pin_pos_x = parser.pin_pos[:original_num_pins]
        pin_pos_y = parser.pin_pos[original_num_pins:]

        selected_original_pins = []
        new_pin2node = []
        new_flat_net2pin = []
        new_starts = [0]

        for net_id in range(int(parser.num_nets)):
            start = int(start_cpu[net_id].item())
            end = int(start_cpu[net_id + 1].item())
            net_pin_count = 0
            for flat_idx in range(start, end):
                original_pin = int(flat_net2pin_cpu[flat_idx].item())
                node_id = int(pin2node_cpu[original_pin].item())
                if 0 <= node_id < num_movable_nodes:
                    selected_original_pins.append(original_pin)
                    new_pin2node.append(node_id)
                    new_flat_net2pin.append(len(new_pin2node) - 1)
                    net_pin_count += 1

            if net_pin_count:
                new_starts.append(len(new_flat_net2pin))

        if not new_pin2node:
            raise RuntimeError(
                "Differentiable-3D-Partitioner movable-only graph has no "
                "pins; check LEF/DEF connectivity and movable node count")

        device = parser.pin2node_map.device
        selected_pin_tensor = torch.tensor(
            selected_original_pins, dtype=torch.long, device=device)
        pin_pos = torch.cat(
            (pin_pos_x.index_select(0, selected_pin_tensor),
             pin_pos_y.index_select(0, selected_pin_tensor)), dim=0)
        pin2node_map = torch.tensor(
            new_pin2node, dtype=parser.pin2node_map.dtype, device=device)
        flat_net2pin_map = torch.tensor(
            new_flat_net2pin,
            dtype=parser.flat_net2pin_map.dtype,
            device=device)
        flat_net2pin_start_map = torch.tensor(
            new_starts,
            dtype=parser.flat_net2pin_start_map.dtype,
            device=device)
        num_nets = int(flat_net2pin_start_map.numel() - 1)
        num_pins = int(pin2node_map.numel())

        self.logger.info(
            "Differentiable-3D-Partitioner movable graph: "
            "nodes=%d/%d, nets=%d/%d, pins=%d/%d",
            num_movable_nodes, parser.num_nodes, num_nets, parser.num_nets,
            num_pins, parser.num_pins)

        return {
            "num_nodes": num_movable_nodes,
            "num_nets": num_nets,
            "num_pins": num_pins,
            "pin_pos": pin_pos,
            "flat_net2pin_map": flat_net2pin_map,
            "flat_net2pin_start_map": flat_net2pin_start_map,
            "pin2node_map": pin2node_map,
        }

    def _prepare_dreamplace_config(self):
        config = self.params.flatten_2d.toJson()
        if self.aux_input is not None:
            config["aux_input"] = str(self.aux_input)
        else:
            config["aux_input"] = ""
        config["result_dir"] = str(self.result_dir)
        config["plot_flag"] = 0
        config["legalize_flag"] = 0
        config["detailed_place_flag"] = 0

        # D3D uses DREAMPlace density_op during partitioning.  DREAMPlace
        # builds that op from PlaceObj only when global placement is enabled,
        # so create a zero-iteration GP stage to initialize ops without moving
        # the Bookshelf placement generated by D2D.
        stage = {}
        if config.get("global_place_stages"):
            stage.update(config["global_place_stages"][0])
        stage.setdefault("num_bins_x", config.get("num_bins_x", 512))
        stage.setdefault("num_bins_y", config.get("num_bins_y", 512))
        stage.setdefault("learning_rate", 0.01)
        stage.setdefault("wirelength", "weighted_average")
        stage.setdefault("optimizer", "nesterov")
        stage["iteration"] = 0
        config["global_place_stages"] = [stage]
        config["global_place_flag"] = 1
        config["random_center_init_flag"] = 1
        config["gp_noise_ratio"] = 0.0

        self.dreamplace_config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.dreamplace_config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, sort_keys=True)
        return self.dreamplace_config_path

    def _prepare_flow_config(self):
        import yaml

        source_config = self._resolve_config_path()
        with open(source_config, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        if config is None:
            config = {}

        config = copy.deepcopy(config)
        config.setdefault("design", {})["name"] = self.case_name

        result_dir = self.work_dir / "results"
        visual_dir = self.work_dir / "visualizations"
        config.setdefault("output", {})["result_dir"] = str(result_dir)
        config.setdefault("visualization",
                          {})["output_dir"] = str(visual_dir)
        self._normalize_balance_config(config)

        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.config_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(config, f, sort_keys=False)

        result_path = result_dir / self.case_name / "binary_assignment.pt"
        self.logger.info(
            "Differentiable-3D-Partitioner: using config %s", source_config)
        return self.config_path, result_path

    def _normalize_balance_config(self, config):
        partitioner_config = config.setdefault("partitioner", {})
        if partitioner_config.get("ignore_net_degree") is None:
            # Propagate DREAMPlace's high-fanout threshold into the generated
            # partitioner YAML. The partitioner also has its own safe default,
            # but making the value explicit keeps D2D and DREAMPlace aligned.
            partitioner_config["ignore_net_degree"] = int(
                self._get_param("ignore_net_degree", default=100))
        balance_config = partitioner_config.setdefault("balance_loss", {})
        threshold = balance_config.get("threshold_factor", 0.7)
        balance_config.setdefault("top_threshold_factor", threshold)
        balance_config.setdefault("bottom_threshold_factor", threshold)
        balance_config.setdefault("num_bins_x", 4)
        balance_config.setdefault("num_bins_y", 4)

    def _write_hmetis_style_part_file(self, assignment, expected_nodes):
        assignment = torch.as_tensor(assignment).detach().cpu().to(
            torch.int32).view(-1)
        if assignment.numel() < expected_nodes:
            raise RuntimeError(
                "Differentiable-3D-Partitioner assignment is shorter than "
                f"D2D movable nodes: {assignment.numel()} < {expected_nodes}")

        assignment = assignment[:expected_nodes]
        unique = set(int(v) for v in assignment.unique().tolist())
        if not unique.issubset({0, 1}):
            raise RuntimeError(
                "Differentiable-3D-Partitioner produced non-binary "
                f"assignments: {sorted(unique)}")

        self.part_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.part_file, "w", encoding="utf-8") as f:
            for value in assignment.tolist():
                f.write(f"{int(value)}\n")

        top_count = int(assignment.sum().item())
        self.logger.info(
            "Differentiable-3D-Partitioner wrote %s (tier0=%d, tier1=%d)",
            self.part_file, expected_nodes - top_count, top_count)

    def _expected_num_movable_nodes(self):
        if self.data_2d is not None:
            return int(self.data_2d.placedb.num_movable_nodes)
        return int(self.params.flatten_2d.num_movable_nodes)

    def _load_external_api(self, load_parser):
        self._prepare_external_sys_path()
        from partitioner import Differentiable3DPartitionerFlow
        DreamplaceParser = None
        if load_parser:
            from partitioner import DreamplaceParser
        return Differentiable3DPartitionerFlow, DreamplaceParser

    def _prepare_external_sys_path(self):
        loaded = sys.modules.get("partitioner")
        if loaded is not None and hasattr(loaded, "__file__"):
            loaded_path = Path(loaded.__file__).resolve()
            if not self._is_path_under(loaded_path, self.project_root):
                for name in list(sys.modules):
                    if name == "partitioner" or name.startswith(
                            "partitioner."):
                        del sys.modules[name]

        dreamplace_roots = [self.project_root / "thirdparty" / "DREAMPlace"]
        source_root = None
        placer_source_dir = compile_configurations.get("PLACER_SOURCE_DIR")
        if placer_source_dir:
            source_root = Path(placer_source_dir).resolve().parent
            dreamplace_roots.append(source_root / "thirdparty" / "DREAMPlace")

        dreamplace_loaded = any(
            name == "dreamplace" or name.startswith("dreamplace.")
            for name in sys.modules)
        path_entries = [self.project_root]
        if not dreamplace_loaded:
            optional_entries = []
            for dreamplace_root in dreamplace_roots:
                optional_entries.extend([
                    dreamplace_root / "install",
                    dreamplace_root / "install" / "dreamplace",
                    dreamplace_root / "build",
                    dreamplace_root / "build" / "dreamplace",
                    dreamplace_root,
                    dreamplace_root / "dreamplace",
                ])
            if source_root is not None:
                top_build = source_root / "build" / "thirdparty" / "DREAMPlace"
                optional_entries.extend([top_build, top_build / "dreamplace"])
            install_prefix = compile_configurations.get("CMAKE_INSTALL_PREFIX")
            if install_prefix:
                install_root = Path(install_prefix).resolve()
                optional_entries.extend([install_root, install_root / "dreamplace"])
            path_entries.extend(path for path in optional_entries if path.exists())

        for path in reversed(path_entries):
            path_str = str(path)
            if path_str in sys.path:
                sys.path.remove(path_str)
            sys.path.insert(0, path_str)

        self._clear_conflicting_dreamplace_modules(dreamplace_roots)

    def _clear_conflicting_dreamplace_modules(self, dreamplace_roots):
        for name, module in list(sys.modules.items()):
            if name == "thirdparty" or name.startswith("thirdparty.DREAMPlace"):
                del sys.modules[name]
                continue
            if name == "dreamplace" or name.startswith("dreamplace."):
                continue

            module_file = getattr(module, "__file__", None)
            if module_file is None:
                continue

            module_path = Path(module_file).resolve()
            if ("DREAMPlace" in module_path.parts and
                    not any(self._is_path_under(module_path, root)
                            for root in dreamplace_roots)):
                del sys.modules[name]

    def _resolve_project_root(self):
        candidates = []
        placer_thirdparty_dir = compile_configurations.get("PLACER_THIRDPARTY_DIR")
        if placer_thirdparty_dir:
            candidates.append(
                Path(placer_thirdparty_dir) / "Differentiable-3D-Partitioner")

        placer_source_dir = compile_configurations.get("PLACER_SOURCE_DIR")
        if placer_source_dir:
            source_root = Path(placer_source_dir).resolve().parent
            candidates.append(
                source_root / "thirdparty" / "Differentiable-3D-Partitioner")

        install_prefix = compile_configurations.get("CMAKE_INSTALL_PREFIX")
        if install_prefix:
            install_root = Path(install_prefix).resolve()
            candidates.append(
                install_root / "thirdparty" / "Differentiable-3D-Partitioner")

        for candidate in candidates:
            candidate = candidate.resolve()
            if (candidate / "partitioner" / "core" / "flow.py").exists():
                return candidate

        tried = "\n".join(f"  - {candidate}" for candidate in candidates)
        raise FileNotFoundError(
            "Cannot locate the Differentiable-3D-Partitioner submodule. "
            "Expected it under D2D-placer/thirdparty/"
            "Differentiable-3D-Partitioner. Run "
            "`git submodule update --init --recursive` after cloning. "
            "Tried:\n" + tried)

    def _resolve_config_path(self):
        config_root = self._get_param(
            "differentiable_partitioner_config_root",
            "d3d_partitioner_config_root",
            default=None)
        configs_dir = self.project_root / "configs"
        case_config_name = f"{self.case_name}.yaml"

        if config_root:
            root_path = Path(config_root).expanduser()
            if not root_path.is_absolute():
                root_path = configs_dir / root_path
            config_path = root_path / case_config_name
            if config_path.exists():
                return config_path.resolve()
            raise FileNotFoundError(
                f"Differentiable-3D-Partitioner config not found: "
                f"{config_path}")

        direct_config = configs_dir / case_config_name
        if direct_config.exists():
            return direct_config.resolve()

        raise FileNotFoundError(
            "Differentiable-3D-Partitioner config not found. Set "
            "differentiable_partitioner_config_root to one config "
            "subdirectory, e.g. openroad/asap7. Tried:\n"
            f"  - {direct_config}")

    def _get_param(self, *names, default=None):
        holders = (self.params, self.params.flatten_2d)
        for name in names:
            for holder in holders:
                if hasattr(holder, name):
                    value = getattr(holder, name)
                    if value is not None:
                        return value
        return default

    def _is_path_under(self, path, root):
        try:
            path.resolve().relative_to(root.resolve())
            return True
        except ValueError:
            return False
