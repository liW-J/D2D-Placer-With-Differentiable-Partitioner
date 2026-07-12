'''
Author: JeanneWillis hi@jeannewillis.cn
Date: 2025-06-13 20:00:00
LastEditors: JeanneWillis hi@jeannewillis.cn
LastEditTime: 2026-01-20 02:03:44
FilePath: /D2D-placer/placer/d2d_params.py
Description: 
'''
import dreamplace.Params as Params
import hashlib
import json
import os
import re
import time
from pathlib import Path
from configure import compile_configurations


class D2DDieSpec:

    def __init__(self,
                 numTechnologies=2,
                 dieSizeX=0,
                 dieSizeY=0,
                 topDieMaxUtil=100,
                 bottomDieMaxUtil=100,
                 terminalSizeX=0,
                 terminalSizeY=0,
                 terminalSpacing=0):
        self.numTechnologies = numTechnologies
        self.dieSizeX = dieSizeX
        self.dieSizeY = dieSizeY
        self.topDieMaxUtil = topDieMaxUtil
        self.bottomDieMaxUtil = bottomDieMaxUtil
        self.terminalSizeX = terminalSizeX
        self.terminalSizeY = terminalSizeY
        self.terminalSpacing = terminalSpacing


class D2DParams:

    _VERILOG_BUS_DECL_RE = re.compile(
        r"^(\s*)(input|output|inout|wire)\s+((?:reg\s+)?)"
        r"\[\s*(\d+)\s*:\s*(\d+)\s*\]\s+(.+?)\s*;(\s*(?://.*)?)$")
    _VERILOG_ESCAPED_IDENT_RE = re.compile(r"\\(\S+)")
    _VERILOG_BIT_NAME_RE = re.compile(
        r"(?<![A-Za-z0-9_/.\\$])([A-Za-z_/][A-Za-z0-9_/$/.]*)"
        r"\[(\d+)\]")
    _VERILOG_DOLLAR_NAME_RE = re.compile(
        r"(?<![A-Za-z0-9_/.\\])([A-Za-z_/][A-Za-z0-9_/$/.]*"
        r"\$[A-Za-z0-9_/$/.]*)")
    _DEF_BIT_NAME_RE = re.compile(
        r"([A-Za-z_][A-Za-z0-9_/$/.]*)(?:\\?\[(\d+)\\?\])")
    _DEF_DOLLAR_NAME_RE = re.compile(
        r"(?<![A-Za-z0-9_/.\\])([A-Za-z_][A-Za-z0-9_/$/.]*"
        r"\$[A-Za-z0-9_/$/.]*)")

    def __init__(self, json_path):
        self.json_path = json_path
        self.repo_root = Path(__file__).resolve().parents[1]
        if self.repo_root.name == "install":
            self.project_root = self.repo_root.parent
        else:
            self.project_root = self.repo_root
        self.install_root = self.project_root / "install"

        with open(json_path, "r", encoding="utf-8") as f:
            self.input_config = json.load(f)

        self.tt_format = time.strftime("%Y-%m-%d_%H-%M-%S",
                                       time.localtime())
        self.case_name = os.path.splitext(os.path.basename(json_path))[0]
        self.run_tmp_dir_root = f"{compile_configurations['PLACER_RUNTMP_DIR']}/{self.case_name}"
        self.result_dir_root = f"{compile_configurations['PLACER_RESULT_DIR']}/{self.case_name}/{self.tt_format}"
        self.input_format = self._detect_input_format(self.input_config)
        self.is_lefdef_input = self.input_format == "lefdef"
        self.is_txt_input = self.input_format == "txt"

        self.flatten_2d = Params.Params()
        self.terminal = Params.Params()

        self.flatten_2d.load(json_path)
        self.terminal.load(json_path)
        self.num_tiers = self.flatten_2d.num_tiers
        self.partitioner = getattr(self.flatten_2d, "partitioner", "hmetis")

        self.terminal.aux_input = f"{self.run_tmp_dir_root}/terminal/terminal.aux"

        self.flatten_2d.result_dir = self.result_dir_root
        self.terminal.result_dir = self.result_dir_root

        self.flattened_tier = [Params.Params() for _ in range(self.num_tiers)]
        self.partition_tier = [Params.Params() for _ in range(self.num_tiers)]

        for i in range(self.num_tiers):
            self.flattened_tier[i].load(json_path)
            self.partition_tier[i].load(json_path)
            self.flattened_tier[i].result_dir = self.result_dir_root
            self.partition_tier[i].result_dir = self.result_dir_root
            self.partition_tier[
                i].aux_input = f"{self.run_tmp_dir_root}/partition/tier{i}.aux"

        if self.is_lefdef_input:
            self._lefdef_preprocess_cache = {}
            self._setup_lefdef_inputs()
        else:
            self.flatten_2d.aux_input = f"{self.run_tmp_dir_root}/flattened-2d/flattened-2d.aux"
            for i in range(self.num_tiers):
                self.flattened_tier[
                    i].aux_input = f"{self.run_tmp_dir_root}/flattened-2d/tier{i}.aux"

        self._use_bookshelf_input(self.terminal, self.terminal.aux_input)
        for i in range(self.num_tiers):
            self._use_bookshelf_input(self.partition_tier[i],
                                      self.partition_tier[i].aux_input)

        # special params
        # Use intersection-center positions written by terminal_insert (.pl file)
        # instead of placing all terminals at the die center, so co-place starts
        # with a well-spread initial distribution and avoids the large iter-0
        # density overflow that causes density_weight to accumulate excessively.
        self.terminal.random_center_init_flag = True
        self.terminal.global_place_stages[0]["iteration"] = 1000
        self.terminal.stop_overflow = 0.01

        self.terminal.global_place_flag = True
        self.terminal.legalize_flag = False
        self.terminal.detailed_place_flag = False
        self.terminal.ntuplace_flag = True

        self.flatten_2d.global_place_flag = True
        self.flatten_2d.legalize_flag = False
        self.flatten_2d.detailed_place_flag = False
        self.flatten_2d.ntuplace_flag = True
        # self.flatten_2d.target_density = 2.0

        # co-placement (top cell + bottom cell + terminal optimized together)
        # falls back to legacy iterative die-by-die flow when False
        self.co_place_flag = bool(getattr(self.flatten_2d, "co_place_flag",
                                          True))
        # tunables for the co-place inner GP loop; can be overridden via json
        self.co_place_iteration = int(
            getattr(self.flatten_2d, "co_place_iteration", 1000))
        # Default lowered from 0.10 to 0.07: when terminals start at their
        # intersection-center positions (random_center_init_flag=False), initial
        # die overflow is ~0.097, which is just below the old 0.10 threshold and
        # would trigger an immediate early stop. 0.07 stays below the natural
        # starting die overflow while still allowing early exit once converged.
        self.co_place_stop_overflow = float(
            getattr(self.flatten_2d, "co_place_stop_overflow", 0.07))
        self.co_place_lr = float(getattr(self.flatten_2d, "co_place_lr", 0.01))
        self.co_place_target_density = float(
            getattr(self.flatten_2d, "co_place_target_density", 1.0))
        # plotting cadence for co-place; <=0 disables, otherwise plot every N iters
        self.co_place_plot_freq = int(
            getattr(self.flatten_2d, "co_place_plot_freq", 50))

    def _detect_input_format(self, config):
        input_format = str(config.get("input_format", "")).lower()
        if input_format in ("lefdef", "lef/def", "lef-def"):
            return "lefdef"
        if input_format in ("txt", "iccad", "iccad_txt"):
            return "txt"

        lefdef_markers = (
            "lefdef_input",
            "top_die_tech",
            "bottom_die_tech",
            "top_tech",
            "bottom_tech",
            "top_die_lef_input",
            "bottom_die_lef_input",
            "top_die_def_input",
            "bottom_die_def_input",
        )
        if any(config.get(key) for key in lefdef_markers):
            return "lefdef"
        if config.get("def_input") and config.get("lef_input"):
            return "lefdef"
        return "txt"

    def _unique_roots(self):
        roots = [Path.cwd(), self.repo_root, self.project_root, self.install_root]
        unique = []
        for root in roots:
            if root not in unique:
                unique.append(root)
        return unique

    def _resolve_path(self, path_value):
        if not path_value:
            return ""
        path = Path(str(path_value)).expanduser()
        if path.is_absolute():
            return str(path)
        for root in self._unique_roots():
            candidate = root / path
            if candidate.exists():
                return str(candidate.resolve())
        return str((self.repo_root / path).resolve())

    def _resolve_paths(self, path_value):
        if not path_value:
            return []
        if isinstance(path_value, (str, os.PathLike)):
            values = [path_value]
        else:
            values = path_value
        return [self._resolve_path(value) for value in values]

    def _get_first(self, *values, default=None):
        for value in values:
            if value not in (None, "", []):
                return value
        return default

    def _get_layer_config(self, *names):
        lefdef_config = self.input_config.get("lefdef_input", {})
        if not isinstance(lefdef_config, dict):
            lefdef_config = {}
        for name in names:
            value = lefdef_config.get(name)
            if isinstance(value, dict):
                return value
            value = self.input_config.get(name)
            if isinstance(value, dict):
                return value
        return {}

    def _tech_lef_input(self, tech):
        if not tech:
            return []
        tech_path = Path(str(tech)).expanduser()
        candidates = []
        if tech_path.is_absolute():
            candidates.append(tech_path)
        else:
            candidates.extend([root / tech_path for root in self._unique_roots()])
            candidates.extend([
                self.repo_root / "benchmarks" / "lef" / str(tech),
                self.install_root / "benchmarks" / "lef" / str(tech),
                self.project_root / "install" / "benchmarks" / "lef" /
                str(tech),
            ])

        tech_dir = None
        for candidate in candidates:
            if candidate.is_dir():
                tech_dir = candidate
                break
        if tech_dir is None:
            raise FileNotFoundError(
                "Cannot find LEF tech directory for %s. Provide explicit "
                "lef_input or place LEFs under install/benchmarks/lef/<tech>."
                % tech)

        def lef_sort_key(path):
            name = path.name.lower()
            return (0 if "tech" in name else 1, name)

        lefs = sorted(tech_dir.glob("*.lef"), key=lef_sort_key)
        if not lefs:
            raise FileNotFoundError("No .lef files found under %s" % tech_dir)
        return [str(path.resolve()) for path in lefs]

    def _layer_lefs(self, layer_config, role, fallback=None):
        explicit = self._get_first(
            layer_config.get("lef_input"),
            self.input_config.get("%s_die_lef_input" % role),
            self.input_config.get("%s_lef_input" % role),
            default=None)
        if explicit:
            return self._resolve_paths(explicit)

        tech = self._get_first(
            layer_config.get("tech"),
            self.input_config.get("%s_die_tech" % role),
            self.input_config.get("%s_tech" % role),
            default=None)
        if tech:
            return self._tech_lef_input(tech)
        return list(fallback or [])

    def _layer_def(self, layer_config, role, fallback=None):
        value = self._get_first(
            layer_config.get("def_input"),
            self.input_config.get("%s_die_def_input" % role),
            self.input_config.get("%s_def_input" % role),
            default=fallback)
        return self._resolve_path(value) if value else ""

    def _layer_verilog(self, layer_config, role, fallback=None):
        value = self._get_first(
            layer_config.get("verilog_input"),
            self.input_config.get("%s_die_verilog_input" % role),
            self.input_config.get("%s_verilog_input" % role),
            default=fallback)
        return self._resolve_path(value) if value else ""

    def _use_lefdef_input(self, params, lef_input, def_input, verilog_input=""):
        params.aux_input = ""
        params.lef_input = list(lef_input)
        params.def_input = def_input
        params.verilog_input = verilog_input
        params.sol_file_format = "DEF"

    def _use_bookshelf_input(self, params, aux_input):
        params.aux_input = aux_input
        params.lef_input = ""
        params.def_input = ""
        params.verilog_input = ""

    def _dreamplace_preprocess_enabled(self):
        value = self._get_first(
            self.input_config.get("dreamplace_preprocess_lefdef"),
            self.input_config.get("preprocess_lefdef_for_dreamplace"),
            default=True)
        if isinstance(value, str):
            return value.lower() not in ("0", "false", "no", "off")
        return bool(value)

    def _sanitize_dreamplace_identifier(self, name):
        safe = str(name).strip()
        if safe.startswith("\\"):
            safe = safe[1:]
        safe = safe.replace("\\", "")
        safe = re.sub(r"\[(\d+)\]", r"_\1_", safe)
        safe = safe.replace("$", "_")
        safe = re.sub(r"[^A-Za-z0-9_]", "_", safe)
        safe = re.sub(r"_+", "_", safe)
        if not safe:
            safe = "_"
        if not re.match(r"[A-Za-z_]", safe):
            safe = "_" + safe
        return safe

    def _sanitize_bus_bit(self, base_name, bit_index):
        return self._sanitize_dreamplace_identifier("%s[%s]" %
                                                    (base_name, bit_index))

    def _bus_bit_range(self, msb, lsb):
        step = -1 if msb >= lsb else 1
        return range(msb, lsb + step, step)

    def _preprocess_verilog_for_dreamplace(self, verilog_text):
        bus_ports = {}
        lines = []
        changed = False

        for line in verilog_text.splitlines():
            match = self._VERILOG_BUS_DECL_RE.match(line)
            if not match:
                lines.append(line)
                continue

            indent, kind, qualifier, msb, lsb, names_text, comment = (
                match.groups())
            bits = self._bus_bit_range(int(msb), int(lsb))
            scalars = []
            for raw_name in [n.strip() for n in names_text.split(",")
                             if n.strip()]:
                base_name = raw_name.split("=", 1)[0].strip()
                name_scalars = [
                    self._sanitize_bus_bit(base_name, bit) for bit in bits
                ]
                scalars.extend(name_scalars)
                if kind in ("input", "output", "inout"):
                    bus_ports[base_name] = name_scalars
            lines.append("%s%s %s%s;%s" %
                         (indent, kind, qualifier, ", ".join(scalars),
                          comment))
            changed = True

        text = "\n".join(lines)
        if verilog_text.endswith("\n"):
            text += "\n"

        if bus_ports:
            text, port_changed = self._scalarize_module_ports(text, bus_ports)
            changed = changed or port_changed

        def replace_escaped(match):
            return self._sanitize_dreamplace_identifier(match.group(1))

        def replace_bit(match):
            return self._sanitize_bus_bit(match.group(1), match.group(2))

        def replace_dollar(match):
            return self._sanitize_dreamplace_identifier(match.group(1))

        for pattern, repl in ((self._VERILOG_ESCAPED_IDENT_RE,
                               replace_escaped),
                              (self._VERILOG_BIT_NAME_RE, replace_bit),
                              (self._VERILOG_DOLLAR_NAME_RE,
                               replace_dollar)):
            new_text = pattern.sub(repl, text)
            if new_text != text:
                changed = True
                text = new_text

        return text, changed

    def _scalarize_module_ports(self, verilog_text, bus_ports):
        module_re = re.compile(
            r"\bmodule\s+([A-Za-z_][A-Za-z0-9_]*)\s*\((.*?)\)\s*;",
            re.S)
        changed_any = False

        def replace_module(match):
            nonlocal changed_any
            module_name, port_text = match.groups()
            ports = [port.strip() for port in port_text.split(",")
                     if port.strip()]
            scalar_ports = []
            module_changed = False
            for port in ports:
                if port in bus_ports:
                    scalar_ports.extend(bus_ports[port])
                    module_changed = True
                else:
                    safe_port = (self._sanitize_dreamplace_identifier(port)
                                 if "$" in port else port)
                    scalar_ports.append(safe_port)
                    module_changed = module_changed or safe_port != port
            if not module_changed:
                return match.group(0)
            changed_any = True
            return "module %s (\n\t%s);\n" % (module_name,
                                                 ", \n\t".join(scalar_ports))

        text = module_re.sub(replace_module, verilog_text)
        return text, changed_any

    def _preprocess_def_for_dreamplace(self, def_text):
        changed = False

        def replace_bit(match):
            return self._sanitize_bus_bit(match.group(1), match.group(2))

        def replace_dollar(match):
            return self._sanitize_dreamplace_identifier(match.group(1))

        text = self._DEF_BIT_NAME_RE.sub(replace_bit, def_text)
        if text != def_text:
            changed = True
        new_text = self._DEF_DOLLAR_NAME_RE.sub(replace_dollar, text)
        if new_text != text:
            changed = True
        return new_text, changed

    def _preprocess_lefdef_for_dreamplace(self, def_input, verilog_input):
        if (not self._dreamplace_preprocess_enabled() or not def_input
                or not verilog_input):
            return def_input, verilog_input

        key = (str(Path(def_input).resolve()),
               str(Path(verilog_input).resolve()))
        cached = self._lefdef_preprocess_cache.get(key)
        if cached:
            return cached

        with open(verilog_input, "r", encoding="utf-8") as f:
            verilog_text = f.read()
        with open(def_input, "r", encoding="utf-8") as f:
            def_text = f.read()

        preprocessed_verilog, verilog_changed = (
            self._preprocess_verilog_for_dreamplace(verilog_text))
        preprocessed_def, def_changed = self._preprocess_def_for_dreamplace(
            def_text)

        if not (verilog_changed or def_changed):
            self._lefdef_preprocess_cache[key] = (def_input, verilog_input)
            return def_input, verilog_input

        out_dir = Path(self.run_tmp_dir_root) / "lefdef-dreamplace"
        out_dir.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha1(("%s\0%s" % key).encode("utf-8")).hexdigest()[:10]
        stem = "%s__%s__%s" % (Path(def_input).stem,
                               Path(verilog_input).stem, digest)
        out_def = out_dir / ("%s.def" % stem)
        out_verilog = out_dir / ("%s.v" % stem)
        with open(out_verilog, "w", encoding="utf-8") as f:
            f.write(preprocessed_verilog)
        with open(out_def, "w", encoding="utf-8") as f:
            f.write(preprocessed_def)

        result = (str(out_def), str(out_verilog))
        self._lefdef_preprocess_cache[key] = result
        return result

    def _setup_lefdef_inputs(self):
        flat_config = self._get_layer_config("flattened", "flatten_2d",
                                             "flat")
        top_config = self._get_layer_config("top", "top_die")
        bottom_config = self._get_layer_config("bottom", "bottom_die", "bot")

        common_def = self._resolve_path(self.input_config.get("def_input"))
        common_verilog = self._resolve_path(
            self.input_config.get("verilog_input"))
        common_lefs = self._resolve_paths(self.input_config.get("lef_input"))

        flat_def = self._layer_def(flat_config, "flattened", common_def)
        if not flat_def:
            raise ValueError(
                "LEF/DEF input requires def_input or "
                "lefdef_input.flattened.def_input for flattened 2D placement.")

        flat_lefs = self._layer_lefs(flat_config, "flattened", common_lefs)
        if not flat_lefs:
            flat_lefs = self._layer_lefs(top_config, "top", [])
        if not flat_lefs:
            raise ValueError(
                "LEF/DEF input requires lef_input, flattened tech, or top tech."
            )

        flat_verilog = self._layer_verilog(flat_config, "flattened",
                                           common_verilog)
        top_def = self._layer_def(top_config, "top", flat_def)
        bottom_def = self._layer_def(bottom_config, "bottom", flat_def)
        top_lefs = self._layer_lefs(top_config, "top", flat_lefs)
        bottom_lefs = self._layer_lefs(bottom_config, "bottom", flat_lefs)
        top_verilog = self._layer_verilog(top_config, "top", flat_verilog)
        bottom_verilog = self._layer_verilog(bottom_config, "bottom",
                                             flat_verilog)

        if self.num_tiers != 2:
            raise ValueError("LEF/DEF input currently expects num_tiers == 2")

        flat_def, flat_verilog = self._preprocess_lefdef_for_dreamplace(
            flat_def, flat_verilog)
        top_def, top_verilog = self._preprocess_lefdef_for_dreamplace(
            top_def, top_verilog)
        bottom_def, bottom_verilog = self._preprocess_lefdef_for_dreamplace(
            bottom_def, bottom_verilog)

        self._use_lefdef_input(self.flatten_2d, flat_lefs, flat_def,
                               flat_verilog)
        self._use_lefdef_input(self.flattened_tier[0], top_lefs, top_def,
                               top_verilog)
        self._use_lefdef_input(self.flattened_tier[1], bottom_lefs, bottom_def,
                               bottom_verilog)

        self.lefdef_inputs = {
            "flattened": {
                "lef_input": flat_lefs,
                "def_input": flat_def,
                "verilog_input": flat_verilog,
            },
            "top": {
                "lef_input": top_lefs,
                "def_input": top_def,
                "verilog_input": top_verilog,
            },
            "bottom": {
                "lef_input": bottom_lefs,
                "def_input": bottom_def,
                "verilog_input": bottom_verilog,
            },
        }

    def _config_number(self, *keys, default=0):
        for key in keys:
            value = self.input_config.get(key)
            if value not in (None, ""):
                return value
        return default

    def build_lefdef_die_spec(self):
        terminal_size = self.input_config.get("terminal_size",
                                              self.input_config.get(
                                                  "terminalSize", None))
        if terminal_size:
            terminal_size_x = terminal_size[0]
            terminal_size_y = terminal_size[1]
        else:
            terminal_size_x = self._config_number("terminal_size_x",
                                                  "terminalSizeX",
                                                  default=0)
            terminal_size_y = self._config_number("terminal_size_y",
                                                  "terminalSizeY",
                                                  default=0)

        die_size = self.input_config.get("die_size",
                                         self.input_config.get("dieSize",
                                                               None))
        if die_size:
            die_size_x = die_size[0]
            die_size_y = die_size[1]
        else:
            die_size_x = self._config_number("die_size_x",
                                             "dieSizeX",
                                             default=0)
            die_size_y = self._config_number("die_size_y",
                                             "dieSizeY",
                                             default=0)

        return D2DDieSpec(
            numTechnologies=int(
                self._config_number("num_technologies",
                                    "numTechnologies",
                                    default=self.num_tiers)),
            dieSizeX=int(float(die_size_x)),
            dieSizeY=int(float(die_size_y)),
            topDieMaxUtil=int(
                float(
                    self._config_number("top_die_max_util",
                                        "topDieMaxUtil",
                                        "TopDieMaxUtil",
                                        default=100))),
            bottomDieMaxUtil=int(
                float(
                    self._config_number("bottom_die_max_util",
                                        "bottomDieMaxUtil",
                                        "BottomDieMaxUtil",
                                        default=100))),
            terminalSizeX=int(float(terminal_size_x)),
            terminalSizeY=int(float(terminal_size_y)),
            terminalSpacing=int(
                float(
                    self._config_number("terminal_spacing",
                                        "terminalSpacing",
                                        "TerminalSpacing",
                                        default=0))))

    def finalize_lefdef_die_spec(self, die_spec, dreamplace):
        widths = []
        heights = []
        row_heights = []
        for dp in [dreamplace.dp_2d] + list(dreamplace.dp_tier):
            placedb = dp.placedb
            widths.append(float(placedb.xh) - float(placedb.xl))
            heights.append(float(placedb.yh) - float(placedb.yl))
            if getattr(placedb, "row_height", 0):
                row_heights.append(float(placedb.row_height))

        if die_spec.dieSizeX <= 0 and widths:
            die_spec.dieSizeX = int(round(max(widths)))
        if die_spec.dieSizeY <= 0 and heights:
            die_spec.dieSizeY = int(round(max(heights)))

        default_terminal = int(round(min(row_heights))) if row_heights else 1
        default_terminal = max(default_terminal, 1)
        if die_spec.terminalSizeX <= 0:
            die_spec.terminalSizeX = default_terminal
        if die_spec.terminalSizeY <= 0:
            die_spec.terminalSizeY = default_terminal

    def set_die_place_flags(self,
                            random_center_init_flag=False,
                            global_place_flag=False,
                            legalize_flag=False,
                            detailed_place_flag=False,
                            ntuplace_flag=False):
        """
        @brief Set flags for all tiers.
        @param random_center_init_flag: Whether to use random center initialization.
        @param global_place_flag: Whether to use global placement.
        @param legalize_flag: Whether to use legalization.
        @param detailed_place_flag: Whether to use detailed placement.
        @param ntuplace_flag: Whether to use NTUplace.
        """
        for i in range(self.num_tiers):
            self.partition_tier[
                i].random_center_init_flag = random_center_init_flag
            self.partition_tier[i].global_place_flag = global_place_flag
            self.partition_tier[i].legalize_flag = legalize_flag
            self.partition_tier[i].detailed_place_flag = detailed_place_flag
            self.partition_tier[i].ntuplace_flag = ntuplace_flag
