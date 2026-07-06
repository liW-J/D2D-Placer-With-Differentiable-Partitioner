# D2D-Placer with Differentiable 3D Partitioner

D2D-Placer is a 2.5D/3D placement flow built on DREAMPlace. This repository
also integrates `thirdparty/Differentiable-3D-Partitioner`, including its nested
DREAMPlace submodule, so a top-level build can compile and install the complete
flow.

## Quick Start

Run all commands from the repository root unless noted otherwise.

### 1. Fetch submodules

If you cloned without `--recursive`, initialize every submodule, including the
nested DREAMPlace under Differentiable-3D-Partitioner:

```bash
git submodule update --init --recursive
```

### 2. Prepare the build environment

Make sure the following tools and libraries are available from your normal
shell environment: Python 3.9, PyTorch, a C++17 compiler, Boost, Bison, Flex,
Zlib, and CMake 3.x. CMake must be available as `cmake3` or `cmake` in `PATH`,
or passed explicitly with `CMAKE=/path/to/cmake3`.

The Makefile does not assume conda, micromamba, or any named environment. If
your dependencies are installed in a non-standard prefix, pass `ENV_PREFIX` in
the build commands below.

### 3. Build and install

```bash
make -j$(nproc)
make install
```

For dependencies installed in a non-standard prefix:

```bash
make ENV_PREFIX=/path/to/env -j$(nproc)
make ENV_PREFIX=/path/to/env install
```

The default install prefix is `./install`. The top-level `Makefile` also builds
the DREAMPlace submodule inside Differentiable-3D-Partitioner.

If your active environment exposes CMake 4.x and DREAMPlace/Limbo rejects its
policy settings, point the wrapper to a CMake 3.x executable explicitly:

```bash
make ENV_PREFIX=/path/to/env CMAKE=/path/to/cmake3 -j$(nproc)
make ENV_PREFIX=/path/to/env CMAKE=/path/to/cmake3 install
```

### 4. Verify the install

```bash
export LD_LIBRARY_PATH=/path/to/env/lib:$LD_LIBRARY_PATH

(
  cd install
  python - <<'PY'
import importlib
import placer
importlib.import_module("placer.ops.draw_layout_result.draw_layout_result_cpp")
importlib.import_module("thirdparty.DREAMPlace.dreamplace.ops.place_io.place_io_cpp")
print("D2D install import OK")
PY
)

(
  cd install/thirdparty/Differentiable-3D-Partitioner/thirdparty/DREAMPlace/install
  python - <<'PY'
import importlib
import dreamplace
importlib.import_module("dreamplace.ops.place_io.place_io_cpp")
print("nested DREAMPlace install import OK")
PY
)
```

### 5. Run a placement case with the integrated D3D partitioner

Run from the install directory so benchmark and result paths match the installed
configuration. The example below copies `case2.json` to `/tmp`, keeps the file
name as `case2.json` so it matches `configs/iccad2022/case2.yaml`, and enables
the integrated Differentiable-3D-Partitioner:

```bash
cd install
export LD_LIBRARY_PATH=/path/to/env/lib:$LD_LIBRARY_PATH

mkdir -p /tmp/d2d-case2-d3d
python - <<'PY'
import json
from pathlib import Path

src = Path("test/iccad2022/case2.json")
dst = Path("/tmp/d2d-case2-d3d/case2.json")
cfg = json.loads(src.read_text())
cfg["partitioner"] = "d3d"
cfg["differentiable_partitioner_config_root"] = "iccad2022"
dst.write_text(json.dumps(cfg, indent=2) + "\n")
print(dst)
PY

python placer/d2d_placer.py /tmp/d2d-case2-d3d/case2.json
```

Results are written to:

```text
install/results/<case>/<timestamp>/
```

Runtime partitioners and detailed placers are selected by the JSON config. Some
flows require external tools such as `bin/hmetis`, `openroad`, SpecPart, or
NTUPlace. See [Partitioner Selection](#partitioner-selection).

## What `make install` Installs

```text
install/
  placer/                                  D2D-Placer Python package and ops
  thirdparty/DREAMPlace/                   Top-level DREAMPlace install tree
  thirdparty/Differentiable-3D-Partitioner/ D3D partitioner source/package
    thirdparty/DREAMPlace/install/         Nested DREAMPlace built for D3D
  benchmarks/                              Installed benchmark inputs
  test/                                    Installed JSON configs
  results/                                 Runtime output directory
  run_tmp/                                 Runtime scratch directory
```

## Partitioner Selection

Set the `partitioner` field in the placement JSON file.

```json
{
  "txt_input": "benchmarks/iccad2022/case2.txt",
  "partitioner": "d3d",
  "differentiable_partitioner_config_root": "iccad2022"
}
```

Supported values:

- `hmetis`: default fallback; expects an executable at `bin/hmetis` relative to
  the runtime working directory.
- `tritonpart`: uses OpenROAD and `placer/scripts/tritonpart.tcl`.
- `bin-based-tritonpart`: uses OpenROAD with placement-aware bin partitioning.
- `specpart`: uses the SpecPart flow under `thirdparty/HypergraphPartitioning`.
- `d3d`, `d3d_partitioner`, `differentiable-3d-partitioner`: uses the integrated
  Differentiable-3D-Partitioner.

For the differentiable partitioner, D2D looks for YAML configs inside:

```text
install/thirdparty/Differentiable-3D-Partitioner/configs/
```

If `differentiable_partitioner_config_root` is set, D2D loads:

```text
configs/<differentiable_partitioner_config_root>/<case_name>.yaml
```

For example, `test/iccad2022/case2.json` with
`"differentiable_partitioner_config_root": "iccad2022"` uses
`configs/iccad2022/case2.yaml`.

## Useful Build Variables

The top-level `Makefile` accepts these overrides:

```bash
make BUILD_DIR=/tmp/d2d-build INSTALL_PREFIX=/tmp/d2d-install
make CMAKE=/path/to/cmake3 PYTHON_EXECUTABLE=/path/to/python
make ENV_PREFIX=/path/to/env
make CUDA_TOOLKIT_ROOT_DIR=/path/to/cuda
make BOOST_INCLUDE_DIR=/path/to/boost/include BOOST_LIBRARY_DIR=/path/to/boost/lib
```

Common maintenance commands:

```bash
make clean
make distclean
```

`distclean` removes both `build/` and `install/`.

## Troubleshooting

### Missing nested DREAMPlace

If configure reports that the Differentiable-3D-Partitioner DREAMPlace submodule
is missing, run:

```bash
git submodule update --init --recursive
```

### CMake is not found

If `make` reports `env: cmake: No such file or directory`, CMake is not in your
`PATH`. Install CMake 3.x, load it with your site module system, or pass it
explicitly:

```bash
make CMAKE=/path/to/cmake3 -j$(nproc)
make CMAKE=/path/to/cmake3 install
```

If CMake is inside a non-standard dependency prefix, `ENV_PREFIX` is enough as
long as `${ENV_PREFIX}/bin/cmake` or `${ENV_PREFIX}/bin/cmake3` exists:

```bash
make ENV_PREFIX=/path/to/env -j$(nproc)
make ENV_PREFIX=/path/to/env install
```

### CMake 4.x fails in DREAMPlace/Limbo

Some bundled DREAMPlace/Limbo revisions set `CMP0048` to `OLD`, which CMake 4.x
rejects. Use CMake 3.x:

```bash
make CMAKE=/path/to/cmake3 -j$(nproc)
```

If CMake 3.x is not available, a local uncommitted workaround is to change
`CMAKE_POLICY(SET CMP0048 OLD)` to `CMAKE_POLICY(SET CMP0048 NEW)` in both
DREAMPlace checkouts. Do not commit that change unless you intentionally own the
third-party submodule.

### `GLIBCXX_*` import errors

If runtime libraries are installed in a non-standard prefix, put that prefix first:

```bash
export LD_LIBRARY_PATH=/path/to/env/lib:$LD_LIBRARY_PATH
```

### External runtime tool is missing

The build installs D2D, DREAMPlace, and Differentiable-3D-Partitioner. It does
not automatically install every optional runtime binary. Depending on your JSON
config, you may still need to provide:

- `bin/hmetis` for the default HMetis path.
- `openroad` for TritonPart-based partitioning.
- SpecPart/Julia dependencies for `partitioner: "specpart"`.
- `thirdparty/ntuplace3`, which is installed to `install/thirdparty/ntuplace3`
  when present in the source tree.

## Development Notes

- Avoid editing DREAMPlace, nested DREAMPlace, 3d-placer, or other third-party
  submodule source unless the change is intentionally local. These repositories
  are not owned by this wrapper project.
- The integrated build support for Differentiable-3D-Partitioner lives in that
  LiW-J-owned submodule. Commit changes there first, then commit the updated
  submodule pointer in this top-level repository.
- Before handing off a build change, run:

```bash
make -j$(nproc)
make install
```
