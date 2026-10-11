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
configuration. The ASAP7 configs select the integrated
Differentiable-3D-Partitioner and load the matching YAML from
`configs/openroad/asap7/`:

```bash
cd install
export LD_LIBRARY_PATH=/path/to/env/lib:$LD_LIBRARY_PATH

python placer/d2d_placer.py test/openroad/asap7/aes.json
```

Results are written to:

```text
install/results/<case>/<timestamp>/
```

Runtime partitioners and detailed placers are selected by the JSON config. The
build installs bundled hMETIS to `install/bin/hmetis`; other flows may still
require external tools such as `openroad`, SpecPart, or NTUPlace. See
[Partitioner Selection](#partitioner-selection).

## What `make install` Installs

```text
install/
  bin/hmetis                               Bundled hMETIS wrapper
  bin/hmetis2.0pre1                        Bundled hMETIS binary
  placer/                                  D2D-Placer Python package and ops
  thirdparty/DREAMPlace/                   Top-level DREAMPlace install tree
  thirdparty/Differentiable-3D-Partitioner/ D3D partitioner source/package
    thirdparty/DREAMPlace/install/         Nested DREAMPlace built for D3D
  benchmarks/                              Installed benchmark inputs
    openroad/asap7/<case>/                  TXT netlists and JSON mappings
  test/                                    Installed JSON configs
    openroad/asap7/                         D3D configs; hmetis/ and tritonpart/ variants
  results/                                 Runtime output directory
  run_tmp/                                 Runtime scratch directory
```

## Partitioner Selection

Set the `partitioner` field in the placement JSON file.

```json
{
  "txt_input": "benchmarks/openroad/asap7/aes/aes.txt",
  "partitioner": "d3d",
  "differentiable_partitioner_config_root": "openroad/asap7"
}
```

Supported values:

- `hmetis`: default fallback; uses the bundled executable installed at
  `bin/hmetis` relative to the runtime working directory.
- `tritonpart`: uses OpenROAD and `placer/scripts/tritonpart.tcl`.
- `bin-based-tritonpart`: uses OpenROAD with placement-aware bin partitioning.
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

For example, `test/openroad/asap7/aes.json` with
`"differentiable_partitioner_config_root": "openroad/asap7"` uses
`configs/openroad/asap7/aes.yaml`. Keep the JSON basename aligned with the
YAML basename when selecting the differentiable partitioner.

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

- `openroad` for TritonPart-based partitioning.
- SpecPart/Julia dependencies for `partitioner: "specpart"`.
- `thirdparty/ntuplace3`, which is installed to `install/thirdparty/ntuplace3`
  when present in the source tree.


## Experimental Results

| Circuit | Nets | Cross-die nets (%) | Total D2D HPWL | Cross-die HPWL | HBT HPWL overhead | Overhead / total HPWL |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| aes | 11,430 | 2,893 (25.3%) | 27,215,653 | 14,239,650 | 3,254,578 | 11.96% |
| ariane133 | 104,987 | 6,776 (6.5%) | 380,315,910 | 173,501,260 | 31,456,288 | 8.27% |
| cva6 | 115,617 | 10,150 (8.8%) | 513,254,138 | 157,968,081 | 60,345,632 | 11.76% |
| ethmac | 61,083 | 11,885 (19.5%) | 142,050,460 | 79,569,786 | 11,803,479 | 8.31% |
| ibex | 11,190 | 1,184 (10.6%) | 33,113,103 | 14,412,018 | 2,404,017 | 7.26% |
| jpeg | 36,070 | 4,829 (13.4%) | 60,007,577 | 22,138,001 | 7,814,889 | 13.02% |
| mock-alu | 14,022 | 1,688 (12.0%) | 31,682,808 | 15,925,548 | 2,818,707 | 8.90% |
| mock-cpu | 39,094 | 2,332 (6.0%) | 48,000,117 | 14,773,983 | 4,983,559 | 10.38% |
| riscv32i | 9,305 | 172 (1.8%) | 27,255,209 | 5,132,677 | 2,864,201 | 10.51% |
| uart | 812 | 126 (15.5%) | 1,308,968 | 424,156 | 123,247 | 9.42% |

HBT HPWL overhead is the increase in cross-die HPWL caused by including hybrid
bonding terminals: cross-die HPWL with terminals minus cross-die HPWL without
terminals.

## Citation

If you use this code or the differentiable partitioning method in your research,
please cite [our paper](https://arxiv.org/abs/2609.06408):

```bibtex
@misc{jiang2026differentiablepartitioning,
  title         = {Differentiable Partitioning With Dual-Max 3D Wirelength Model and Hybrid Bonding Terminal Awareness for Optimized 3D Placement},
  author        = {Liwen Jiang and Xu Shi and Rufeng Xiao and Changhao Yan and Rujun Jiang and Zhiang Wang and Keren Zhu},
  year          = {2026},
  eprint        = {2609.06408},
  archivePrefix = {arXiv},
  url           = {https://arxiv.org/abs/2609.06408}
}
```
