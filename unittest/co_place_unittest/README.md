# Co-placement anchor regressions

The position assembly in `D2DCoPlace` is shared by CPU and CUDA. The anchor
mapping indices are allocated on the terminal position tensor's device, and
`index_select`/`index_copy_` mirror the current optimization candidate there.
Both GPU AUX wrappers export through the CPU C++ exporters. Consequently the
anchor and pin-origin changes in `7b20e9c` also apply to GPU placement.

Run the synchronization suite:

```sh
python -m unittest discover -s unittest/co_place_unittest -v
```

It runs six CPU tests and, when CUDA is available, the same six tests for each
of float32 and float64. CUDA tests are explicitly skipped without a device.
These tests replace the placement objectives with small analytical models;
they test dispatch and synchronization, not the compiled wirelength/density
kernels.

With an installation built from the current sources, also run:

```sh
python unittest/co_place_unittest/pin_views_integration.py --runtime /path/to/install --device cpu
python unittest/co_place_unittest/pin_views_integration.py --runtime /path/to/install --device cuda:0
```

This exercises compiled AUX export, Bookshelf parsing and PinPos. CUDA mode
moves the actual parsed views onto the GPU before mapping, candidate assembly,
PinPos evaluation and writeback. It verifies cell pin and HBT pin coordinates
for even/odd dimensions, permuted pin indices and fractional translations.
Requesting CUDA without an available GPU fails rather than silently falling
back to CPU. The AUX extensions must be rebuilt after C++ changes; copying only
Python files into an old installation is insufficient.

## CUDA detailed-placement loading regression

The full flow also requires the CUDA host stubs to have hidden visibility.
FindCUDA's NVCC build does not apply `CUDA_VISIBILITY_PRESET` to host code;
the top-level CMake explicitly passes `-fvisibility=hidden` to NVCC's host
compiler. Without it, ISM can work when loaded alone but fail with the complete
operator set loaded (`invalid device ordinal` in CUB); global swap can similarly
report illegal memory access. Rebuild the CUDA operators after changing flags.

```sh
python unittest/co_place_unittest/cuda_detail_integration.py --runtime /path/to/install
```

This small native test loads BasicPlace's full operator set, then runs both ISM
and global swap on a legal four-cell fixture. It requires CUDA and checks finite
results after synchronization. It is separate from the anchor synchronization
regressions because the failure occurs in detailed placement, after co-placement.
