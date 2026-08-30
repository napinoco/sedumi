# sedumi-py — MATLAB/Octave-free port of SeDuMi

This directory holds the in-progress port of SeDuMi to a standalone
C library + Python (NumPy/SciPy) package, with no MATLAB or GNU Octave
runtime dependency.

**New contributor?** Read [`CONTRIBUTING.md`](CONTRIBUTING.md) first —
it has the current phase-by-phase status, the porting workflow this
project follows, known scope limitations, and the prioritized list of
remaining work. (It's in Japanese; ask if you'd like an English
translation.) The summary below is kept only for a quick orientation
and may lag behind CONTRIBUTING.md.

See the phase plan tracked in the project task list:

- **Phase 0** — Verification baseline (this directory's `tools/generate_golden.m`
  and `tests/golden/`): run the existing Octave/MEX build of SeDuMi on the
  problems in `examples/*.mat` and record `x`, `y`, `info` (including the
  DIMACS error vector `info.err`) as the ground truth every later phase is
  checked against.
- **Phase 1** — Strip the `mex.h`/`mxArray` dependency out of the C kernels
  (`blksdp.h` and friends) and build them into a standalone `libsedumi`
  shared library with a plain C API. **Done**: every `.c` file in the
  repository (except the pre-existing dead file `bwblkslv2.c`) now builds
  with `-DSEDUMI_STANDALONE` and links into a single shared library with
  zero MATLAB/Octave/MEX dependency -- see `sedumi_platform.h` and
  `tools/build_libsedumi.sh`. The MEX/Octave build is unchanged and still
  reproduces the Phase 0 golden reference bit-for-bit
  (`tools/check_no_regression.m`).
- **Phase 2** — Python bindings for `libsedumi` (pybind11/CFFI), operating
  directly on `scipy.sparse.csc_matrix` and NumPy arrays.
- **Phase 3** — Port the ~90 `.m` files implementing the interior-point
  method itself to Python, calling into the Phase 2 bindings for the
  performance-critical kernels.
- **Phase 4** — High-level `sedumi(A, b, c, K, pars) -> x, y, info` API and
  `.mat`/SDPA I/O compatibility.
- **Phase 5** — Correctness (against Phase 0 golden data) and performance
  validation.
- **Phase 6** — Packaging (wheels via cibuildwheel) and release.

## Layout

```
python_port/
  tools/
    generate_golden.m   # Phase 0: (re)generate the golden reference data
  tests/
    golden/             # Phase 0 output: golden_*.mat + summary.mat
  sedumi_port/          # (Phase 2+) the Python package itself
```

## Regenerating the golden reference

Requires Octave with the existing SeDuMi MEX/OCT files built
(`install_sedumi` from the repository root):

```
octave-cli --no-gui --eval "cd python_port/tools; generate_golden"
```

This must be re-run (and the new files committed) only if the *reference*
solver output is intentionally expected to change (e.g. a bug fix in the
original MATLAB/Octave code) — never to make a failing Python-port test
pass.
