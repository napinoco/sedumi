#!/usr/bin/env bash
# build_libsedumi.sh
#
# Phase 1: builds every SeDuMi C kernel into a single standalone shared
# library with NO MATLAB/Octave/MEX dependency at all (no mex.h, no
# mxArray, nothing). This is the artifact Phase 2 (Python bindings) will
# link against.
#
# bwblkslv2.c is intentionally excluded: it is dead code (not referenced
# by install_sedumi.m or any .m file) that already failed to compile
# before this port -- it #includes a "blkchol.h" that does not exist in
# this repository. Left untouched rather than guessed at.
#
# Usage (from the repository root):
#   python_port/tools/build_libsedumi.sh [output_path]

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
out="${1:-/tmp/libsedumi.so}"

cd "$repo_root"

sources=()
for f in *.c; do
  if [ "$f" = "bwblkslv2.c" ]; then
    continue
  fi
  sources+=("$f")
done

echo "Building $out from ${#sources[@]} source files (SEDUMI_STANDALONE, no mex.h)..."
gcc -DSEDUMI_STANDALONE -O2 -Wall -fPIC -shared -I. "${sources[@]}" -o "$out" -lblas -lm

echo "OK: $out"
nm -D "$out" 2>/dev/null | grep -c ' T ' | xargs -I{} echo "  {} exported functions"
