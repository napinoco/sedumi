"""Generates small, deterministic, well-conditioned SPD sparse test
matrices for cross-validating the numeric_cholesky() (blkchol.c) binding
against the real Octave/MEX build.

Run from the repository root:
    .venv/bin/python python_port/tests/fixtures/generate_blkchol_fixtures.py
"""

from pathlib import Path

import numpy as np
import scipy.io
import scipy.sparse

OUT_DIR = Path(__file__).parent / "blkchol"


def _make_spd_like(pattern_csc, rng, diag_boost=2.0):
    """Fill a given sparsity pattern (symmetric, as produced for the
    ordmmd/symfct fixtures) with random values, then make it strongly
    diagonally dominant (hence SPD) by boosting the diagonal -- so the
    "happy path" (no skipped/stabilized pivots) is exercised first.

    Stores BOTH triangles explicitly (as symbchol.m's ADA_sedumi_ always
    is in real use): ordmmd_/symfct_ build an adjacency list from
    whichever entries are actually stored, and require every edge to
    appear in both endpoints' adjacency lists, so a lower-triangle-only
    matrix is not a valid input even though it carries the same
    mathematical symmetric matrix.
    """
    A = pattern_csc.tocoo()
    n = pattern_csc.shape[0]
    vals = rng.standard_normal(A.data.shape[0]) * 0.1
    M = scipy.sparse.coo_matrix((vals, (A.row, A.col)), shape=(n, n)).tocsr()
    M = (M + M.T) * 0.5
    row_abs_sum = np.abs(M).sum(axis=1).A.ravel()
    M = M.tolil()
    for i in range(n):
        M[i, i] = row_abs_sum[i] + diag_boost
    return M.tocsc()


def make_fixtures():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(999)

    ordmmd_dir = Path(__file__).parent / "ordmmd"
    for name in ["path5", "star5", "rand10", "rand25", "grid5x5"]:
        data = scipy.io.loadmat(ordmmd_dir / f"{name}.mat")
        A_pattern = data["A"].tocsc()
        X = _make_spd_like(A_pattern, rng)
        scipy.io.savemat(OUT_DIR / f"{name}_spd.mat", {"X": X})
        print(f"wrote {name}_spd.mat: n={X.shape[0]} nnz={X.nnz}")


if __name__ == "__main__":
    make_fixtures()
