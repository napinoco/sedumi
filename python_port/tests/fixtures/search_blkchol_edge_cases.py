"""Search for small test matrices that actually exercise blkchol's
pivot-skip and diagonal-stabilization paths (nskip>0 / nadd>0), which
none of the diagonally-dominant fixtures in generate_blkchol_fixtures.py
trigger. Rather than trying to hand-derive exact trigger conditions from
reading blkchol2.c's pivoting logic, this generates a batch of candidate
near-indefinite matrices (small/negative diagonal shifts on a graph
pattern) and reports which ones, when run through the REAL Octave/MEX
blkchol(), actually produce nskip>0 or nadd>0 -- those get promoted to
committed fixtures afterward.

Run from the repository root:
    .venv/bin/python python_port/tests/fixtures/search_blkchol_edge_cases.py
then feed the surviving candidates to Octave to confirm/measure nskip/nadd.
"""

from pathlib import Path

import numpy as np
import scipy.io
import scipy.sparse

OUT_DIR = Path(__file__).parent / "blkchol_edge_candidates"


def _make_near_indefinite(pattern_csc, rng, diag_shift, off_scale=1.0):
    A = pattern_csc.tocoo()
    n = pattern_csc.shape[0]
    vals = rng.standard_normal(A.data.shape[0]) * off_scale
    M = scipy.sparse.coo_matrix((vals, (A.row, A.col)), shape=(n, n)).tocsr()
    M = (M + M.T) * 0.5
    row_abs_sum = np.abs(M).sum(axis=1).A.ravel()
    M = M.tolil()
    for i in range(n):
        # diag_shift can be an array (per-node) or scalar.
        shift = diag_shift[i] if hasattr(diag_shift, "__len__") else diag_shift
        M[i, i] = row_abs_sum[i] * shift
    return M.tocsc()


def make_candidates():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ordmmd_dir = Path(__file__).parent / "ordmmd"
    rng = np.random.default_rng(2024)

    candidates = {}
    for name in ["rand10", "rand25", "grid5x5"]:
        data = scipy.io.loadmat(ordmmd_dir / f"{name}.mat")
        A_pattern = data["A"].tocsc()
        n = A_pattern.shape[0]

        for shift in [0.999, 0.5, 0.1, 0.0, -0.1, -0.5]:
            key = f"{name}_shift{shift}"
            candidates[key] = _make_near_indefinite(A_pattern, rng, shift)

        # A version where only ONE node has a tiny/negative relative diagonal.
        for node_frac, shift in [(0.0, 0.001), (0.0, -0.2)]:
            shifts = np.ones(n)
            shifts[int(node_frac * (n - 1))] = shift
            key = f"{name}_singlenode{shift}"
            candidates[key] = _make_near_indefinite(A_pattern, rng, shifts)

    for name, X in candidates.items():
        scipy.io.savemat(OUT_DIR / f"{name}.mat", {"X": X})

    print(f"wrote {len(candidates)} candidates to {OUT_DIR}")


if __name__ == "__main__":
    make_candidates()
