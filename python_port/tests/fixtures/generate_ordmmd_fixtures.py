"""Generates small, deterministic symmetric sparse test matrices for
cross-validating the ordmmd/symfct Python bindings against the real
Octave/MEX build (the actual oracle, run separately via
tools/generate_ordmmd_oracle.m).

Run from the repository root:
    .venv/bin/python python_port/tests/fixtures/generate_ordmmd_fixtures.py
"""

from pathlib import Path

import numpy as np
import scipy.io
import scipy.sparse

OUT_DIR = Path(__file__).parent / "ordmmd"


def _symmetrize(A):
    A = A.maximum(A.T)
    A.setdiag(1.0)  # ordmmd only looks at the pattern; a nonzero diagonal
    # matches how symbchol.m's ADA_sedumi_ always looks (a Cholesky
    # candidate is never pattern-singular on its diagonal)
    A.eliminate_zeros()
    return scipy.sparse.csc_matrix(A)


def make_fixtures():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cases = {}

    # 1. Tiny hand-built path graph: 0-1-2-3-4 (tridiagonal pattern).
    n = 5
    A = scipy.sparse.lil_matrix((n, n))
    for i in range(n - 1):
        A[i, i + 1] = 1.0
    cases["path5"] = _symmetrize(A)

    # 2. Tiny hand-built star graph: node 0 connected to 1..4.
    n = 5
    A = scipy.sparse.lil_matrix((n, n))
    for i in range(1, n):
        A[0, i] = 1.0
    cases["star5"] = _symmetrize(A)

    # 3. Small random sparse symmetric matrices, several sizes/densities.
    rng = np.random.default_rng(12345)
    for n, density, name in [(10, 0.3, "rand10"), (25, 0.15, "rand25"),
                              (50, 0.08, "rand50")]:
        A = scipy.sparse.random(n, n, density=density, random_state=rng, format="lil")
        cases[name] = _symmetrize(A)

    # 4. A grid graph (5x5), a classic minimum-degree-ordering test shape.
    gx, gy = 5, 5
    n = gx * gy
    A = scipy.sparse.lil_matrix((n, n))

    def idx(x, y):
        return x * gy + y

    for x in range(gx):
        for y in range(gy):
            if x + 1 < gx:
                A[idx(x, y), idx(x + 1, y)] = 1.0
            if y + 1 < gy:
                A[idx(x, y), idx(x, y + 1)] = 1.0
    cases["grid5x5"] = _symmetrize(A)

    for name, A in cases.items():
        scipy.io.savemat(OUT_DIR / f"{name}.mat", {"A": A})
        print(f"wrote {name}.mat: n={A.shape[0]} nnz={A.nnz}")


if __name__ == "__main__":
    make_fixtures()
