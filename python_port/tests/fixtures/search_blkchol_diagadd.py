"""Second, more targeted search for a matrix that exercises blkchol's
"diagonal add" (stabilize, don't skip) path specifically -- distinct
from "pure skip" (see search_blkchol_edge_cases.py, which found several
pure-skip fixtures but zero diag-add ones).

Reading blkchol2.c's cholonBlk() pivoting logic directly: a column k is
stabilized-by-adding (not skipped) when, at the time it's eliminated,
its current diagonal xkk satisfies lb[k] < xkk < ub (ub = max(diag)/
maxu^2, a small threshold since maxu=500 by default) AND some entry
below it in the same column has |value| > xkk * maxu. That means: a
small-but-positive diagonal paired with a disproportionately large
off-diagonal coupling in the same column, at elimination time -- so this
tries exactly that shape (one node with a tiny diagonal strongly coupled
to neighbors with larger diagonals), across several magnitudes, since
elimination order (from ordmmd) shifts the effective values enough that
hand-solving the exact threshold isn't reliable -- letting the real
Octave/MEX oracle confirm which ones actually land in the window is
easier than re-deriving genmmd's ordering by hand.
"""

from pathlib import Path

import numpy as np
import scipy.io
import scipy.sparse

OUT_DIR = Path(__file__).parent / "blkchol_diagadd_candidates"


def make_star_with_weak_center(n, center_diag, off_diag_scale, other_diag):
    """Star graph: node 0 (the "center") connects to nodes 1..n-1. Node 0
    gets a tiny diagonal; the others get a large one; edges get a
    moderately large weight -- so whichever node ends up eliminated with
    a tiny surviving diagonal has a strong neighbor coupling relative to
    it.
    """
    M = scipy.sparse.lil_matrix((n, n))
    M[0, 0] = center_diag
    for i in range(1, n):
        M[i, i] = other_diag
        M[0, i] = off_diag_scale
        M[i, 0] = off_diag_scale
    return M.tocsc()


def make_candidates():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    candidates = {}

    for n in [5, 10]:
        for center_diag in [1e-3, 1e-4, 1e-5, 1e-6, 1e-8]:
            for off in [0.5, 1.0, 2.0]:
                for other in [10.0, 25.0, 100.0]:
                    key = f"star{n}_c{center_diag}_o{off}_d{other}"
                    candidates[key] = make_star_with_weak_center(
                        n, center_diag, off, other
                    )

    for name, X in candidates.items():
        scipy.io.savemat(OUT_DIR / f"{name}.mat", {"X": X})

    print(f"wrote {len(candidates)} candidates to {OUT_DIR}")


if __name__ == "__main__":
    make_candidates()
