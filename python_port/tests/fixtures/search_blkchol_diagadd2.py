"""Third search attempt for a genuine diag-add (nadd>0) case.

Debugging via a hand-built 3-node example (see conversation/commit
history) pinned down the exact mechanism in blkchol2.c's cholonBlk():
a column is "added to" (stabilized) rather than "purely skipped" when,
at ITS OWN elimination time (i.e. after Schur-complement updates from
already-eliminated neighbors), its diagonal has shrunk to something
SMALL BUT STILL POSITIVE, while some later-eliminated neighbor's
coupling in the same column is disproportionately large (> maxu=500x).
If the shrinkage instead pushes the diagonal negative, that lands in
the *pure-skip* branch instead (which the first search already found
several examples of).

Precisely engineering "small but still positive" requires knowing the
elimination order (from ordmmd/genmmd) in advance, which isn't practical
to predict by hand -- so this casts a wide net: many random small graphs
with one deliberately weak node and a mix of small/large edge weights,
relying on the real Octave/MEX oracle to say which land in the window.
"""

from pathlib import Path

import numpy as np
import scipy.io
import scipy.sparse

OUT_DIR = Path(__file__).parent / "blkchol_diagadd_candidates2"


def make_candidate(n, rng):
    M = scipy.sparse.lil_matrix((n, n))
    weak = rng.integers(0, n)
    diag = rng.uniform(5, 50, size=n)
    diag[weak] = 10 ** rng.uniform(-8, -2)
    for i in range(n):
        M[i, i] = diag[i]
    # Random sparse edges, mixing small and large weights, always
    # touching the weak node at least twice (once "early", once "late"
    # in whatever order ordmmd picks -- we don't control which, so cover
    # both by giving it several neighbors of varying edge weight).
    n_edges = rng.integers(n, 2 * n)
    for _ in range(n_edges):
        i, j = rng.choice(n, size=2, replace=False)
        w = 10 ** rng.uniform(-1, 2) * rng.choice([-1, 1])
        M[i, j] = w
        M[j, i] = w
    for i in range(n):
        M[i, i] = diag[i]  # re-assert after edge writes
    return M.tocsc()


def make_candidates():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(7)
    n_written = 0
    for trial in range(400):
        n = int(rng.integers(4, 8))
        X = make_candidate(n, rng)
        scipy.io.savemat(OUT_DIR / f"cand{trial:04d}.mat", {"X": X})
        n_written += 1
    print(f"wrote {n_written} candidates to {OUT_DIR}")


if __name__ == "__main__":
    make_candidates()
