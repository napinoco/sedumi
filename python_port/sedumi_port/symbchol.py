"""Port of symbchol.m: one-time ordering + symbolic block-sparse Cholesky
factorization of ADA's sparsity pattern. Called once before sedumi.py's
main predictor-corrector loop starts -- the numeric factorization done
every iteration inside the loop (_native.numeric_cholesky, i.e.
blkchol.m) reuses this same L["perm"]/L["L"]/L["xsuper"]/L["tmpsiz"]
every time; only the *numeric values* of ADA change from one iteration
to the next, never its sparsity pattern (ADA = A*diag(positive)*A' for a
fixed A, and a strictly-positive diagonal scaling never changes which
entries are structurally nonzero).

This mirrors symbchol.m's own `spars(ADA)==1` branch exactly (not just
approximately): for a genuinely fully dense ADA (every entry nonzero --
common for small test problems built from dense random data, since
A*diag(positive)*A' for a fully dense A is itself fully dense), real
SeDuMi skips minimum-degree ordering entirely and uses the identity
permutation with one big supernode instead. This is not merely a
performance shortcut that any ordering would emulate equally well:
ordmmd() on a fully-connected graph does NOT generally reduce to the
identity permutation, so using it there would give the numeric
Cholesky a different pivot order than real SeDuMi's, and interior-point
methods are sensitive enough to that to change the exact iteration
count (confirmed by direct comparison against the real Octave build --
this was this port's first sedumi.py oracle mismatch, on dense-A LP
test cases). So this branch is replicated exactly, not approximated.
"""

from __future__ import annotations

import scipy.sparse as sp

from . import _native


def symbchol(ADA) -> dict:
    """L = symbchol(ADA): ADA is a square, symmetric scipy.sparse (or
    dense-convertible) matrix -- only its sparsity pattern matters."""
    ADA_csc = ADA.tocsc() if sp.issparse(ADA) else sp.csc_matrix(ADA)
    m = ADA_csc.shape[0]
    density = (ADA_csc.nnz / (m * m)) if m else 0.0
    if density < 1.0:
        perm = _native.ordmmd(ADA_csc)
        return _native.symbolic_cholesky(ADA_csc, perm)
    return _native.symbolic_cholesky_dense(m)
