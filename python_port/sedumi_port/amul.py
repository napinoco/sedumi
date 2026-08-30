"""Port of Amul.m: y = A*x or A'*x, with a correction for columns that
were pulled out of the sparse A into a separate dense matrix (the
"dense columns" bookkeeping -- see dense.cols/dense.A, produced by
getdense.m, not yet ported; a dense dict with empty `cols` skips the
correction entirely, same as the .m file's own `if ~isempty(dense.cols)`
guard)."""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp


def amul(At, dense: dict, x, transp: bool = False):
    """y = amul(At,dense,x,transp): transp=False computes A*x (x has
    length m, y has length N); transp=True computes A'*x (x has length
    N, y has length m). At is stored transposed (N x m), matching
    SeDuMi's own convention."""
    x = np.asarray(x).ravel()
    if not transp:
        y = np.asarray(At.T @ x).ravel() if sp.issparse(At) else (At.T @ x)
    else:
        y = np.asarray(At @ x).ravel() if sp.issparse(At) else (At @ x)

    cols = np.asarray(dense.get("cols", np.zeros(0, dtype=np.int64))).ravel()
    if cols.size:
        idx = cols.astype(np.int64) - 1
        A = dense["A"]
        if not transp:
            y = y + np.asarray(A @ x[idx]).ravel()
        else:
            y = y.copy()
            y[idx] = np.asarray(A.T @ x).ravel()
    return y
