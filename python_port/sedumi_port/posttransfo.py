"""Port of posttransfo.m: transforms the solution (x,y) from SeDuMi's
internal format back into the user's original external format, by
applying pretransfo.m's own QR transformation matrix in reverse."""

from __future__ import annotations

import numpy as np


def posttransfo(x, y, prep: dict, K: dict):
    """[xp,yp,K] = posttransfo(x,y,prep,K)

    `x = (x'*prep.QR)'` in the .m file uses MATLAB's conjugate-transpose
    `'` throughout, so with prep.QR complex and x real this actually
    computes `QR.conj().T @ x` (not a plain transpose) -- the internal
    solution x is always real, but the external x can come back complex
    whenever pretransfo.m folded complex problem data into QR.

    Unlike the .m file, this never converts the returned x to a sparse
    array (`if nnz(x)/numel(x) < 1/2: x = sparse(x)`) -- that's a MATLAB
    storage optimization with no effect on the actual solution values,
    and this port's callers work with dense NumPy arrays throughout.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.complex128).ravel()
    K = dict(K)
    K["l"] = K["l"] - 1

    QR = prep["QR"]
    x = np.asarray(QR.conj().T @ x).ravel()

    ycomplex = np.asarray(K.get("ycomplex", np.zeros(0))).ravel()
    if ycomplex.size:
        ylen = y.size - ycomplex.size
        yc = np.zeros(ylen, dtype=np.complex128)
        yc[ycomplex.astype(np.int64) - 1] = 1j * y[ylen:]
        y = y[:ylen] + yc

    if not np.iscomplexobj(x) or not np.any(x.imag):
        x = x.real
    if not np.any(y.imag):
        y = y.real

    return x, y, K
