"""Port of sddir.m: direction decomposition for the Ye-Todd-Mizuno self-
dual embedding, given the Lsd factorization from sdfactor.py."""

from __future__ import annotations

import numpy as np

from .cone import frameit
from .pcg import wrapPcg


def sddir(L, Lden, Lsd, pv, d, v, vfrm, At, DAt, dense, R, K, y, y0, b, pars, pMode):
    """[dx,dy,dz,dy0,err] = sddir(...): p = pv is the direction p=dx+dz.
    pMode selects how pv/dy0 are derived: 1 = spectral values w.r.t.
    vfrm (pv is reframed via frameit); 2 = affine scaling (pv=-v ignored,
    dy0=-y0); 3 = pv used directly (dy0 = v'*pv/R.b0)."""
    v = np.asarray(v, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()

    if pMode == 1:
        dy0 = float(vfrm["lab"] @ np.asarray(pv, dtype=np.float64).ravel()) / R["b0"]
        pv = frameit(pv, vfrm["q"], vfrm["s"], K)
    elif pMode == 2:
        dy0 = -y0
        pv = -v
    elif pMode == 3:
        dy0 = float(v @ np.asarray(pv, dtype=np.float64).ravel()) / R["b0"]
        pv = np.asarray(pv, dtype=np.float64).ravel()
    else:
        raise ValueError(f"sddir: unknown pMode {pMode!r}")

    dy, dx, err_kcg, err_b = wrapPcg(
        L, Lden, At, dense, d, DAt, K, dy0 * R["b"], dy0 * Lsd["DRc"] - pv,
        pars["cg"], min(1, y0) * R["maxRb"],
    )

    rdx0 = (y0 * (Lsd["DRc"] @ dx + R["b"] @ dy) - err_b @ y) / Lsd["denom"]

    dy = dy - rdx0 * Lsd["y"]
    dx = rdx0 * Lsd["x"] - dx
    err_b = rdx0 * Lsd["b"] - err_b
    err = {"kcg": err_kcg, "b": err_b, "maxb": float(np.max(np.abs(err_b))) if err_b.size else 0.0}
    dx = dx.copy()
    dx[0] = rdx0 * v[0]
    dz = pv - dx

    return dx, dy, dz, dy0, err
