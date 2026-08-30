"""Port of sdfactor.m: factors the self-dual embedding's extra (x0,z0)
row/column, producing the Lsd struct sddir.py needs on every iteration."""

from __future__ import annotations

import numpy as np

from .cone import asmDxq, psdscale
from .pcg import wrapPcg


def sdfactor(L, Lden, dense, DAt, d, v, y, At, c, K, R, y0, pars):
    """Lsd = sdfactor(L,Lden,dense,DAt,d,v,y,At,c,K,R,y0,pars)"""
    Kl = int(K["l"])
    v = np.asarray(v, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    c = np.asarray(c, dtype=np.float64).ravel()

    Rc = R["c"]
    DRc = np.concatenate([np.sqrt(d["l"]) * Rc[:Kl], asmDxq(d, Rc, K), psdscale(d, Rc, K)])

    Lsd_y, Lsd_x, Lsd_kcg, Lsd_b = wrapPcg(
        L, Lden, At, dense, d, DAt, K, y0 * R["b"], y0 * DRc - 2 * v,
        pars["cg"], min(1, y0) * R["maxRb"],
    )

    Lsd_y = Lsd_y - y
    Lsd_x = Lsd_x + v
    Lsd_denom = float(np.sum(Lsd_x**2) + Lsd_b @ Lsd_y)

    return {
        "DRc": DRc,
        "y": Lsd_y,
        "x": Lsd_x,
        "kcg": Lsd_kcg,
        "b": Lsd_b,
        "denom": Lsd_denom,
    }
