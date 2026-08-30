"""Port of maxstep.m: computes the maximal step length to the boundary
of the cone K along direction dx from point x."""

from __future__ import annotations

import numpy as np

from . import _native
from .cone import minpsdeig, psdinvscale, tdet


def maxstep(dx, x, auxx: dict, K: dict):
    """tp = maxstep(dx,x,auxx,K): auxx.tdet (Lorentz "t-determinant" of
    x, doubled -- see wregion.m's own `uxc.tdet = 2*vfrm.lab(...)`
    construction) and auxx.u (psdfactor(x,K)) are the per-block
    "reference point" data the step-length formulas are built around."""
    dx = np.asarray(dx, dtype=np.float64).ravel()
    x = np.asarray(x, dtype=np.float64).ravel()
    Kl = int(K["l"])

    mindx = float(np.min(dx[:Kl] / x[:Kl])) if Kl else np.inf

    if len(K.get("q", [])):
        ix = np.asarray(K["mainblks"], dtype=np.int64).ravel()
        i1, i2, i3 = int(ix[0]), int(ix[1]), int(ix[2])
        reltr = x[i1 - 1 : i2 - 1] * dx[i1 - 1 : i2 - 1] - _native.ddot(
            x[i2 - 1 : i3 - 1], dx, K["qblkstart"]
        )
        auxx_tdet = np.asarray(auxx["tdet"]).ravel()
        norm2 = reltr**2 - tdet(dx, K) * auxx_tdet
        if np.all(norm2 > 0):
            norm2 = np.sqrt(norm2)
        mindxq = float(np.min((reltr - norm2) / auxx_tdet))
        mindx = min(mindx, mindxq)

    if len(K.get("s", [])):
        reldx = psdinvscale(auxx["u"], dx, K)
        mindxs = minpsdeig(reldx, K)
        mindx = min(mindx, mindxs)

    return 1.0 / max(-mindx, 1e-16)
