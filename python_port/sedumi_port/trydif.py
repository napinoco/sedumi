"""Port of trydif.m: builds w=D(x)z's spectral values at a trial point
and checks its wide-neighborhood membership, falling back to the
previous (t=0) point if it isn't close enough."""

from __future__ import annotations

import numpy as np

from . import _native
from .cone import psdeig, psdfactor, psdscale, tdet
from .neighborhood import iswnbr


def trydif(t, wrIN: dict, wIN: dict, x, z, pars: dict, K: dict):
    """[t,wr,w] = trydif(t,wrIN,wIN,x,z,pars,K)"""
    x = np.asarray(x, dtype=np.float64).ravel()
    z = np.asarray(z, dtype=np.float64).ravel()
    thetaSQR = pars["theta"] ** 2
    Kl = int(K["l"])
    lorN = len(K.get("q", []))

    w: dict = {"tdetx": tdet(x, K), "tdetz": tdet(z, K)}
    detxz = w["tdetx"] * w["tdetz"] / 4

    if lorN == 0:
        lab2q = np.zeros(0)
    else:
        ix = np.asarray(K["mainblks"], dtype=np.int64).ravel()
        i1, i2, i3 = int(ix[0]), int(ix[1]), int(ix[2])
        halfxz = (
            x[i1 - 1 : i2 - 1] * z[i1 - 1 : i2 - 1]
            + _native.ddot(x[i2 - 1 : i3 - 1], z, K["qblkstart"])
        ) / 2
        tmp = halfxz**2 - detxz
        if np.all(tmp > 0):
            lab2q = halfxz + np.sqrt(tmp)
        else:
            lab2q = halfxz

    w["ux"], _ = psdfactor(x, K)
    w["s"] = psdscale(w["ux"], z, K)
    w["lab"] = np.concatenate([x[:Kl] * z[:Kl], detxz / lab2q, lab2q, psdeig(w["s"], K)])

    delta, h, alpha = iswnbr(w["lab"], thetaSQR)
    wr = {"delta": delta, "h": h, "alpha": alpha, "desc": wrIN["desc"]}

    if wr["delta"] > pars["beta"]:
        t = 0.0
        w = wIN
        wr = wrIN

    return t, wr, w
