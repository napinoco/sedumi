"""Port of widelen.m: computes an approximate wide-region neighborhood
step length via a bounded bisection line search, doing the extensive
search only when it pays off (final rate at most twice the best
possible, step length at least half the best possible)."""

from __future__ import annotations

import numpy as np

from . import _native
from .cone import psdeig, psdfactor, psdscale, tdet
from .neighborhood import iswnbr


def _build_w(xM, zM, K: dict):
    Kl = int(K["l"])
    w: dict = {"tdetx": tdet(xM, K), "tdetz": tdet(zM, K)}
    detxz = w["tdetx"] * w["tdetz"] / 4

    lorN = len(K.get("q", []))
    if lorN == 0:
        lab2q = np.zeros(0)
    else:
        ix = np.asarray(K["mainblks"], dtype=np.int64).ravel()
        i1, i2, i3 = int(ix[0]), int(ix[1]), int(ix[2])
        halfxz = (
            xM[i1 - 1 : i2 - 1] * zM[i1 - 1 : i2 - 1]
            + _native.ddot(xM[i2 - 1 : i3 - 1], zM, K["qblkstart"])
        ) / 2
        tmp = halfxz**2 - detxz
        if np.all(tmp > 0):
            lab2q = halfxz + np.sqrt(tmp)
        else:
            lab2q = halfxz

    w["ux"], _ = psdfactor(xM, K)
    w["s"] = psdscale(w["ux"], zM, K)
    w["lab"] = np.concatenate([xM[:Kl] * zM[:Kl], detxz / lab2q, lab2q, psdeig(w["s"], K)])
    return w


def widelen(xc, zc, y0: float, dx, dz, dy0: float, d2y0: float, maxt: float, pars: dict, K: dict):
    """[t,wr,w] = widelen(xc,zc,y0,dx,dz,dy0,d2y0,maxt,pars,K)"""
    xc = np.asarray(xc, dtype=np.float64).ravel()
    zc = np.asarray(zc, dtype=np.float64).ravel()
    dx = np.asarray(dx, dtype=np.float64).ravel()
    dz = np.asarray(dz, dtype=np.float64).ravel()
    thetaSQR = pars["theta"] ** 2

    if dy0 < -1e-5 * y0:
        if d2y0 < 0:
            fullt = 2 * y0 / (-dy0 + np.sqrt(dy0**2 - 4 * y0 * d2y0))
        else:
            fullt = y0 / (-dy0)
        if fullt <= 0:
            raise AssertionError("widelen: fullt <= 0")
    else:
        fullt = 2 * maxt

    tR = min(maxt, fullt)
    if tR < 0:
        raise AssertionError("widelen: tR >= 0")

    t = 0.0
    ntry = 0
    w = wM = None
    deltaM = hM = alphaM = None

    while (t < 0.5 * tR) or ((fullt - tR) + (1e-7 * fullt) < (tR - t)) or ntry == 0:
        ntry = 1
        tM = 0.1 * t + 0.9 * tR if tR == maxt else 0.5 * (t + tR)
        xM = xc + tM * dx
        zM = zc + tM * dz
        wM = _build_w(xM, zM, K)
        deltaM, hM, alphaM = iswnbr(wM["lab"], thetaSQR)

        if (deltaM <= pars["beta"]) or ((tM < fullt / 10) and (deltaM < 1)):
            w = wM
            t = tM
            wr = {"h": hM, "alpha": alphaM, "delta": deltaM}
        else:
            tR = tM

    if t == 0:
        w = wM
        t = tM
        wr = {"h": hM, "alpha": alphaM, "delta": deltaM}

    wr["desc"] = 1  # always descent direction
    return t, wr, w
