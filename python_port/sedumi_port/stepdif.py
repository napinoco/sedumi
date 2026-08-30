"""Port of stepdif.m: primal-dual step-length differentiation for the
self-dual model. Pure scalar/vector arithmetic -- no cone math or C
kernels needed."""

from __future__ import annotations

import numpy as np


def stepdif(d: dict, R: dict, y0: float, x, y, z, dy0: float, dx, dy, dz, b, mint: float, tpmtd: float):
    """[t,rcdx] = stepdif(d,R,y0,x,y,z,dy0,dx,dy,dz,b,mint,tpmtd)"""
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    z = np.asarray(z, dtype=np.float64).ravel()
    dx = np.asarray(dx, dtype=np.float64).ravel()
    dy = np.asarray(dy, dtype=np.float64).ravel()
    dz = np.asarray(dz, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()

    d0 = np.sqrt(d["l"][0])

    rdx0 = dx[0] / x[0]
    rdy0 = dy0 / y0 - rdx0
    rcdx = float(b @ dy - rdx0 * (b @ y)) - (dz[0] - rdx0 * z[0]) / d0
    rcdx = rdy0 * R["sd"] + rcdx / y0
    gap = R["b0"] * y0

    if tpmtd > 0:
        del1 = float(z @ dx) / gap
        dRg = rdx0 * R["sd"] + rcdx
    else:
        del1 = float(x @ dz) / gap
        dRg = (dy0 / y0) * R["sd"] - rcdx

    usegap = (R["sd"] > 0) or (R["sd"] == 0 and dRg > 0)
    w = R["w"]
    if usegap:
        r0 = w[0] + w[1] + R["sd"]
        beta = (rdy0 * w[0] + rcdx) / r0
    else:
        r0 = w[0] + w[1]
        beta = rdy0 * w[0] / r0

    if tpmtd > 0:
        beta = rdx0 + beta
    else:
        beta = (dy0 / y0) - beta

    def _c(beta):
        return np.array([2 * beta - (rdx0 + del1), 2 * rdx0 * del1 - (rdx0 + del1) * beta])

    c = _c(beta)
    if c[0] <= 0:
        if c[1] >= 0:
            t = abs(tpmtd)
        else:
            t = min(abs(tpmtd), c[0] / c[1])
    else:
        if c[1] >= 0:
            t = mint
        else:
            t = max(mint, c[0] / c[1])

    if dRg != 0:
        tg = -R["sd"] / dRg
    else:
        tg = t

    if tg <= 0:
        if t > 0:
            tg = t
    else:
        if t < 0:
            tg = t

    if abs(t) > abs(tg):
        if usegap:
            beta = rdy0 * w[0] / r0
            alpha = 1 - R["sd"] / r0
        else:
            beta = (rdy0 * w[0] + rcdx) / r0
            alpha = 1 + R["sd"] / r0

        if tpmtd > 0:
            beta = rdx0 * alpha + beta
        else:
            beta = (dy0 / y0) * alpha - beta

        c = _c(beta)
        if t >= 0:
            if c[0] - tg * c[1] <= 0:
                if c[1] >= 0:
                    t = abs(tpmtd)
                else:
                    t = min(abs(tpmtd), c[0] / c[1])
            else:
                t = tg
        else:
            if c[0] - tg * c[1] >= 0:
                if c[1] >= 0:
                    t = mint
                else:
                    t = max(mint, c[0] / c[1])
            else:
                t = tg

    if y0 + t * dy0 <= 0:
        t = -y0 / dy0

    rcdx = y0 * rcdx
    return t, rcdx
