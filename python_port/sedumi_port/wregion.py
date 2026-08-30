"""Port of wregion.m: the Sturm-Zhang wide-region interior-point method
-- one full predictor(+corrector) step, tying together nearly every
other Phase 3-c piece (sddir, maxstep, widelen, stepdif, trydif) plus
several cone.py primitives (tdet, psdfactor, qinvjmul, qjmul, psdjmul,
frameit) and the native psdinvjmul binding."""

from __future__ import annotations

import numpy as np

from . import _native
from .cone import frameit, psdfactor, psdjmul, qinvjmul, qjmul, tdet
from .maxstep import maxstep
from .sddir import sddir
from .stepdif import stepdif
from .trydif import trydif
from .widelen import widelen


def wregion(L, Lden, Lsd, d, v, vfrm, A, DAt, dense, R, K, y, y0, b, pars, wr):
    """[xscl,y,zscl,y0,w,relt,dxmdz,err,wr] = wregion(...)"""
    v = np.asarray(v, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    Kl = int(K["l"])
    n = vfrm["lab"].size

    STOP = 0
    dxmdz = None
    err = None

    # ---- initial centering ----
    if wr["delta"] > 0.0:
        vTAR = (1 - wr["alpha"]) * np.maximum(wr["h"], vfrm["lab"])
        pv = 2 * (vTAR - vfrm["lab"])
        pMode = 1
        dx, dy, dz, dy0, errc = sddir(L, Lden, Lsd, pv, d, v, vfrm, A, DAt, dense, R, K, y, y0, b, pars, pMode)

        xc = v + dx
        zc = v + dz
        yc = y + dy
        y0c = y0 + dy0
        uxc = {"tdet": tdet(xc, K)}
        uzc = {"tdet": tdet(zc, K)}
        uxc["u"], xispos = psdfactor(xc, K)
        uzc["u"], zispos = psdfactor(zc, K)

        critval = max(y0, np.sqrt(min(d["l"][0], 1 / d["l"][0])) * v[0])
        critval = max(1e-3, pars["cg"]["restol"]) * critval * R["maxRb"]
        if (
            (not xispos)
            or (not zispos)
            or (errc["maxb"] > critval)
            or (uxc["tdet"].size and np.min(uxc["tdet"]) <= 0.0)
            or (uzc["tdet"].size and np.min(uzc["tdet"]) <= 0.0)
        ):
            STOP = -1
            dxmdz = None
            err = errc
        pv = -vTAR
    else:
        vTAR = vfrm["lab"]
        xc = v
        ix = np.asarray(K["mainblks"], dtype=np.int64).ravel()
        i1, i2 = int(ix[0]), int(ix[1])
        uxc = {"tdet": 2 * vfrm["lab"][i1 - 1 : i2 - 1] * vfrm["lab"][i2 - 1 : 2 * i2 - i1 - 1]}
        uxc["u"], _ = psdfactor(xc, K)
        zc = v
        uzc = uxc
        yc = y
        y0c = y0
        errc = {"b": np.zeros(y.size), "maxb": 0.0, "db0": 0.0}
        pv = None  # means pv = -v (sddir's pMode==2 handles this itself)
        pMode = 2

    # ---- predictor (+ corrector) ----
    if STOP != -1:
        dx, dy, dz, dy0, err = sddir(L, Lden, Lsd, pv, d, v, vfrm, A, DAt, dense, R, K, y, y0, b, pars, pMode)
        dxmdz = dx - dz

        if pars["alg"] != 0:
            pMode = 3
            gd1 = np.concatenate(
                [
                    dxmdz[:Kl] / vTAR[:Kl],
                    qinvjmul(vTAR, vfrm["q"], dxmdz, K),
                    _native.psdinvjmul(vTAR, vfrm["s"], dxmdz, K),
                ]
            )
            maxt1 = min(maxstep(dx, xc, uxc, K), maxstep(dz, zc, uzc, K))

            if pars["alg"] == 1:  # v-expansion (Sturm-Zhang)
                tTAR = 1.0 - (1.0 - maxt1)
                pv = tTAR**2 * np.concatenate(
                    [gd1[:Kl] * dxmdz[:Kl], qjmul(gd1, dxmdz, K), psdjmul(gd1, dxmdz, K)]
                )
                pv2 = 2 * tTAR * (1 - tTAR) * ((np.sum(vTAR) / n) * np.ones(n) - vTAR) - (2 * tTAR) * vTAR
            elif pars["alg"] == 2:  # v^2-expansion (Mehrotra)
                tTAR = 1.0 - (1.0 - maxt1) ** 3
                pv = (tTAR / 4) * np.concatenate(
                    [gd1[:Kl] * dxmdz[:Kl], qjmul(gd1, dxmdz, K), psdjmul(gd1, dxmdz, K)]
                )
                pv2 = ((1 - tTAR) * tTAR * R["b0"] * y0 / n) / vTAR - (1 + tTAR / 4) * vTAR
            else:
                raise ValueError(f"wregion: unsupported pars['alg'] {pars['alg']!r}")

            pv = pv + frameit(pv2, vfrm["q"], vfrm["s"], K)
            dx, dy, dz, dy0, err = sddir(
                L, Lden, Lsd, pv, d, v, vfrm, A, DAt, dense, R, K, y, y0, b, pars, pMode
            )

        PHI = 0.5
        if dy0 < 0 and (PHI * dy0**2 * R["maxRb"]) != 0:
            critval = -(PHI * dy0 * R["maxRb"] + err["maxb"]) * y0c / (PHI * dy0**2 * R["maxRb"])
        else:
            critval = 1.0

        if critval <= 0:
            STOP = -1
        else:
            tp = maxstep(dx, xc, uxc, K)
            td = maxstep(dz, zc, uzc, K)
            if dy0 < 0:
                tp = min(tp, critval)
            if xc[0] + td * dx[0] < 0:
                td = xc[0] / (-dx[0])
            maxt = min(tp, td)

            t, wr, w = widelen(xc, zc, y0c, dx, dz, dy0, 0.0, maxt, pars, K)

            xscl = xc + t * dx
            y = yc + t * dy
            zscl = zc + t * dz
            y0 = y0c + t * dy0

            if pars.get("stepdif") == 1:
                tdif, rcdx = stepdif(d, R, y0, xscl, y, zscl, dy0, dx, dy, dz, b, -t, tp - td)
                if tdif != 0:
                    rdx0 = dx[0] / xscl[0]
                    mu = 1 + tdif * rdx0
                    if tp > td:
                        newx = xscl + tdif * dx
                        newz = mu * zscl
                    else:
                        newx = mu * xscl
                        newz = zscl + tdif * dz
                    tdif, wr, w = trydif(tdif, wr, w, newx, newz, pars, K)
            else:
                tdif = 0

            if tdif != 0:
                rdy0 = dy0 - rdx0 * y0
                zscl = newz
                xscl = newx
                if tp > td:
                    y = mu * y
                    y0 = mu * y0
                    err["b"] = (tdif * rdy0) * R["b"] + errc["b"] + (t + tdif) * err["b"]
                    err["g"] = tdif * rcdx
                    relt = {"p": (t + tdif) / tp, "d": t / td}
                else:
                    y = y + tdif * dy
                    y0 = y0 + tdif * dy0
                    err["b"] = -(tdif * rdy0) * R["b"] + mu * (errc["b"] + t * err["b"])
                    err["g"] = -tdif * rcdx
                    relt = {"p": t / tp, "d": (t + tdif) / td}
            else:
                err["b"] = errc["b"] + t * err["b"]
                err["g"] = 0.0
                relt = {"p": t / maxt, "d": t / maxt}

            wr["tpmtd"] = tp - td
            err["maxb"] = errc["maxb"] + t * err["maxb"]
            err["db0"] = float(xscl @ zscl) - y0 * R["b0"]

    if STOP == -1:
        relt = {"p": 0.0, "d": 0.0}
        w = None
        xscl = None
        zscl = None
        err["b"] = np.zeros_like(b)
        err["db0"] = 0.0
        err["g"] = 0.0

    return xscl, y, zscl, y0, w, relt, dxmdz, err, wr
