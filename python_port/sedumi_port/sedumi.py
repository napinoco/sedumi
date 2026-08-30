"""Port of sedumi.m: the top-level predictor-corrector interior-point
driver that ties together every other Phase 3-c/3-d piece into one
solve of MINIMIZE c'*x SUCH THAT A*x=b, x in K.

SCOPE (v1): LP and second-order-cone (Lorentz K.q/K.r -- pretransfo.py
already folds rotated K.r cones into standard K.q ones) problems only.
Two restrictions, both inherited from pieces this driver calls rather
than invented here:

  - No PSD blocks: K.s must be empty. sedumi.m's own main loop branches
    on `sum(K.s)==0` for how it updates ADA every iteration; this port's
    getada.py implements only that branch (the else branch needs
    getada1/getada2/getada3 orchestration -- a separate follow-on
    increment, tracked as such in getada.py's own docstring). Raises
    NotImplementedError if K.s is nonempty rather than silently
    answering wrong.

  - No dense-column preconditioning: `dense.cols`/`dense.q` are always
    treated as empty, i.e. getdense.m's detection heuristic is not
    ported and every column is always factored as "sparse". This is a
    *performance* optimization in real SeDuMi, not a correctness
    requirement (the underlying A*P(d)*A' linear system is solved
    identically either way, just via a different -- for problems with
    genuinely dense columns, less numerically favorable -- Cholesky
    conditioning), and every downstream piece this driver calls
    (getdatm.py, pcg.py, deninfac.py) already raises NotImplementedError
    on nonempty dense.cols/dense.q, so this is consistent, not a new gap.

Also not ported (cosmetic/diagnostic, no effect on the returned
(x,y,info)): the console progress printout (my_fprintf/pars.fid),
pars.vplot's v-plot, pars.stopat's interactive debug break, the
optional pre-solve rank/infeasibility diagnostic (a warning heuristic),
and the origcoeff DIMACS error-measures block (info.err).
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from . import _native
from .amul import amul
from .checkpars import checkpars
from .cone import asmDxq, frameit, maxeigK, psdscale
from .deninfac import deninfac
from .getada import getada
from .getdatm import getDAtm
from .optstep import optstep
from .posttransfo import posttransfo
from .pretransfo import pretransfo
from .sdfactor import sdfactor
from .sdinit import sdinit
from .symbchol import symbchol
from .updtransfo import updtransfo
from .wregion import wregion


def _empty_dense(m: int) -> dict:
    return {
        "cols": np.zeros(0, dtype=np.int64),
        "q": np.zeros(0, dtype=np.int64),
        "l": 0,
        "A": sp.csc_matrix((m, 0)),
    }


def sedumi(A, b, c, K: dict, pars: dict | None = None):
    """x, y, info = sedumi(A, b, c, K, pars=None)"""
    A2, b2, c2, K2, prep, _origcoeff = pretransfo(A, b, c, K, pars or {})
    b2 = np.asarray(b2, dtype=np.float64).ravel()
    c2 = np.asarray(c2, dtype=np.float64).ravel()

    if len(K2.get("s", [])):
        raise NotImplementedError(
            "sedumi: PSD blocks (K.s nonempty) are not implemented in this "
            "port yet -- see this module's docstring."
        )

    lponly = int(K2["l"]) == len(c2)
    pars = checkpars(pars)

    dense = _empty_dense(b2.size)

    d, v, vfrm, y, y0, R = sdinit(A2, b2, c2, dense, K2, pars)
    n = vfrm["lab"].size
    Kl = int(K2["l"])

    DAt = getDAtm(A2, dense, d, K2)
    ADA0, _absd0 = getada(A2, K2, d, DAt)
    Lsym = symbchol(ADA0)
    symLden = None

    merit = (float(np.sum(R["w"])) + max(R["sd"], 0.0)) ** 2 * y0 / R["b0"]

    STOP = 0
    iter_ = 0
    wr = {"delta": 0.0, "desc": 1}
    feasratio = 0.0
    xsol = ysol = None
    Lnum = None
    Lsd = {"kcg": 0}
    err = {"kcg": 0}

    while STOP == 0:
        iter_ += 1

        if pars["stepdif"] == 2 and (
            iter_ > 20
            or (iter_ > 1 and (err["kcg"] + Lsd["kcg"] > 3))
            or (iter_ > 5 and abs(1 - feasratio) < 0.05)
        ):
            pars["stepdif"] = 1

        # ---- ADA update + factorization ----
        DAt = getDAtm(A2, dense, d, K2)
        ADA, absd = getada(A2, K2, d, DAt)

        fact = _native.numeric_cholesky(Lsym, ADA, pars["chol"], absd)
        Lnum = {
            "L": fact["L"], "d": fact["d"], "skip": fact["skip"],
            "perm": Lsym["perm"], "xsuper": Lsym["xsuper"], "tmpsiz": Lsym["tmpsiz"],
        }
        Lden, Lnum["d"] = deninfac(symLden, Lnum, dense, DAt, d, absd, K2.get("qblkstart"), pars["chol"])

        Lsd = sdfactor(Lnum, Lden, dense, DAt, d, v, y, A2, c2, K2, R, y0, pars)

        y0Old = y0
        xscl, yNxt, zscl, y0Nxt, w, relt, dxmdz, err, wr = wregion(
            Lnum, Lden, Lsd, d, v, vfrm, A2, DAt, dense, R, K2, y, y0, b2, pars, wr
        )

        if y0Nxt > 0:
            R["b"] = R["b"] + err["b"] / y0Nxt
            R["sd"] = R["sd"] + err["g"] / y0Nxt
            R["b0"] = R["b0"] + err["db0"] / y0Nxt
            y0 = y0Nxt
        else:
            R["b"] = (y0Nxt * R["b"] + err["b"]) / y0Old
            R["sd"] = (y0Nxt * R["sd"] + err["g"]) / y0Old
            R["b0"] = (y0Nxt * R["b0"] + err["db0"]) / y0Old
            R["w"][1] = abs(y0Nxt / y0Old) * R["w"][1]
            R["c"] = (y0Nxt / y0Old) * R["c"]
            R["maxRc"] = float(np.linalg.norm(R["c"], np.inf)) if R["c"].size else 0.0
            y0 = y0Old

        R["maxRb"] = float(np.linalg.norm(R["b"], np.inf)) if R["b"].size else 0.0
        R["w"][0] = 2 * pars["w"][0] * R["maxRb"] / (1 + R["maxb"])
        meritOld = merit
        merit = (float(np.sum(R["w"])) + max(R["sd"], 0.0)) ** 2 * y0 / R["b0"]
        rate = merit / meritOld

        if rate >= 0.9999 and wr["desc"] == 1:
            STOP = -1
            iter_ -= 1
            y0 = y0Old
            break

        feasratio = float(dxmdz[0] / v[0])

        y = yNxt
        by = float(np.sum(b2 * y))
        d, vfrm = updtransfo(xscl, zscl, w, d, K2)
        v = frameit(vfrm["lab"], vfrm["q"], vfrm["s"], K2)
        x0 = float(np.sqrt(d["l"][0]) * v[0])

        if lponly and rate < 0.05:
            xsol_try, ysol_try = optstep(
                A2, b2, c2, y0, y, d, v, dxmdz, K2, Lnum, symLden, dense, feasratio, R, pars
            )
            if xsol_try is not None:
                STOP = 2
                feasratio = 1 - 2 * (xsol_try[0] == 0)
                xsol, ysol = xsol_try, ysol_try
                break
        elif by > 0 and abs(1 + feasratio) < 0.05 and R["b0"] * y0 < 0.5:
            if maxeigK(amul(A2, dense, y, transp=True), K2) <= pars["eps"] * by:
                STOP = 3
                break

        r0 = float(np.sum(R["w"]))
        cx = by + y0 * R["sd"] - x0 / d["l"][0]
        rgap = max(cx - by, 0.0) / max(abs(cx), abs(by), 1e-3 * x0)
        precision1 = y0 * r0 / (1 + x0)
        precision2 = (y0 * r0 + rgap) / x0
        if precision1 < pars["eps"]:
            if precision2 < pars["eps"]:
                STOP = 1
                break
            elif y0 * R["maxRb"] + x0 * R["maxb"] < -pars["eps"] * cx:
                STOP = 1
                break
            elif y0 * R["maxRc"] + x0 * R["maxc"] < pars["eps"] * by:
                STOP = 1
                break

        if iter_ >= pars["maxiter"]:
            STOP = -1

    # ************************************************************
    # FINAL TASKS
    # ************************************************************
    info = {"iter": iter_, "feasratio": feasratio, "pinf": 0, "dinf": 0, "numerr": 0, "r0": np.inf}

    if STOP == 2:
        x = xsol
        y = ysol
    elif STOP == 3:
        x = np.zeros(len(c2))
    else:
        x = np.concatenate([np.sqrt(d["l"]) * v[:Kl], asmDxq(d, v, K2), psdscale(d, v, K2, transp=True)])

    x0 = float(x[0])
    cx = float(np.sum(c2 * x))
    abscx = float(np.sum(np.abs(c2) * np.abs(x)))
    by = float(np.sum(b2 * y))
    Ax = amul(A2, dense, x, transp=False)
    Ay = amul(A2, dense, y, transp=True)
    normy = float(np.linalg.norm(y))
    normx = float(np.linalg.norm(x[1:]))

    pinf = float(np.linalg.norm(x0 * b2 - Ax))
    dinf = float(maxeigK(Ay - x0 * c2, K2))
    if x0 > 0:
        relinf = max(pinf / (1 + R["maxb"]), dinf / (1 + R["maxc"])) / x0
        if relinf > pars["eps"]:
            pdirinf = float(np.linalg.norm(Ax))
            ddirinf = float(maxeigK(Ay, K2))
            reldirinf = pdirinf / (-cx) if cx < 0.0 else np.inf
            if by > 0.0:
                reldirinf = min(reldirinf, ddirinf / by)
            if reldirinf < pars["eps"] or relinf > max(pars["bigeps"], reldirinf):
                x0 = 0.0
                pinf = pdirinf
                dinf = ddirinf

    if x0 > 0:
        x = x / x0
        y = y / x0
        pinf = pinf / x0
        dinf = dinf / x0
        cx = cx / x0
        by = by / x0
        normx = normx / x0
        normy = normy / x0
        if cx <= by:
            r0 = 0.0
        elif cx == 0.0:
            r0 = -by / (R["maxb"] * normy + 1e-10 * x0)
        elif by == 0.0:
            r0 = cx / (R["maxc"] * normx + 1e-10 * x0)
        else:
            r0 = (cx - by) / (abs(by) + 1e-5 * (x0 + abscx))

        denom = np.array([1.0, 1 + R["maxb"] + 1e-3 * R["maxRb"], 1 + R["maxc"] + 1e-3 * R["maxRc"]])
        info["r0"] = float(np.max(np.array([r0, pinf, dinf]) / denom))
        if STOP == -1:
            if info["r0"] > pars["bigeps"]:
                info["numerr"] = 2
            elif info["r0"] > pars["eps"]:
                info["numerr"] = 1
            else:
                info["numerr"] = 0
        else:
            info["r0"] = min(info["r0"], pars["eps"])
    else:
        if pinf < -pars["bigeps"] * cx:
            info["r0"] = abs(pinf / cx)
            info["dinf"] = 1
            abscx = -cx
            pinf = pinf / abscx
            normx = normx / abscx
            x = x / abscx
        if dinf < pars["bigeps"] * by:
            info["r0"] = abs(dinf / by)
            info["pinf"] = 1
            dinf = dinf / by
            normy = normy / by
            y = y / by
        if info["pinf"] + info["dinf"] == 0:
            info["numerr"] = 2
        elif STOP == -1:
            if pinf > -pars["eps"] * cx and dinf > pars["eps"] * by:
                info["numerr"] = 1
            else:
                info["numerr"] = 0

    x, y, _K_out = posttransfo(x, y, prep, K2)

    return x, y, info
