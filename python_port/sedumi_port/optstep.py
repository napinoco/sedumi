"""Port of optstep.m: Mehrotra-Ye style optimality-face projection,
tried once SeDuMi's iteration enters the superlinear convergence
region for a pure-LP problem.

SCOPE: only the sum(K.s)==0 path is implemented. This is not a
simplification of convenience -- it's the ONLY path optstep.m's real
caller (sedumi.m) ever reaches: sedumi.m calls optstep.m exactly once,
gated by `if lponly && (rate < 0.05)`, and `lponly = (K.l==length(c))`
forces K.q and K.s empty too. The sum(K.s)!=0 branch (getada1/getada2/
getada3) is genuine dead code in real SeDuMi, so it isn't ported here;
raises NotImplementedError if K.q or K.s is nonempty instead of
silently answering wrong.

The Ablkjc/Aord/ADA_sedumi_ parameters optstep.m takes are also dropped
from this port's signature: Ablkjc is unused by this port's getDAtm()
(which doesn't need the extractA-based block-partition table -- see
getdatm.py), and Aord/ADA_sedumi_ are only used by the unreachable
getada1/getada2/getada3 branch.
"""

from __future__ import annotations

import numpy as np

from . import _native
from .amul import amul
from .deninfac import deninfac
from .getada import getada
from .getdatm import getDAtm
from .pcg import wrapPcg


def optstep(A, b, c, y0: float, y, d: dict, v, dxmdz, K: dict, L: dict, symLden, dense: dict, feasratio: float, R: dict, pars: dict):
    """[x,y] = optstep(A,b,c,y0,y,d,v,dxmdz,K,L,symLden,dense,feasratio,R,pars)"""
    if len(K.get("q", [])) or len(K.get("s", [])):
        raise NotImplementedError(
            "optstep: only the LP-only path (K.q and K.s empty) is implemented -- "
            "see this module's docstring."
        )

    if abs(abs(feasratio) - 1) >= 0.1:
        return None, None

    b = np.asarray(b, dtype=np.float64).ravel()
    c = np.asarray(c, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    v = np.asarray(v, dtype=np.float64).ravel()
    dxmdz = np.asarray(dxmdz, dtype=np.float64).ravel()

    x0 = np.sqrt(d["l"][0]) * v[0]
    z0 = x0 / d["l"][0]
    if feasratio < -0.5 and x0 < z0 * z0:
        x0 = 0.0

    lpNB = np.nonzero(dxmdz < 0)[0]
    d = dict(d)
    d["l"] = d["l"].copy()
    d["l"][lpNB] = 0.0

    DAt = getDAtm(A, dense, d, K)
    ADA, absd = getada(A, K, d, DAt)

    sym = {"L": L["L"], "perm": L["perm"], "xsuper": L["xsuper"], "tmpsiz": L.get("tmpsiz")}
    fact = _native.numeric_cholesky(sym, ADA, pars["chol"], absd)
    Lnum = {"L": fact["L"], "d": fact["d"], "skip": fact["skip"], "perm": L["perm"], "xsuper": L["xsuper"]}

    Lden, Lnum["d"] = deninfac(symLden, Lnum, dense, DAt, d, absd, K.get("qblkstart"), pars["chol"])

    psi, dx, kcg, errb = wrapPcg(
        Lnum, Lden, A, dense, d, DAt, K, (-x0) * b, v, pars["cg"], pars["eps"] / pars["cg"]["restol"]
    )
    x = np.sqrt(d["l"]) * dx

    if np.min(x) < 0.0 or (np.max(np.abs(errb)) if errb.size else 0.0) > 2 * max(
        max(y0, 1e-10 * x0) * R["maxb"], y0 * R["maxRb"]
    ):
        return None, None

    rhs = np.sqrt(d["l"]) * (x0 * c - amul(A, dense, y, transp=True))
    dy, _dx2, _k2, _errb2 = wrapPcg(
        Lnum, Lden, A, dense, d, DAt, K, np.zeros(b.size), rhs, pars["cg"], pars["eps"] / pars["cg"]["restol"]
    )
    y = y + dy

    z = x0 * c - amul(A, dense, y, transp=True)
    z = z.copy()
    z[0] = 0.0
    zB = z.copy()
    zB[lpNB] = 0.0
    normzB = float(np.max(np.abs(zB))) if zB.size else 0.0
    cx = float(c @ x)
    by = float(b @ y)
    z0 = by - cx

    if (lpNB.size and np.min(z[lpNB]) < 0.0) or normzB > 5 * max(
        1e-10 * (x0 + (x0 == 0)) * np.linalg.norm(c), min(y0, 1e-8) * np.linalg.norm(R["c"])
    ):
        return None, None

    if x0 == 0:
        if z0 <= 0:
            return None, None
    elif z0 < -5e-8 * (1 + abs(by) + float(np.max(np.abs(b)))):
        return None, None

    return x, y
