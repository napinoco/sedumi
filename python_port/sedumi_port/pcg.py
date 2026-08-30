"""Port of loopPcg.m and wrapPcg.m: the preconditioned-conjugate-gradient
solver for A*P(d)*A' * y = rhs that sits at the heart of every interior-
point iteration (sddir.m/sdfactor.m both delegate to wrapPcg), plus
sparfwslv.m/sparbwslv.m (thin wrappers SeDuMi itself uses around
fwblkslv/bwblkslv -- i.e. this port's own _native.fwsolve/bwsolve).

SCOPE NOTE: the dense-column preconditioning correction (dense.q/
DAt.denq, from the Phase 2 "dense columns" subsystem still deferred --
getada1/getada2/incorder/iswnbr/symbfwblk) is NOT implemented; both
loopPcg() and wrapPcg() raise NotImplementedError if dense.q or
dense.cols is nonempty. The Lorentz-cone rank-1 scaling correction
(DAt.q, from getdatm.py) IS implemented and always applied when
K.q is nonempty -- unlike the dense-column correction, this one is
needed for every Lorentz-cone problem, not just ones with dense
columns (see getdatm.py's own module docstring).
"""

from __future__ import annotations

import numpy as np

from . import _native
from .amul import amul
from .cone import PopK, asmDxq, psdscale


def sparfwslv(L: dict, b):
    """y = sparfwslv(L,b): forward-solve y := L\\b (L a blkchol.m-style
    dict: L["L"] the unit-lower-triangular CSC factor, L["xsuper"] the
    supernode boundaries).

    _native.fwsolve() mutates its `y` argument IN PLACE (by design, it
    wraps fwblkslv.c's own in-place solve) -- but MATLAB's sparfwslv/
    fwblkslv have ordinary value semantics (the caller's own `b` is
    never touched). Copying `b` here keeps that contract for every
    caller in this port (loopPcg/wrapPcg both reuse their `r` after
    calling sparfwslv) instead of requiring each call site to remember
    to copy -- forgetting this once (silently corrupting `r` under
    `Lr = fwdpr1(Lden, sparfwslv(L, r))`, then using the now-mutated `r`
    again for `rnew = r - alpha*tmp2`) is exactly what caused this
    port's first loopPcg/wrapPcg oracle mismatch.
    """
    return _native.fwsolve(L["L"], L["xsuper"], np.array(b, dtype=np.float64, copy=True))


def sparbwslv(L: dict, b):
    """y = sparbwslv(L,b): backward-solve y := L'\\b. See sparfwslv()'s
    docstring for why `b` is copied before the in-place native call."""
    return _native.bwsolve(L["L"], L["xsuper"], np.array(b, dtype=np.float64, copy=True))


def _check_no_dense(dense: dict):
    cols = np.asarray(dense.get("cols", np.zeros(0))).ravel()
    if cols.size:
        raise NotImplementedError(
            "pcg: dense-column preconditioning (dense.cols nonempty) is not "
            "implemented -- see this module's docstring."
        )


def _DAy_from_y(At, dense, d, K, y):
    """[sqrt(d.l).*Ap(1:K.l); asmDxq(d,Ap,K); psdscale(d,Ap,K)] for Ap =
    vecsym(Amul(At,dense,y,1),K) -- the common "DA'y" tail shared by
    loopPcg's k>1 path (once or twice, for a hi/lo quadadd pair)."""
    Kl = int(K["l"])
    Ap = _native.vecsym(amul(At, dense, y, transp=True), K)
    return np.concatenate([np.sqrt(d["l"]) * Ap[:Kl], asmDxq(d, Ap, K), psdscale(d, Ap, K)])


def loopPcg(L, Lden, At, dense, d, DAt, K, b, p, ssqrNew, cgpars, restol):
    """[y,k,DAy] = loopPcg(...): solve A*P(d)*A' * y = b via PCG with L
    as the (block sparse Cholesky) preconditioner. p=None starts PCG
    from scratch; ssqrNew is only used when p is not None (continuing a
    previous step's search direction)."""
    _check_no_dense(dense)
    b = np.asarray(b, dtype=np.float64).ravel()
    Kl = int(K["l"])
    lorN = len(K.get("q", []))

    k = 0
    r = b.copy()
    finew = 0.0
    y = None
    y_lo = None  # None means y (if set) is a plain array, not a hi/lo quad-precision pair
    normrmin = float(np.max(np.abs(r))) if r.size else 0.0
    ymin = ymin_lo = None
    alpha = None
    STOP = 0
    Ap = DApq = DAps = None

    while STOP == 0:
        Lr = _native.fwdpr1(Lden, sparfwslv(L, r))
        tmp = Lr / L["d"]
        if p is None:
            ssqrNew = float(Lr @ tmp)
            p = sparbwslv(L, _native.bwdpr1(Lden, tmp))
        else:
            ssqrOld = ssqrNew
            ssqrNew = float(Lr @ tmp)
            p = (ssqrNew / ssqrOld) * p
            p = p + sparbwslv(L, _native.bwdpr1(Lden, tmp))

        Ap = _native.vecsym(amul(At, dense, p, transp=True), K)
        DDAp, DApq, DAps, ssqrDAp = PopK(d, Ap, K)
        if ssqrDAp > 0.0:
            k += 1
            alpha = ssqrNew / ssqrDAp
            if y is not None:
                if y_lo is not None:
                    y, y_lo = _native.quadadd(y, y_lo, alpha * p)
                else:
                    y = y + alpha * p
            elif cgpars.get("qprec", 1) > 0:
                y = alpha * p
                y_lo = np.zeros_like(p)
            else:
                y = alpha * p

            tmp2 = amul(At, dense, DDAp, transp=False)
            if lorN:
                tmp2 = tmp2 + DAt["q"].T @ DApq
            r = r - alpha * tmp2

            fiprev = finew
            if y_lo is not None:
                finew = float((b + r) @ y + (b + r) @ y_lo)
            else:
                finew = float((b + r) @ y)
            normr = float(np.max(np.abs(r))) if r.size else 0.0
            if normr < normrmin:
                ymin, ymin_lo = y, y_lo
                normrmin = normr
            if normr < restol:
                STOP = 1
            elif finew - fiprev < cgpars["stagtol"] * fiprev:
                STOP = 2
            elif k >= cgpars["maxiter"]:
                STOP = 2
        else:
            STOP = 1

    if STOP == 2:
        y, y_lo = ymin, ymin_lo

    if y is None:
        return None, k, None

    if k == 1:
        DAy = alpha * np.concatenate(
            [np.sqrt(d["l"]) * Ap[:Kl], asmDxq(d, Ap, K, ddotx=DApq), DAps]
        )
    else:
        DAy = _DAy_from_y(At, dense, d, K, y)
        if y_lo is not None:
            DAy = DAy + _DAy_from_y(At, dense, d, K, y_lo)

    return y, k, DAy


def wrapPcg(L, Lden, At, dense, d, DAt, K, rb, rv, cgpars, y0):
    """[y,dx,k,r] = wrapPcg(...): solve AP(d)A'*y = rb + AD(rv) (rb may
    be None/empty, matching the .m file's `if ~isempty(rb)` guard), with
    one unconditional CG-preconditioned step followed by refinement
    passes via loopPcg() as needed."""
    _check_no_dense(dense)
    Kl = int(K["l"])
    restol = y0 * cgpars["restol"]

    dx = np.concatenate([np.sqrt(d["l"]) * rv[:Kl], asmDxq(d, rv, K), psdscale(d, rv, K, transp=True)])
    r = amul(At, dense, dx, transp=False)
    if rb is not None and rb.size:
        r = r + rb

    p = _native.fwdpr1(Lden, sparfwslv(L, r))
    y = p / L["d"]
    ssqrNew = float(p @ y)
    p = sparbwslv(L, _native.bwdpr1(Lden, y))
    x = _native.vecsym(amul(At, dense, p, transp=True), K)
    dx = np.concatenate([np.sqrt(d["l"]) * x[:Kl], asmDxq(d, x, K), psdscale(d, x, K)])
    ssqrdx = float(np.sum(dx**2))
    if ssqrdx <= 0.0:
        return np.zeros_like(r), rv, 0, r

    k = 1
    alpha = ssqrNew / ssqrdx
    y = alpha * p
    dx = rv - alpha * dx
    x = np.concatenate([np.sqrt(d["l"]) * dx[:Kl], asmDxq(d, dx, K), psdscale(d, dx, K, transp=True)])
    r = amul(At, dense, x, transp=False)
    if rb is not None and rb.size:
        r = r + rb
    normr = float(np.max(np.abs(r))) if r.size else 0.0
    if normr < restol:
        return y, dx, k, r

    trial = 0
    while True:
        dy, dk, x = loopPcg(L, Lden, At, dense, d, DAt, K, r, p, ssqrNew, cgpars, restol)
        if dy is None:
            return y, dx, k, r
        k = k + dk
        y = y + dy
        dx = dx - x
        x = np.concatenate(
            [np.sqrt(d["l"]) * dx[:Kl], asmDxq(d, dx, K), psdscale(d, dx, K, transp=True)]
        )
        r = amul(At, dense, x, transp=False)
        if rb is not None and rb.size:
            r = r + rb
        normr = float(np.max(np.abs(r))) if r.size else 0.0
        if normr < restol:
            return y, dx, k, r
        if trial >= cgpars["refine"]:
            return y, dx, k, r
        p = None
        trial += 1
