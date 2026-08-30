"""Port of iswnbr.c (the real, compiled kernel -- iswnbr.m's own body is
just a `sedumi_binary_error()` stub with dead reference code below it,
kept as documentation of the *intended* algorithm, not what actually
runs. NOT bound via ctypes: iswnbr.c's Q-set sort uses the exact same
undefined-behavior pattern already confirmed to cause real, reproducible
divergence in sortnnz.c during this port (a comparator declared to
return `signed char` but invoked through qsort() via a cast to an
`int(*)(const void*,const void*)` function pointer type) -- ported the
algorithm's clearly-stated intent directly instead (ascending sort of
the "inconclusive" set), matching the .m reference's own `sort(wQ)`
call, which has no such UB.

Also note the dead .m reference is INCOMPLETE: it only implements the
`0 < theta < 1` branch. iswnbr.c has a separate, simpler theta==1
special case (`1-thetaSQR <= 1e-8`) the .m file never shows -- and
checkpars.m defaults `pars.theta=1` for `pars.alg==0`, so this branch is
genuinely reachable, not a dead corner. Ported here from the C source
directly.
"""

from __future__ import annotations

import numpy as np


def iswnbr(vSQR, thetaSQR: float):
    """[delta,h,alpha] = iswnbr(vSQR,thetaSQR): proximity measure w.r.t.
    the wide neighborhood C(theta). Returns (1e100, 0.0, 0.0) if any
    entry of vSQR is <= 0 (matching iswnbr.c's own early return -- an
    infeasible/degenerate point)."""
    w = np.asarray(vSQR, dtype=np.float64).ravel()
    n = w.size
    gap = float(np.sum(w))
    r = n / thetaSQR

    if 1.0 - thetaSQR <= 1e-8:
        hSQR = float(np.max(w)) if n else 0.0
        h = np.sqrt(hSQR)
        sumdifw = float(np.sum(hSQR - w))
        sumdifv = float(np.sum(h - np.sqrt(w)))
    else:
        sumwNT = gap
        cardT = 0
        sumdifv = 0.0
        sumdifw = 0.0
        cardQ = n
        hSQR = sumwNT / (r - cardT)
        hubSQR = sumwNT / (r - (n - 1))
        wQ = []
        for wj in w:
            if wj >= hubSQR:  # not in T
                cardQ -= 1
                hubSQR = sumwNT / (r - cardT - cardQ)
            elif wj < hSQR:  # in T
                if wj <= 0.0:
                    return 1e100, 0.0, 0.0
                cardT += 1
                cardQ -= 1
                hubSQR *= 1 - wj / sumwNT
                sumwNT -= wj
                oldhSQR = hSQR
                hSQR = sumwNT / (r - cardT)
                sumdifw += (oldhSQR - wj) + cardT * (hSQR - oldhSQR)
                sumdifv += (np.sqrt(oldhSQR) - np.sqrt(wj)) + cardT * (
                    np.sqrt(hSQR) - np.sqrt(oldhSQR)
                )
            else:  # inconclusive
                wQ.append(wj)

        for wj in sorted(wQ):
            if wj >= hSQR:
                break
            cardT += 1
            sumwNT -= wj
            oldhSQR = hSQR
            hSQR = sumwNT / (r - cardT)
            sumdifw += (oldhSQR - wj) + cardT * (hSQR - oldhSQR)
            sumdifv += (np.sqrt(oldhSQR) - np.sqrt(wj)) + cardT * (
                np.sqrt(hSQR) - np.sqrt(oldhSQR)
            )
        h = np.sqrt(hSQR)

    alpha = sumdifv / (r * h)
    deltaSQR = alpha * (2 - alpha) - (1 - alpha) ** 2 * sumdifw / gap
    delta = np.sqrt(r * deltaSQR)
    return delta, h, alpha
