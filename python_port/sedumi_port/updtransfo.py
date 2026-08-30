"""Port of updtransfo.m: updates the Nesterov-Todd scaling point/frame
using a numerically stable method, given w = D(Xscl)*Zscl (from
trydif.m/widelen.m, not yet ported) and the previous scaling dIN.

SCOPE NOTE: real-symmetric PSD blocks only, like the qrK/urotorder/
givensrot/sqrtinv bindings this builds on (Phase 2 cluster 3's own
scope) -- a problem with complex Hermitian PSD blocks isn't supported
by this port yet.
"""

from __future__ import annotations

import numpy as np

from . import _native
from .cone import asmDxq, psdeig, triumtriu


def updtransfo(x, z, w: dict, dIN: dict, K: dict):
    """[d,vfrm] = updtransfo(x,z,w,dIN,K)"""
    x = np.asarray(x, dtype=np.float64).ravel()
    z = np.asarray(z, dtype=np.float64).ravel()
    Kl = int(K["l"])
    lorN = len(K.get("q", []))
    s_sizes = [int(v) for v in K.get("s", [])]

    wlab_full = np.asarray(w["lab"], dtype=np.float64).copy()
    if s_sizes:
        wlab, q = psdeig(w["s"], K, want_vectors=True)
        wlab_full[Kl + 2 * lorN :] = wlab
    else:
        q = np.zeros(0)

    vfrm: dict = {"lab": np.sqrt(wlab_full)}

    d: dict = {"l": dIN["l"] * (x[:Kl] / z[:Kl])}

    if lorN == 0:
        d["det"] = np.zeros(0)
        d["q1"] = np.zeros(0)
        d["q2"] = np.zeros(0)
        d["auxdet"] = np.zeros(0)
        d["auxtr"] = np.zeros(0)
        vfrm["q"] = np.zeros(0)
    else:
        ix = np.asarray(K["mainblks"], dtype=np.int64).ravel()
        i1, i2 = int(ix[0]), int(ix[1])
        nq = i2 - i1
        j3 = i2 + nq - 1
        lq = int(K["lq"])

        s = np.sqrt(np.asarray(w["tdetx"]).ravel() / np.asarray(w["tdetz"]).ravel())
        d["det"] = dIN["det"] * s
        psi1 = s * z[i1 - 1 : i2 - 1]
        psi2 = _native.qblkmul(s, z, K["qblkstart"])
        tmp = vfrm["lab"][i1 - 1 : i2 - 1] + vfrm["lab"][i2 - 1 : j3]
        chi1 = (x[i1 - 1 : i2 - 1] + psi1) / tmp
        chi2 = _native.qblkmul(1.0 / tmp, x[i2 - 1 : lq] - psi2, K["qblkstart"])
        psi1 = x[i1 - 1 : i2 - 1] - psi1
        psi2 = x[i2 - 1 : lq] + psi2

        dq = asmDxq(dIN, np.concatenate([chi1, chi2]), K)
        d["q1"] = dq[:nq]
        d["q2"] = dq[nq:]
        d["auxdet"] = np.sqrt(2 * d["det"])
        d["auxtr"] = np.sqrt(2) * (d["q1"] + d["auxdet"])

        alpha = (dIN["q1"] * psi1 + _native.ddot(dIN["q2"], psi2, K["qblkstart"])) / d["auxtr"]
        tmp2 = 2 * np.sqrt(s)
        psi1 = (psi1 - alpha * chi1) / tmp2
        psi2 = psi2 - _native.qblkmul(alpha, chi2, K["qblkstart"])
        psi2 = _native.qblkmul(1.0 / tmp2, psi2, K["qblkstart"])
        gamma = (np.sqrt(2) * psi1 + alpha) / dIN["auxtr"]

        tmp3 = vfrm["lab"][i2 - 1 : j3] - vfrm["lab"][i1 - 1 : i2 - 1]
        tmp3 = np.where(tmp3 == 0, 1.0, tmp3)  # avoid division by zero
        psi2 = psi2 + _native.qblkmul(gamma, dIN["q2"], K["qblkstart"])
        vfrm["q"] = _native.qblkmul(1.0 / tmp3, psi2, K["qblkstart"])

    # D = QUD'*QUD, where QUD(:,udIN.perm) = diag(1/sqrt(vlab))*Q'*ux*ud.
    d["u"] = triumtriu(w["ux"], dIN["u"], K)
    d["u"], d["perm"], gjc, g = _native.urotorder(d["u"], K, 1.1, perm_in=dIN.get("perm"))
    q = _native.givensrot(gjc, g, q, K)  # rotate Q accordingly
    vinv = _native.sqrtinv(q, vfrm["lab"], K)

    vfrm["s"], r = _native.qrK(vinv, K)
    d["u"] = triumtriu(r, d["u"], K)

    return d, vfrm
