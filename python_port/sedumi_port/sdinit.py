"""Port of sdinit.m: builds the trivial identity solution on the central
path (y=0, v=mu*identity) for SeDuMi's self-dual model, the scaling
point d, its Jordan frame vfrm, and the initial residuals R that the
rest of the predictor-corrector iteration (sddir/wregion/etc., Phase
3-c) is built around."""

from __future__ import annotations

import numpy as np

from . import _native
from .amul import amul
from .cone import eyeK


def sdinit(At, b, c, dense: dict, K: dict, pars: dict):
    """[d,v,vfrm,y,y0,R] = sdinit(At,b,c,dense,K,pars)"""
    b = np.asarray(b, dtype=np.float64).ravel()
    c = np.asarray(c, dtype=np.float64).ravel()
    m = b.size
    lorN = len(K.get("q", []))
    n = int(K["l"]) + 2 * lorN + int(K["rLen"]) + int(K["hLen"])

    R: dict = {}
    R["maxb"] = float(np.max(np.abs(b))) if b.size else 0.0
    R["maxc"] = float(np.max(np.abs(c))) if c.size else 0.0

    y = np.zeros(m, dtype=np.float64)
    mu = pars["mu"] * np.sqrt((1 + R["maxb"]) * (1 + R["maxc"]))
    id_ = eyeK(K)
    v = mu * id_
    y0 = n * mu
    R["b0"] = mu

    d0 = np.sqrt((1 + R["maxb"]) / (1 + R["maxc"]))
    x0 = pars["mu"]
    z0 = mu**2 / x0
    cx = d0 * float(c @ v)
    R["sd"] = (z0 + cx) / y0

    Kl = int(K["l"])
    d: dict = {}
    d["l"] = d0**2 * np.ones(Kl, dtype=np.float64)
    if Kl:
        d["l"][0] = x0 / z0
    d["det"] = d0**2 * np.ones(lorN, dtype=np.float64)
    d["q1"] = (np.sqrt(2) * d0) * np.ones(lorN, dtype=np.float64)
    mainblks = np.asarray(K["mainblks"]).ravel()
    d["q2"] = np.zeros(int(mainblks[2] - mainblks[1]), dtype=np.float64)
    d["auxdet"] = np.sqrt(2 * d["det"])
    d["auxtr"] = np.sqrt(2) * (d["q1"] + d["auxdet"])
    lq = int(K["lq"])
    d["u"] = np.sqrt(d0) * id_[lq:]
    d["perm"] = np.zeros(0, dtype=np.float64)

    vfrm: dict = {}
    vfrm["lab"] = mu * np.ones(n, dtype=np.float64)
    vfrm["q"] = d["q2"]
    vfrm["s"], _r = _native.qrK(d["u"], K)

    R["b"] = d0 * amul(At, dense, v, transp=False)
    R["b"] = (R["b"] - x0 * b) / y0
    R["c"] = _native.vecsym(v / d0 - x0 * c, K) / y0
    R["c"][0] = 0.0  # for artificial (x0,z0)

    R["maxRb"] = max(1e-6, float(np.max(np.abs(R["b"]))) if R["b"].size else 0.0)
    R["maxRc"] = max(1e-6, float(np.max(np.abs(R["c"]))) if R["c"].size else 0.0)
    R["norm"] = max(R["maxRb"], R["maxRc"], R["sd"])
    w = np.asarray(pars["w"], dtype=np.float64).ravel()
    R["w"] = 2 * w * np.array([R["maxRb"], R["maxRc"]]) / np.array([1 + R["maxb"], 1 + R["maxc"]])

    return d, v, vfrm, y, y0, R
