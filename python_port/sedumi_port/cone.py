"""Port of SeDuMi's cone-vector-algebra .m files (eigK.m, vec.m, mat.m,
eyeK.m, maxeigK.m, mineigK.m, ...): pure NumPy, no C bindings needed --
these files are themselves plain MATLAB using the built-in eig(), not
mex functions, in the original source.

Cone K is represented as a dict, in either of the two shapes these .m
files themselves accept:
  - "external" (user-facing): keys among "f", "l", "q", "r", "s", "z",
    with "q"/"r"/"s"/"z" as sequences of block sizes.
  - "internal" (SeDuMi's own, post-pretransfo.m representation): has a
    "rsdpN" key; "q" is stacked with all Lorentz-block x0 values first,
    then the vectors, matching pretransfo.m's own reordering -- ported
    faithfully because sedumi.m's own iteration loop only ever uses this
    representation, not the external one.

All 1-indexed MATLAB slicing has been converted to 0-indexed NumPy
slicing; every reshape of a PSD block uses order="F" (column-major) to
match MATLAB's own memory layout -- getting this wrong is the single
most common way to silently corrupt a port like this one.
"""

from __future__ import annotations

import numpy as np


def vec(X):
    """x = vec(X): column-major flatten, matching MATLAB's reshape(X,
    numel(X), 1) -- NOT np.ravel()'s default row-major order."""
    return np.asarray(X).reshape(-1, order="F")


def mat(x, n=None):
    """X = mat(x, n): inverse of vec() for a square n x n matrix (n
    defaults to sqrt(len(x)))."""
    x = np.asarray(x).ravel(order="F")
    if n is None:
        n = int(np.floor(np.sqrt(x.size)))
        if n * n != x.size:
            raise ValueError("Argument x has to be a square matrix")
    return x.reshape(n, n, order="F")


def _cone_dims(K: dict):
    """Returns (is_int, nf, nl, q_sizes, r_sizes, s_sizes, nrsdp) from
    either cone-K representation, as a common starting point for
    eigK/maxeigK/mineigK/eyeK."""
    is_int = "rsdpN" in K
    if is_int:
        nf = 0
        nl = int(K.get("l", 0))
        q_sizes = [int(v) for v in K.get("q", [])]
        r_sizes = []
        s_sizes = [int(v) for v in K.get("s", [])]
        nrsdp = int(K["rsdpN"])
    else:
        nf = int(K.get("f", 0))
        nl = int(K.get("l", 0))
        q_sizes = [int(v) for v in K.get("q", [])]
        r_sizes = [int(v) for v in K.get("r", [])]
        s_sizes = [int(v) for v in K.get("s", [])]
        nrsdp = len(s_sizes)
    return is_int, nf, nl, q_sizes, r_sizes, s_sizes, nrsdp


def eigK(x, K: dict):
    """lab = eigK(x, K): spectral values of x w.r.t. the symmetric cone K."""
    x = np.asarray(x, dtype=np.float64).ravel(order="F")
    is_int, nf, nl, q_sizes, r_sizes, s_sizes, nrsdp = _cone_dims(K)
    nq, nr, ns = len(q_sizes), len(r_sizes), len(s_sizes)

    if is_int:
        N = nl + 2 * nq + sum(s_sizes)
    else:
        N = nl + 2 * nq + 2 * nr + sum(s_sizes)
        if "z" in K:
            N += sum(K["z"])

    # lab is always real: eigvalsh() of a Hermitian/symmetric matrix (the
    # PSD-block case below) always returns real eigenvalues even when
    # the matrix itself is complex.
    lab = np.zeros(N, dtype=np.float64)
    li = 0
    xi = nf
    lab[li : li + nl] = x[xi : xi + nl]
    xi += nl
    li += nl

    tmp = np.sqrt(0.5)
    if is_int:
        zi = xi
        xi += nq
        for i in range(nq):
            kk = q_sizes[i] - 1
            x0 = x[zi + i]
            nrm = np.linalg.norm(x[xi : xi + kk])
            lab[li] = tmp * (x0 - nrm)
            lab[li + 1] = tmp * (x0 + nrm)
            xi += kk
            li += 2
    else:
        for i in range(nq):
            kk = q_sizes[i]
            x0 = x[xi]
            nrm = np.linalg.norm(x[xi + 1 : xi + kk])
            lab[li] = tmp * (x0 - nrm)
            lab[li + 1] = tmp * (x0 + nrm)
            xi += kk
            li += 2

    for i in range(nr):  # external format only
        ki = r_sizes[i]
        x1, x2 = x[xi], x[xi + 1]
        rest = x[xi + 2 : xi + ki]
        nrm = np.linalg.norm(np.concatenate(([x1 - x2], 2 * rest)))
        lab[li] = 0.5 * (x1 + x2 - nrm)
        lab[li + 1] = 0.5 * (x1 + x2 + nrm)
        xi += ki
        li += 2

    for i in range(ns):
        ki = s_sizes[i]
        qi = ki * ki
        XX = x[xi : xi + qi].copy()
        xi += qi
        if i >= nrsdp:
            XX = XX + 1j * x[xi : xi + qi]
            xi += qi
        XX = XX.reshape(ki, ki, order="F")
        XX = XX + XX.conj().T
        ev = np.linalg.eigvalsh(XX)
        lab[li : li + ki] = 0.5 * ev
        li += ki
    return lab


def maxeigK(x, K: dict):
    """lab = maxeigK(x, K): largest spectral value of x w.r.t. K."""
    x = np.asarray(x, dtype=np.float64).ravel(order="F")
    is_int, nf, nl, q_sizes, r_sizes, s_sizes, nrsdp = _cone_dims(K)
    nq, nr, ns = len(q_sizes), len(r_sizes), len(s_sizes)

    xi = nf
    lab = np.max(np.concatenate(([-np.inf], x[xi : xi + nl]))) if nl else -np.inf
    xi += nl

    tmp = np.sqrt(0.5)
    if is_int:
        zi = xi
        xi += nq
        for i in range(nq):
            kk = q_sizes[i] - 1
            x0 = x[zi + i]
            lab = max(lab, tmp * (x0 + np.linalg.norm(x[xi : xi + kk])))
            xi += kk
    else:
        for i in range(nq):
            kk = q_sizes[i]
            x0 = x[xi]
            lab = max(lab, tmp * (x0 + np.linalg.norm(x[xi + 1 : xi + kk])))
            xi += kk

    for i in range(nr):
        ki = r_sizes[i]
        x1, x2 = x[xi], x[xi + 1]
        rest = x[xi + 2 : xi + ki]
        lab = max(lab, 0.5 * (x1 + x2 + np.linalg.norm(np.concatenate(([x1 - x2], 2 * rest)))))
        xi += ki

    for i in range(ns):
        ki = s_sizes[i]
        qi = ki * ki
        XX = x[xi : xi + qi].copy()
        xi += qi
        if i >= nrsdp:
            XX = XX + 1j * x[xi : xi + qi]
            xi += qi
        XX = XX.reshape(ki, ki, order="F")
        XX = XX + XX.conj().T
        val = np.max(np.linalg.eigvalsh(XX))
        lab = max(lab, 0.5 * val)
    return lab


def mineigK(x, K: dict):
    """lab = mineigK(x, K): smallest spectral value of x w.r.t. K."""
    x = np.asarray(x, dtype=np.float64).ravel(order="F")
    is_int, nf, nl, q_sizes, r_sizes, s_sizes, nrsdp = _cone_dims(K)
    nq, nr, ns = len(q_sizes), len(r_sizes), len(s_sizes)

    xi = nf
    if nl > 0:
        lab = np.min(x[xi : xi + nl])
        xi += nl
    else:
        lab = np.inf

    if nq:
        # NOTE: unlike eigK/maxeigK/eyeK, mineigK.m has NO is_int branch
        # for the Lorentz part -- it always uses the "external", x0-
        # immediately-followed-by-vector layout, even when called with
        # K.rsdpN present. This was confirmed against the real Octave
        # build: adding an is_int branch here (the "obvious" port,
        # mirroring eigK's own handling) gave a DIFFERENT, wrong answer.
        scl = np.sqrt(0.5)
        for k in range(nq):
            kk = q_sizes[k]
            lab = min(lab, scl * (x[xi] - np.linalg.norm(x[xi + 1 : xi + kk])))
            xi += kk

    for k in range(nr):
        kk = r_sizes[k]
        x1, x2 = x[xi], x[xi + 1]
        rest = x[xi + 2 : xi + kk]
        lab = min(lab, 0.5 * (x1 + x2 - np.linalg.norm(np.concatenate(([x1 - x2], 2 * rest)))))
        xi += kk

    for i in range(ns):
        ki = s_sizes[i]
        qi = ki * ki
        XX = x[xi : xi + qi].copy()
        xi += qi
        if i >= nrsdp:
            XX = XX + 1j * x[xi : xi + qi]
            xi += qi
        XX = XX.reshape(ki, ki, order="F")
        XX = XX + XX.conj().T
        lab = min(lab, 0.5 * np.min(np.linalg.eigvalsh(XX)))
    return lab


def eyeK(K: dict):
    """x = eyeK(K): the cone K's identity element."""
    is_int = "rsdpN" in K
    if is_int:
        N = int(K["N"])
    else:
        N = 0
        N += int(K.get("f", 0))
        N += int(K.get("l", 0))
        N += int(sum(K.get("q", [])))
        N += int(sum(K.get("r", [])))
        N += int(sum(int(v) * int(v) for v in K.get("s", [])))
        N += int(sum(int(v) * int(v) for v in K.get("z", [])))

    x = np.zeros(N, dtype=np.float64)
    xi = 0
    if not is_int and "f" in K:
        xi += K["f"]
    if "l" in K:
        x[xi : xi + K["l"]] = 1.0
        xi += K["l"]
    q_sizes = [int(v) for v in K.get("q", [])]
    if q_sizes:
        if is_int:
            x[xi : xi + len(q_sizes)] = np.sqrt(2.0)
        else:
            tmp = np.array(q_sizes[:-1])
            offsets = int(K.get("f", 0)) + int(K.get("k", 0)) + np.concatenate(([1], tmp)).cumsum() - 1
            x[offsets.astype(np.int64)] = np.sqrt(2.0)
        xi += sum(q_sizes)
    r_sizes = [int(v) for v in K.get("r", [])] if not is_int else []
    if r_sizes:
        tmp = np.array(r_sizes[:-1])
        starts = (np.concatenate(([1], tmp)).cumsum() - 1).astype(np.int64)
        x[starts] = 1.0
        x[starts + 1] = 1.0
        xi += sum(r_sizes)
    s_sizes = [int(v) for v in K.get("s", [])]
    if s_sizes:
        nc = len(s_sizes)
        nr_ = int(K["rsdpN"]) if is_int else nc
        for i in range(nc):
            ki = s_sizes[i]
            qi = ki * ki
            x[xi : xi + qi : ki + 1] = 1.0
            xi += (1 + (1 if i >= nr_ else 0)) * qi
    if not is_int and K.get("z"):
        for ki in K["z"]:
            ki = int(ki)
            qi = ki * ki
            x[xi : xi + qi : ki + 1] = 1.0
            xi += qi
    return x
