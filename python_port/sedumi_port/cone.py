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

from . import _native


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


def psdeig(x, K: dict, want_vectors: bool = False):
    """[lab,q] = psdeig(x,K): spectral coefficients (and, optionally, the
    eigenbasis q) of the PSD part of x -- like eigK's own PSD-block
    handling, but PSD-only, and can also return the eigenvectors. `q`'s
    eigenvector signs/phases are not unique (LAPACK's zheev/dsyev may
    pick different ones than MATLAB's eig for degenerate or nearly-
    degenerate eigenvalues), so it is not compared element-wise against
    the Octave oracle -- only its defining property (X = q*diag(2*lab)*q')
    is checked.

    Only the last N = sum(Ks^2) + sum(Ks[rsdpN:]^2) entries of x are used
    (xi = len(x) - N), matching psdeig.m's own indexing -- x may be a
    bare PSD-only vector or a full L+Q+S vector.
    """
    x = np.asarray(x, dtype=np.float64).ravel(order="F")
    s_sizes = [int(v) for v in K.get("s", [])]
    if not s_sizes:
        return (np.zeros(0), np.zeros(0)) if want_vectors else np.zeros(0)
    nr = int(K["rsdpN"])
    nc = len(s_sizes)
    Kq = [ki * ki for ki in s_sizes]
    N = sum(Kq) + sum(Kq[nr:])
    xi = x.size - N
    lab = np.zeros(sum(s_sizes), dtype=np.float64)
    ei = 0
    if want_vectors:
        q = np.zeros(N, dtype=np.float64)
        vi = 0
    for i in range(nc):
        ki = s_sizes[i]
        qi = Kq[i]
        XX = x[xi : xi + qi].copy()
        xi += qi
        if i >= nr:
            XX = XX + 1j * x[xi : xi + qi]
            xi += qi
        XX = XX.reshape(ki, ki, order="F")
        XX = XX + XX.conj().T
        if want_vectors:
            DD, QQ = np.linalg.eigh(XX)
        else:
            DD = np.linalg.eigvalsh(XX)
        lab[ei : ei + ki] = 0.5 * DD
        ei += ki
        if want_vectors:
            q[vi : vi + qi] = np.real(QQ).reshape(-1, order="F")
            vi += qi
            if i >= nr:
                q[vi : vi + qi] = np.imag(QQ).reshape(-1, order="F")
                vi += qi
    return (lab, q) if want_vectors else lab


def psdfactor(x, K: dict):
    """[ux,ispos] = psdfactor(x,K): per-PSD-block lower Cholesky factor
    UX'*UX = X, mirrored into a full (non-triangular-looking) array via
    UX + tril(UX,-1)' -- callers (psdscale/psdinvscale) re-extract the
    triangular part they need themselves, so the mirrored upper part is
    dead weight kept only for byte-for-byte fidelity to psdfactor.m.
    Returns ispos=False (and a partial/garbage ux) the moment any block
    fails to be positive definite, exactly like the real chol()-based
    early return.
    """
    x = np.asarray(x, dtype=np.float64).ravel(order="F")
    s_sizes = [int(v) for v in K.get("s", [])]
    if not s_sizes:
        return np.zeros(0), True
    nr = int(K["rsdpN"])
    nc = len(s_sizes)
    Kq = [ki * ki for ki in s_sizes]
    N = sum(Kq) + sum(Kq[nr:])
    ux = np.zeros(N, dtype=np.float64)
    xi = x.size - N
    ui = 0
    for i in range(nc):
        ki = s_sizes[i]
        qi = Kq[i]
        XX = x[xi : xi + qi].copy()
        xi += qi
        if i >= nr:
            XX = XX + 1j * x[xi : xi + qi]
            xi += qi
        XX = XX.reshape(ki, ki, order="F")
        try:
            L = np.linalg.cholesky(XX)
        except np.linalg.LinAlgError:
            return ux, False
        L = L + np.tril(L, -1).conj().T
        ux[ui : ui + qi] = np.real(L).reshape(-1, order="F")
        ui += qi
        if i >= nr:
            ux[ui : ui + qi] = np.imag(L).reshape(-1, order="F")
            ui += qi
    return ux, True


def _solve_ud_sandwich(TT, XX):
    """inv(TT) @ XX @ inv(TT'), computed via two triangular-agnostic
    solves (as MATLAB's `TT \\ (XX / TT')` does) rather than explicit
    inverses."""
    A = np.linalg.solve(TT, XX)
    return np.linalg.solve(TT.conj(), A.T).T


def psdinvscale(ud, x, K: dict):
    """y = psdinvscale(ud,x,K): y = D(d^-1) x = Ud' \\ X / Ud per PSD
    block, with Ud = triu(reshape(ud[block])) (upper-triangular part
    only, matching psdinvscale.m)."""
    ud = np.asarray(ud, dtype=np.float64).ravel(order="F")
    x = np.asarray(x, dtype=np.float64).ravel(order="F")
    s_sizes = [int(v) for v in K.get("s", [])]
    if not s_sizes:
        return np.zeros(0)
    nr = int(K["rsdpN"])
    nc = len(s_sizes)
    Kq = [ki * ki for ki in s_sizes]
    N = sum(Kq) + sum(Kq[nr:])
    y = np.zeros(N, dtype=np.float64)
    xi = x.size - N
    yi = 0
    ui = 0
    for i in range(nc):
        ki = s_sizes[i]
        qi = Kq[i]
        TT = ud[ui : ui + qi].copy()
        ui += qi
        if i >= nr:
            TT = TT + 1j * ud[ui : ui + qi]
            ui += qi
        TT = np.triu(TT.reshape(ki, ki, order="F"))
        XX = x[xi : xi + qi].copy()
        xi += qi
        if i >= nr:
            XX = XX + 1j * x[xi : xi + qi]
            xi += qi
        XX = XX.reshape(ki, ki, order="F")
        XX = _solve_ud_sandwich(TT, XX)
        y[yi : yi + qi] = np.real(XX).reshape(-1, order="F")
        yi += qi
        if i >= nr:
            imag = np.imag(XX)
            np.fill_diagonal(imag, 0.0)  # needed, otherwise psdfactor() will sometimes fail
            y[yi : yi + qi] = imag.reshape(-1, order="F")
            yi += qi
    return y


def psdjmul(x, y, K: dict):
    """z = psdjmul(x,y,K): (XY+YX)/2 per PSD block. Uses K['N'] and
    K['sblkstart'][0] (not sum(Ks^2) like psdeig/psdfactor/psdscale) to
    size its output, matching psdjmul.m exactly."""
    x = np.asarray(x, dtype=np.float64).ravel(order="F")
    y = np.asarray(y, dtype=np.float64).ravel(order="F")
    s_sizes = [int(v) for v in K.get("s", [])]
    if not s_sizes:
        return np.zeros(0)
    nr = int(K["rsdpN"])
    nc = len(s_sizes)
    Kq = [ki * ki for ki in s_sizes]
    N = int(K["N"]) - int(np.asarray(K["sblkstart"]).ravel()[0]) + 1
    z = np.zeros(N, dtype=np.float64)
    xi = x.size - N
    zi = 0
    for i in range(nc):
        ki = s_sizes[i]
        qi = Kq[i]
        XX = x[xi : xi + qi].copy()
        YY = y[xi : xi + qi].copy()
        xi += qi
        if i >= nr:
            XX = XX + 1j * x[xi : xi + qi]
            YY = YY + 1j * y[xi : xi + qi]
            xi += qi
        XX = XX.reshape(ki, ki, order="F")
        YY = YY.reshape(ki, ki, order="F")
        ZZ = XX @ YY
        ZZ = 0.5 * (ZZ + ZZ.conj().T)
        z[zi : zi + qi] = np.real(ZZ).reshape(-1, order="F")
        zi += qi
        if i >= nr:
            z[zi : zi + qi] = np.imag(ZZ).reshape(-1, order="F")
            zi += qi
    return z


def triumtriu(x, y, K: dict):
    """z = triumtriu(x,y,K): z = x*y for upper-triangular x,y (per PSD
    block), Hermitianized via the strict-upper mirror -- since the
    product of two upper-triangular matrices is itself upper-triangular,
    this just fills in the (all-zero) strict lower part."""
    x = np.asarray(x, dtype=np.float64).ravel(order="F")
    y = np.asarray(y, dtype=np.float64).ravel(order="F")
    s_sizes = [int(v) for v in K.get("s", [])]
    if not s_sizes:
        return np.zeros(0)
    nr = int(K["rsdpN"])
    nc = len(s_sizes)
    Kq = [ki * ki for ki in s_sizes]
    N = int(K["N"]) - int(np.asarray(K["sblkstart"]).ravel()[0]) + 1
    z = np.zeros(N, dtype=np.float64)
    xi = x.size - N
    zi = 0
    for i in range(nc):
        ki = s_sizes[i]
        qi = Kq[i]
        XX = x[xi : xi + qi].copy()
        YY = y[xi : xi + qi].copy()
        xi += qi
        if i >= nr:
            XX = XX + 1j * x[xi : xi + qi]
            YY = YY + 1j * y[xi : xi + qi]
            xi += qi
        XX = np.triu(XX.reshape(ki, ki, order="F"))
        YY = np.triu(YY.reshape(ki, ki, order="F"))
        ZZ = XX @ YY
        ZZ = ZZ + np.triu(ZZ, 1).conj().T
        z[zi : zi + qi] = np.real(ZZ).reshape(-1, order="F")
        zi += qi
        if i >= nr:
            z[zi : zi + qi] = np.imag(ZZ).reshape(-1, order="F")
            zi += qi
    return z


def psdscale(ud, x, K: dict, transp: bool = False):
    """y = psdscale(ud,x,K,transp): y[k] = vec(Ldk' Xk Ldk) (transp=False)
    or vec(Udk' Xk Udk) (transp=True), with Ld=tril(reshape(ud[block])),
    Ud=triu(reshape(ud[block])). `ud` may be a plain vector, or a dict
    {"u": ..., "perm": ...} to apply a per-block pivot ordering (perm's
    entries are LOCAL 1-indexed positions within each block, ki at a
    time -- not global indices into the sum(K.s)-length array)."""
    x = np.asarray(x, dtype=np.float64).ravel(order="F")
    s_sizes = [int(v) for v in K.get("s", [])]
    if not s_sizes:
        return np.zeros(0)
    nr = int(K["rsdpN"])
    nc = len(s_sizes)
    Kq = [ki * ki for ki in s_sizes]
    N = sum(Kq) + sum(Kq[nr:])
    y = np.zeros(N, dtype=np.float64)
    xi = x.size - N
    yi = 0
    ui = 0

    if isinstance(ud, dict):
        perm = np.asarray(ud.get("perm", [])).ravel()
        if perm.size == 0:
            prep = postp = False
        else:
            prep = not transp
            postp = bool(transp)
            pi = 0
        ud_vec = np.asarray(ud["u"], dtype=np.float64).ravel(order="F")
    else:
        prep = postp = False
        ud_vec = np.asarray(ud, dtype=np.float64).ravel(order="F")

    for i in range(nc):
        ki = s_sizes[i]
        qi = Kq[i]
        TT = ud_vec[ui : ui + qi].copy()
        ui += qi
        if i >= nr:
            TT = TT + 1j * ud_vec[ui : ui + qi]
            ui += qi
        TT = TT.reshape(ki, ki, order="F")
        TT = np.triu(TT) if transp else np.tril(TT)
        XX = x[xi : xi + qi].copy()
        xi += qi
        if i >= nr:
            XX = XX + 1j * x[xi : xi + qi]
            xi += qi
        XX = XX.reshape(ki, ki, order="F")
        if prep:
            PP = perm[pi : pi + ki].astype(np.int64) - 1
            pi += ki
            if np.any(np.diff(PP) != 1):
                XX = XX[np.ix_(PP, PP)]
        XX = TT.conj().T @ XX @ TT
        if postp:
            PP = perm[pi : pi + ki].astype(np.int64) - 1
            pi += ki
            if np.any(np.diff(PP) != 1):
                out = np.zeros_like(XX)
                out[np.ix_(PP, PP)] = XX
                XX = out
        y[yi : yi + qi] = np.real(XX).reshape(-1, order="F")
        yi += qi
        if i >= nr:
            imag = np.imag(XX)
            np.fill_diagonal(imag, 0.0)  # needed, otherwise psdfactor() will sometimes fail
            y[yi : yi + qi] = imag.reshape(-1, order="F")
            yi += qi
    return y


def qframeit(lab, frmq, K: dict):
    """x = qframeit(lab,frmq,K): reconstructs the Lorentz "vector" part
    of x from spectral values lab and frame direction frmq.

    IMPORTANT: `lab` here uses a *grouped* layout ([.., lo_1..lo_lorN,
    hi_1..hi_lorN, ..]), NOT eigK()'s own *interleaved* per-block layout
    ([.., lo_1,hi_1,lo_2,hi_2, ..]) -- confirmed against the real Octave
    build (see generate_cone2_oracle.m's docstring). This grouped layout
    is what trydif.m/widelen.m actually build (`w.lab = [...; detxz./
    lab2q; lab2q; ...]`) and is what updtransfo.m feeds into vfrm.lab,
    which is what sedumi.m's own iteration loop passes to frameit/
    qframeit -- so this is the layout to match, not eigK's.
    """
    lab = np.asarray(lab, dtype=np.float64).ravel(order="F")
    frmq = np.asarray(frmq, dtype=np.float64).ravel(order="F")
    lorN = len(K.get("q", []))
    if lab.size > 2 * lorN:
        l = int(K.get("l", 0))
        lab = lab[l : l + 2 * lorN]
    lo = lab[:lorN]
    hi = lab[lorN:]
    x_lo = (lo + hi) / np.sqrt(2.0)
    x_vec = _native.qblkmul(hi - lo, frmq, K["qblkstart"])
    return np.concatenate([x_lo, x_vec])


def qjmul(x, y, K: dict):
    """z = qjmul(x,y,K): Jordan product (x*y)/sqrt(2) for Lorentz cones.
    x,y are full internal-format vectors (K.mainblks-indexed), unless
    shorter than K['lq'], in which case they're treated as starting
    right at the Lorentz part (mainblks shifted so ix[0] becomes 1)."""
    q_sizes = K.get("q", [])
    if len(q_sizes) == 0:
        return np.zeros(0)
    x = np.asarray(x, dtype=np.float64).ravel(order="F")
    y = np.asarray(y, dtype=np.float64).ravel(order="F")
    if x.size != y.size:
        raise ValueError("x,y size mismatch")
    ix = np.asarray(K["mainblks"], dtype=np.int64).ravel().copy()
    if x.size < int(K["lq"]):
        ix = ix + (1 - ix[0])
    i1, i2, i3 = int(ix[0]), int(ix[1]), int(ix[2])
    z1 = x[i1 - 1 : i2 - 1] * y[i1 - 1 : i2 - 1] + _native.ddot(
        x[i2 - 1 : i3 - 1], y, K["qblkstart"]
    )
    z_rest = _native.qblkmul(x[i1 - 1 : i2 - 1], y, K["qblkstart"]) + _native.qblkmul(
        y[i1 - 1 : i2 - 1], x, K["qblkstart"]
    )
    return np.concatenate([z1, z_rest]) / np.sqrt(2.0)


def qinvjmul(labx, frmx, b, K: dict):
    """y = qinvjmul(labx,frmx,b,K): inverse Jordan multiply for Lorentz
    blocks. `labx` uses the same grouped layout as qframeit's `lab` (see
    its docstring), not eigK()'s interleaved one."""
    lorN = len(K.get("q", []))
    if lorN == 0:
        return np.zeros(0)
    labx = np.asarray(labx, dtype=np.float64).ravel(order="F")
    b = np.asarray(b, dtype=np.float64).ravel(order="F")
    if labx.size > 2 * lorN:
        l = int(K.get("l", 0))
        labx = labx[l : l + 2 * lorN]
    detx = labx[:lorN] * labx[lorN:]
    x = qframeit(labx, frmx, K)
    ix = np.asarray(K["mainblks"], dtype=np.int64).ravel().copy()
    if b.size == ix[2] - ix[0]:  # Lorentz only?
        ix = ix + (1 - ix[0])
    i1, i2 = int(ix[0]), int(ix[1])
    y1 = x[:lorN] * b[i1 - 1 : i2 - 1] - _native.ddot(x[lorN:], b, K["qblkstart"])
    y1 = y1 / (np.sqrt(2.0) * detx)
    y_rest = _native.qblkmul(np.sqrt(2.0) / x[:lorN], b, K["qblkstart"]) - _native.qblkmul(
        y1 / x[:lorN], x[lorN:], K["qblkstart"]
    )
    return np.concatenate([y1, y_rest])


def frameit(lab, frmq, frms, K: dict):
    """x = frameit(lab,frmq,frms,K): x = [lab(L-part); qframeit(...);
    psdframeit(...)] -- a pure concatenation, `lab` in the same grouped
    layout qframeit/qinvjmul use (see qframeit's docstring).

    Unlike psdframeit.c's own mexFunction (which accepts either a
    PSD-only lab or a full lab and slices off the L+Lorentz prefix
    itself), _native.psdframeit()'s Python wrapper expects an exact
    PSD-only lab -- so that slicing is done here instead, matching
    frameit.m's actual full-lab call.
    """
    lab = np.asarray(lab, dtype=np.float64).ravel(order="F")
    l = int(K.get("l", 0))
    psd_len = sum(int(v) for v in K.get("s", []))
    lab_psd = lab if lab.size == psd_len else lab[lab.size - psd_len :]
    return np.concatenate(
        [lab[:l], qframeit(lab, frmq, K), _native.psdframeit(lab_psd, frms, K)]
    )
