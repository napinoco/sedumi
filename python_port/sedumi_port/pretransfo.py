"""Port of pretransfo.m: converts a user-facing (external) (At,b,c,K) SDP/
SOCP problem into SeDuMi's internal format (real-only, Lorentz cones
rearranged into trace+norm-bound blocks, rotated Lorentz cones converted
to standard ones, diagonal SDP blocks folded into the LP block, complex
SDP coefficients moved to the lower triangle, and a self-dual (x0,z0)
variable appended).

This is a line-by-line port of a dense, index-arithmetic-heavy MATLAB
file built around constructing one big sparse transformation matrix QR
(via (row,col,value) triplet lists, one per cone-transformation kind)
and applying it to At and c. See the .m file's own comments for the
mathematical meaning of each transformation; this docstring only calls
out where the Python port diverges from a literal translation.

KNOWN UPSTREAM BUG NOT REPLICATED: pretransfo.m's own line
`scplx(K.scomplex&~sdiag) = true;` uses `&` (elementwise AND) between
K.scomplex (a list of 1-indexed complex-SDP-block numbers) and ~sdiag (a
length-L_s logical mask) as if it were an index-set assignment. This
only happens to work when K.scomplex is empty, has exactly one element
(scalar broadcasts against any vector), or has exactly L_s elements
(sizes already match) -- confirmed by direct reproduction against the
real Octave build: `K.s=[2;2;2]; K.scomplex=[1,3]` crashes with
"mx_el_and: nonconformant arguments (op1 is 1x2, op2 is 1x3)" inside
pretransfo.m itself. This is a plain bug (not a deliberate or relied-
upon behavior -- nothing downstream depends on the crash), so unlike
mineigK's/eyeK's bug-compatible quirks this is NOT replicated: this port
uses the evidently-intended logic (`scplx[K.scomplex-1] = True`, an
actual index-set assignment) so that declaring 2+ but not all SDP blocks
complex actually works, matching what K.rsdpN/K.cdim's own downstream
formulas (`nnz(sreal)`, `sum(K.s(scplx).^2)`) clearly assume `scplx`
means: a boolean mask selecting the blocks in K.scomplex.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp


def _norm_block_sizes(K: dict, name: str, minval: int) -> np.ndarray:
    v = K.get(name)
    if v is None:
        return np.zeros(0, dtype=np.int64)
    arr = np.atleast_1d(np.asarray(v, dtype=np.float64)).ravel()
    if arr.size == 0 or not np.any(arr):
        return np.zeros(0, dtype=np.int64)
    if np.any(arr != np.floor(arr)) or np.any(arr < minval) or np.any(~np.isreal(arr)):
        raise ValueError(f"K.{name} should contain only valid block sizes (>= {minval})")
    return arr.astype(np.int64)


def _norm_index_set(K: dict, name: str, upper_bound: int) -> np.ndarray:
    v = K.get(name)
    if v is None:
        return np.zeros(0, dtype=np.int64)
    arr = np.atleast_1d(np.asarray(v, dtype=np.float64)).ravel()
    if arr.size == 0:
        return np.zeros(0, dtype=np.int64)
    arr = np.unique(arr)
    if np.any(arr != np.floor(arr)) or np.any(arr < 1):
        raise ValueError(f"K.{name} should contain only positive integers")
    if np.any(arr > upper_bound):
        raise ValueError(f"Elements of K.{name} are out of range")
    return arr.astype(np.int64)


def _to_csc_complex(A):
    A = sp.csc_matrix(A, dtype=np.complex128) if not sp.issparse(A) else A.tocsc().astype(np.complex128)
    return A


def pretransfo(At, b, c, K: dict, pars: dict):
    """[At,b,c,K,prep,origcoeff] = pretransfo(At,b,c,K,pars)"""
    K = dict(K)
    pars = dict(pars)
    prep: dict = {}

    # ---- normalize K.f, K.l, K.q, K.r, K.s, K.z ----
    Kf = int(K.get("f", 0) or 0)
    Kl = int(K.get("l", 0) or 0)
    if Kf < 0:
        raise ValueError("K.f should be nonnegative integer")
    if Kl < 0:
        raise ValueError("K.l should be nonnegative integer")
    Kq = _norm_block_sizes(K, "q", 2)
    Kr = _norm_block_sizes(K, "r", 3)
    Ks = _norm_block_sizes(K, "s", 1)
    Kz = _norm_block_sizes(K, "z", 1)

    N_f, N_l = Kf, Kl
    N_fl = N_f + N_l
    L_q = Kq.size
    N_q = int(Kq.sum())
    L_r = Kr.size
    N_r = int(Kr.sum())
    N_qr = N_q + N_r
    L_qr = L_q + L_r
    L_s = Ks.size
    L_z = Kz.size
    N_s = int((Ks.astype(np.int64) ** 2).sum())
    N_z = int((Kz.astype(np.int64) ** 2).sum())
    L_qrsz = L_qr + L_s + L_z
    N_flqr = N_fl + N_qr
    N = N_flqr + N_s + N_z

    K_ycomplex = _norm_index_set(K, "ycomplex", b.size)
    K_xcomplex = _norm_index_set(K, "xcomplex", N_flqr)
    K_scomplex = _norm_index_set(K, "scomplex", L_s)

    if L_z:
        Ks = np.concatenate([Ks, Kz])
        K_scomplex = np.concatenate([K_scomplex, np.arange(L_s + 1, L_s + L_z + 1)])
        L_s = L_s + L_z
        N_s = N_s + N_z
        L_z = 0
        N_z = 0

    # ---- verify/normalize At, b, c shapes ----
    if sp.issparse(At):
        At = At.tocsc()
    else:
        At = sp.csc_matrix(np.asarray(At))
    if np.iscomplexobj(At.data) or True:
        At = At.astype(np.complex128)
    if At.shape[0] == N:
        pass
    elif At.shape[1] == N:
        At = At.T.tocsc()
    else:
        raise ValueError("(At,K) size mismatch")

    b = np.asarray(b, dtype=np.complex128).ravel()
    if b.size != At.shape[1]:
        raise ValueError("(At,b) size mismatch")
    c = np.asarray(c, dtype=np.complex128).ravel()
    if c.size != N:
        raise ValueError("(c,K) size mismatch")

    if pars.get("errors", 0) == 1:
        origcoeff = {"At": At.copy(), "c": c.copy(), "b": b.copy(), "K": dict(K)}
    else:
        origcoeff = None

    # ---- flag diagonal SDP blocks for removal ----
    if L_s and pars.get("sdp", 1):
        ssiz = (Ks.astype(np.int64)) ** 2
        strt = np.concatenate([[1], ssiz[:-1]]).cumsum()  # 1-indexed block starts within the SDP part
        sblk = np.zeros(N_s, dtype=np.int64)
        sblk[strt - 1] = 1
        sblk = sblk.cumsum()  # sblk[i] (0-indexed row i) = 1-indexed block number

        c_part = c[N_flqr:N]
        At_part = At[N_flqr:N, :]
        At_row_abs_max = (
            np.asarray(abs(At_part).max(axis=1).todense()).ravel() if At_part.shape[0] else np.zeros(0)
        )
        nnz_row = (np.abs(c_part) != 0) | (At_row_abs_max != 0)
        spattern = np.nonzero(nnz_row)[0] + 1  # 1-indexed positions within the SDP part
        sblk_sel = sblk[spattern - 1]
        rem = (spattern - strt[sblk_sel - 1]) % (Ks[sblk_sel - 1] + 1)
        sblk_sel = sblk_sel[rem != 0]
        sdiag = np.ones(L_s, dtype=bool)
        sdiag[sblk_sel - 1] = False
    else:
        sdiag = Ks == 1

    # ---- split K.ycomplex into pairs of real constraints ----
    if K_ycomplex.size:
        b_new = np.concatenate([np.real(b), np.imag(b[K_ycomplex - 1])])
        At = sp.hstack([At, 1j * At[:, K_ycomplex - 1]]).tocsc()
        b = b_new
    else:
        b = np.real(b)

    # ---- locate complex data (replaces whichcpx.c) ----
    if K_xcomplex.size == 0 and K_scomplex.size == 0:
        K_fcplx = np.zeros(0, dtype=np.int64)
        K_qcplx = np.zeros(0, dtype=np.int64)
        K_rcplx = np.zeros(0, dtype=np.int64)
        scplx = np.zeros(L_s, dtype=bool)
        sreal = ~sdiag
        K_rsdpN = L_s
        N_fc = 0
        K_cdim = 0
    else:
        xc = K_xcomplex.copy()
        tt = xc <= N_fl
        K_fcplx = xc[tt]
        xc = xc[~tt] - N_fl
        tt = xc <= N_q
        K_qcplx = xc[tt]
        xc = xc[~tt] - N_q
        tt = xc <= N_r
        K_rcplx = xc[tt]
        if K_qcplx.size:
            ndxs = np.concatenate([[1], Kq[:-1]]).cumsum()
            t2 = np.isin(K_qcplx, ndxs)
            K_fcplx = np.concatenate([K_fcplx, K_qcplx[t2] + N_fl])
            K_qcplx = K_qcplx[~t2]
            blk_of = (K_qcplx[:, None] > ndxs[None, :]).sum(axis=1)
            Kq = Kq + np.bincount(blk_of - 1, minlength=L_q).astype(np.int64)
            K_qcplx = K_qcplx + np.arange(1, K_qcplx.size + 1)
            N_q = N_q + K_qcplx.size
        if K_rcplx.size:
            ndxr = np.concatenate([[1], Kr[:-1]]).cumsum()
            pair_starts = np.concatenate([ndxr, ndxr + 1])
            t2 = np.isin(K_rcplx, pair_starts)
            K_fcplx = np.concatenate([K_fcplx, K_rcplx[t2] + N_fl + N_q])
            K_rcplx = K_rcplx[~t2]
            blk_of = (K_rcplx[:, None] > ndxr[None, :]).sum(axis=1)
            Kr = Kr + np.bincount(blk_of - 1, minlength=L_r).astype(np.int64)
            K_rcplx = K_rcplx + np.arange(1, K_rcplx.size + 1) + 2 * blk_of
            N_r = N_r + K_rcplx.size
        N_fc = K_fcplx.size
        N_f = N_f + N_fc
        # NOT a literal port of `scplx(K.scomplex&~sdiag)=true` -- see the
        # module docstring's "KNOWN UPSTREAM BUG NOT REPLICATED" note.
        scplx = np.zeros(L_s, dtype=bool)
        if K_scomplex.size:
            scplx[K_scomplex - 1] = True
        scplx = scplx & ~sdiag
        sreal = ~scplx & ~sdiag
        K_rsdpN = int(np.count_nonzero(sreal))
        K_cdim = int(K_xcomplex.size + (Ks[scplx].astype(np.int64) ** 2).sum())

    # ---- build the sparse transformation matrix QR from (i,j,v) triplets ----
    newL = 0
    newQ = np.zeros(0, dtype=np.int64)
    ii_list, jj_list, vv_list = [], [], []

    if "free" not in pars or (pars.get("free") == 2 and L_qrsz):
        pars["free"] = 1

    if N_f and not pars.get("free", 1):
        # `jt = [1:K.f, K.fcplx; 1:K.f, K.fcplx]; jj{end+1} = jt(:)';` --
        # MATLAB's `(:)'` flattens column-major, INTERLEAVING the two
        # identical rows (idx[0],idx[0],idx[1],idx[1],...), not
        # concatenating them (idx,idx) -- same for vt below (+1/-1 or
        # -1j/+1j interleaved per free variable, positive-part then
        # negative-part of THAT SAME variable, not all positives then
        # all negatives).
        idx = np.concatenate([np.arange(1, Kf + 1), K_fcplx]).astype(np.float64)
        jt = np.stack([idx, idx], axis=0).flatten(order="F")
        vt = np.stack(
            [
                np.concatenate([np.ones(Kf), -1j * np.ones(N_fc)]),
                np.concatenate([-np.ones(Kf), 1j * np.ones(N_fc)]),
            ],
            axis=0,
        ).flatten(order="F")
        ii_list.append(np.arange(1, 2 * N_f + 1))
        jj_list.append(jt)
        vv_list.append(vt)
        newL = 2 * N_f
        prep["freeL"] = N_f

    if Kl:
        ii_list.append(np.arange(newL + 1, newL + Kl + 1))
        jj_list.append(np.arange(Kf + 1, Kf + Kl + 1))
        vv_list.append(np.ones(Kl))
        newL += Kl

    if np.any(sdiag):
        dsize = Ks[sdiag].astype(np.int64)
        sdpL = int(dsize.sum())
        prep["sdiag"] = dsize
        jstrt_all = np.concatenate([[N_flqr + 1], (Ks.astype(np.int64) ** 2)[:-1]]).cumsum()
        jstrt = jstrt_all[sdiag]
        istrt = np.concatenate([[1], dsize[:-1]]).cumsum()
        dsize1 = dsize + 1
        dblks = np.zeros(sdpL, dtype=np.int64)
        dblks[istrt - 1] = 1
        dblks = dblks.cumsum()
        ii_list.append(np.arange(newL + 1, newL + sdpL + 1))
        jj_list.append(jstrt[dblks - 1] + dsize1[dblks - 1] * (np.arange(1, sdpL + 1) - istrt[dblks - 1]))
        vv_list.append(np.ones(sdpL))
        newL += sdpL

    tr_off = newL
    nb_off = newL + L_qr
    if N_f and pars.get("free", 1):
        tr_off += 1
        nb_off += 1
        ii_list.append(np.arange(nb_off + 1, nb_off + N_f + 1))
        jj_list.append(np.concatenate([np.arange(1, Kf + 1), K_fcplx]))
        vv_list.append(np.concatenate([np.ones(Kf), -1j * np.ones(N_fc)]))
        nb_off += Kf
        newQ = np.array([N_f + 1], dtype=np.int64)

    if N_q:
        ndxs = np.concatenate([[1], Kq[:-1]]).cumsum()
        it = np.zeros(N_q, dtype=np.int64)
        it[ndxs - 1] = np.arange(tr_off + 1, tr_off + L_q + 1)
        n_nb = N_q - L_q
        it[it == 0] = np.arange(nb_off + 1, nb_off + n_nb + 1)
        jt = np.arange(Kf + Kl + 1, Kf + Kl + N_q + 1).astype(np.float64)
        vt = np.ones(N_q, dtype=np.complex128)
        if K_qcplx.size:
            shift = np.zeros(N_q, dtype=np.int64)
            shift[K_qcplx - 1] = 1
            shift = shift.cumsum()
            jt = jt - shift
            vt[K_qcplx - 1] = -1j
        ii_list.append(it)
        jj_list.append(jt)
        vv_list.append(vt)
        tr_off += L_q
        nb_off += n_nb

    if N_r:
        ndxr = np.concatenate([[1], Kr[:-1]]).cumsum()
        ndxp = ndxr + 2 * np.arange(L_r)
        total_r = N_r + 2 * L_r
        it = np.zeros(total_r, dtype=np.int64)
        it[ndxp - 1] = np.arange(tr_off + 1, tr_off + L_r + 1)
        it[ndxp] = -1
        it[ndxp + 1] = it[ndxp - 1]
        n_nb = N_r - L_r
        zero_mask = it == 0
        it[zero_mask] = np.arange(nb_off + 1, nb_off + n_nb + 1)
        it[ndxp] = it[ndxp + 2]
        jt = np.zeros(total_r, dtype=np.float64)
        jt[0] = Kf + Kl + N_q + 1
        jt[1:] = 1.0
        jt[ndxp] = 0
        jt[ndxp + 2] = 0
        vt = np.ones(total_r, dtype=np.complex128)
        vt[ndxp - 1] = np.sqrt(0.5)
        vt[ndxp] = np.sqrt(0.5)
        vt[ndxp + 1] = np.sqrt(0.5)
        vt[ndxp + 2] = -np.sqrt(0.5)
        if K_rcplx.size:
            jt[K_rcplx - 1] = 0
            vt[K_rcplx - 1] = -1j
        ii_list.append(it)
        jj_list.append(jt.cumsum())
        vv_list.append(vt)
        nb_off += n_nb

    if K_rsdpN:
        dsize = Ks[sreal].astype(np.int64)
        sdpL = int((dsize.astype(np.int64) ** 2).sum())
        jstrt_all = np.concatenate([[N_flqr + 1], (Ks.astype(np.int64) ** 2)[:-1]]).cumsum()
        jstrt = jstrt_all[sreal]
        istrt = np.concatenate([[1], (dsize[:-1].astype(np.int64) ** 2)]).cumsum()
        dblks = np.zeros(sdpL, dtype=np.int64)
        dblks[istrt - 1] = 1
        dblks = dblks.cumsum()
        istrt_nb = istrt + nb_off
        dsize_b = dsize[dblks - 1]
        istrt_b = istrt_nb[dblks - 1]
        jndxs = np.arange(nb_off + 1, nb_off + sdpL + 1) - istrt_b
        cols = jndxs // dsize_b
        rows = jndxs - dsize_b * cols
        ii_list.append(np.maximum(rows, cols) + np.minimum(rows, cols) * dsize_b + istrt_b)
        jj_list.append((jndxs + jstrt[dblks - 1]).astype(np.float64))
        vv_list.append(np.ones(sdpL, dtype=np.complex128))
        nb_off += sdpL

    if K_rsdpN < Ks.size:
        dsize = Ks[scplx].astype(np.int64)
        jsize = dsize.astype(np.int64) ** 2
        sdpL = int(2 * jsize.sum())
        jstrt_all = np.concatenate([[N_flqr + 1], (Ks.astype(np.int64) ** 2)[:-1]]).cumsum()
        jstrt = jstrt_all[scplx]
        bstrt = np.concatenate([[1], (2 * jsize[:-1])]).cumsum()
        dblks = np.zeros(sdpL, dtype=np.int64)
        dblks[bstrt - 1] = 1
        dblks = dblks.cumsum()
        istrt = bstrt + nb_off
        dsize_b = dsize[dblks - 1]
        istrt_b = istrt[dblks - 1]
        bndxs = np.arange(nb_off + 1, nb_off + sdpL + 1) - istrt_b
        cols = bndxs // dsize_b
        rows = bndxs - dsize_b * cols
        imgv = cols >= dsize_b
        cols = cols - imgv * dsize_b
        indxs = np.maximum(rows, cols) + np.minimum(rows, cols) * dsize_b + imgv * jsize[dblks - 1] + istrt_b
        vals = 1 + imgv * (-1 + 1j * (1 - 2 * (rows > cols).astype(np.float64)))
        keep = (~imgv) | (rows != cols)
        jndxs = rows + cols * dsize_b + jstrt[dblks - 1]
        ii_list.append(indxs[keep])
        jj_list.append(jndxs[keep].astype(np.float64))
        vv_list.append(vals[keep])

    # ---- update free/nonnegative/Lorentz variable counts ----
    Kf = 0
    Kl = newL
    Kq = np.concatenate([newQ, Kq, Kr]).astype(np.int64)
    Kr = np.zeros(0, dtype=np.int64)
    Ks = np.concatenate([Ks[~scplx & ~sdiag], Ks[scplx & ~sdiag]]).astype(np.int64)
    K_rsdpN = int(np.count_nonzero(~scplx & ~sdiag))
    K_N = int(Kl + Kq.sum() + (Ks[:K_rsdpN].astype(np.int64) ** 2).sum() + 2 * (Ks[K_rsdpN:].astype(np.int64) ** 2).sum())

    # ---- self-dual (x0,z0) augmentation ----
    K_N += 1
    Kl += 1
    K_m = b.size

    ii_all = np.concatenate(ii_list) if ii_list else np.zeros(0, dtype=np.int64)
    jj_all = np.concatenate(jj_list) if jj_list else np.zeros(0, dtype=np.float64)
    vv_all = np.concatenate(vv_list) if vv_list else np.zeros(0, dtype=np.complex128)

    QR = sp.coo_matrix(
        (vv_all, (ii_all.astype(np.int64), jj_all.astype(np.int64) - 1)),
        shape=(K_N, c.size),
    ).tocsr()

    At2 = np.real(QR @ At).tocsc() if sp.issparse(At) else np.real(QR @ At)
    c2 = np.real(QR @ c)
    prep["QR"] = QR

    K2 = dict(K)
    K2["f"] = Kf
    K2["l"] = Kl
    K2["q"] = Kq
    K2["r"] = Kr
    K2["s"] = Ks
    K2["rsdpN"] = K_rsdpN
    K2["N"] = K_N
    K2["m"] = K_m
    K2["fcplx"] = K_fcplx
    K2["qcplx"] = K_qcplx
    K2["rcplx"] = K_rcplx
    K2["cdim"] = K_cdim if K_xcomplex.size or K_scomplex.size else 0

    lorN = Kq.size
    Ksr = Ks[:K_rsdpN]
    Ksc = Ks[K_rsdpN:]
    blkstart = np.concatenate(
        [[Kl + 1, lorN + Kr.size], Kq - 1, Ksr.astype(np.int64) ** 2, 2 * (Ksc.astype(np.int64) ** 2)]
    ).cumsum()
    K2["blkstart"] = blkstart
    K2["rLen"] = int(Ksr.sum())
    K2["hLen"] = int(Ksc.sum())
    K2["qMaxn"] = int(Kq.max()) if Kq.size else 0
    K2["rMaxn"] = int(Ksr.max()) if Ksr.size else 0
    K2["hMaxn"] = int(Ksc.max()) if Ksc.size else 0
    K2["mainblks"] = blkstart[np.array([0, 1, 1 + lorN])]
    K2["qblkstart"] = blkstart[1 : 2 + lorN]
    K2["sblkstart"] = blkstart[1 + lorN :]
    K2["lq"] = int(K2["mainblks"][-1] - 1)

    return At2, b, c2, K2, prep, origcoeff
