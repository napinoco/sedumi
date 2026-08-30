"""ctypes bindings over libsedumi.so (the Phase 1 standalone C library).

This is a Phase 2 starting slice, not the final binding surface: it wires
up a handful of kernels (the cone-descriptor parser and a few BLAS-backed
vector primitives) end to end, in the exact layout the C structs use, to
prove the approach before the remaining ~150 exported kernels are wired
up the same way as Phase 3 needs them.

Struct layouts (SedumiKRaw, ConeK) are hand-mirrored from blksdp.h and
must be kept in sync with it field-for-field, in declaration order --
ctypes.Structure uses the platform's normal C struct layout rules (the
System V x86_64 ABI on Linux/macOS), so as long as the field list here
matches blksdp.h's, the memory layout matches what the C side expects.
"""

from __future__ import annotations

import ctypes
import subprocess
import sys
from pathlib import Path

_PKG_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _PKG_DIR.parent.parent
_LIB_PATH = _PKG_DIR / (
    "libsedumi.dylib" if sys.platform == "darwin" else "libsedumi.so"
)


def _ensure_built() -> Path:
    if _LIB_PATH.exists():
        return _LIB_PATH
    build_script = _REPO_ROOT / "python_port" / "tools" / "build_libsedumi.sh"
    if not build_script.exists():
        raise RuntimeError(
            f"{_LIB_PATH} not found and {build_script} is missing; "
            "cannot build libsedumi automatically."
        )
    subprocess.run([str(build_script), str(_LIB_PATH)], check=True, cwd=_REPO_ROOT)
    if not _LIB_PATH.exists():
        raise RuntimeError(f"build_libsedumi.sh ran but did not produce {_LIB_PATH}")
    return _LIB_PATH


_lib = ctypes.CDLL(str(_ensure_built()))

c_size_t_p = ctypes.POINTER(ctypes.c_size_t)
c_double_p = ctypes.POINTER(ctypes.c_double)


class SedumiKRaw(ctypes.Structure):
    """Mirrors `sedumiKRaw` in blksdp.h -- field order matters."""

    _fields_ = [
        ("f", ctypes.c_double),
        ("l", ctypes.c_double),
        ("q", c_double_p),
        ("qN", ctypes.c_size_t),
        ("r", c_double_p),
        ("rN", ctypes.c_size_t),
        ("s", c_double_p),
        ("sN", ctypes.c_size_t),
        ("rsdpNgiven", ctypes.c_char),
        ("rsdpN", ctypes.c_double),
        ("statsGiven", ctypes.c_char),
        ("rLen", ctypes.c_double),
        ("hLen", ctypes.c_double),
        ("qMaxn", ctypes.c_double),
        ("rMaxn", ctypes.c_double),
        ("hMaxn", ctypes.c_double),
        ("blkstart", c_double_p),
        ("blkstartN", ctypes.c_size_t),
    ]


class ConeK(ctypes.Structure):
    """Mirrors `coneK` in blksdp.h -- field order matters."""

    _fields_ = [
        ("frN", ctypes.c_size_t),
        ("lpN", ctypes.c_size_t),
        ("lorN", ctypes.c_size_t),
        ("rconeN", ctypes.c_size_t),
        ("sdpN", ctypes.c_size_t),
        ("rsdpN", ctypes.c_size_t),
        ("qMaxn", ctypes.c_size_t),
        ("rMaxn", ctypes.c_size_t),
        ("hMaxn", ctypes.c_size_t),
        ("rLen", ctypes.c_size_t),
        ("hLen", ctypes.c_size_t),
        ("qDim", ctypes.c_size_t),
        ("rDim", ctypes.c_size_t),
        ("hDim", ctypes.c_size_t),
        ("lorNL", c_double_p),
        ("rconeNL", c_double_p),
        ("sdpNL", c_double_p),
    ]


_lib.conepars_raw.argtypes = [ctypes.POINTER(SedumiKRaw), ctypes.POINTER(ConeK)]
_lib.conepars_raw.restype = None

_lib.realdot.argtypes = [c_double_p, c_double_p, ctypes.c_size_t]
_lib.realdot.restype = ctypes.c_double

_lib.realssqr.argtypes = [c_double_p, ctypes.c_size_t]
_lib.realssqr.restype = ctypes.c_double

_lib.scalarmul.argtypes = [c_double_p, ctypes.c_double, c_double_p, ctypes.c_size_t]
_lib.scalarmul.restype = None

_lib.addscalarmul.argtypes = [c_double_p, ctypes.c_double, c_double_p, ctypes.c_size_t]
_lib.addscalarmul.restype = None

_lib.fwsolve.argtypes = [
    c_double_p,
    c_size_t_p,
    c_size_t_p,
    c_double_p,
    c_size_t_p,
    ctypes.c_size_t,
    c_double_p,
]
_lib.fwsolve.restype = None

_lib.bwsolve.argtypes = [
    c_double_p,
    c_size_t_p,
    c_size_t_p,
    c_double_p,
    c_size_t_p,
    ctypes.c_size_t,
    c_double_p,
]
_lib.bwsolve.restype = None

# ordmmd_'s parameters are all `integer *` in ordmmd.h, and `integer` is
# `mwSignedIndex` (a *signed*, pointer-width int -- ptrdiff_t under
# SEDUMI_STANDALONE), unlike the `mwIndex` (size_t, unsigned) used
# elsewhere in blksdp.h. Using the wrong signedness/width here would be a
# real ABI mismatch, not just a style choice, so this is bound with its
# own pointer type (c_ssize_t) rather than reusing c_size_t_p.
c_ssize_t_p = ctypes.POINTER(ctypes.c_ssize_t)

_lib.ordmmd_.argtypes = [c_ssize_t_p] * 9
_lib.ordmmd_.restype = ctypes.c_int

# sfinit_/symfct_ (symfct.h) use the same `integer` = mwSignedIndex
# convention as ordmmd_.
_lib.sfinit_.argtypes = [c_ssize_t_p] * 15
_lib.sfinit_.restype = ctypes.c_int

_lib.symfct_.argtypes = [c_ssize_t_p] * 17
_lib.symfct_.restype = ctypes.c_int

# expandsub (symfctmex.c) is a plain (non-Fortran-derived) C helper: n and
# nsuper are passed *by value* as mwSize (size_t, unsigned), unlike the
# by-reference mwSignedIndex convention above -- and its pointer params
# are mwIndex* (size_t*, unsigned) even though the buffers they point to
# were just filled by sfinit_/symfct_ through mwSignedIndex* (signed)
# parameters. That mixed signedness is exactly what the original
# mexFunction does too (same buffers, reinterpreted): safe because both
# are pointer-width integers and every value involved is non-negative.
_lib.expandsub.argtypes = [
    ctypes.c_size_t,
    ctypes.c_size_t,
    c_size_t_p,
    c_size_t_p,
    c_size_t_p,
    c_size_t_p,
]
_lib.expandsub.restype = None

_lib.gettmpsiz.argtypes = [c_size_t_p, c_size_t_p, c_size_t_p, ctypes.c_size_t, c_size_t_p]
_lib.gettmpsiz.restype = ctypes.c_size_t

_lib.permuteP.argtypes = [
    c_size_t_p, c_size_t_p, c_double_p,
    c_size_t_p, c_size_t_p, c_double_p,
    c_size_t_p, c_double_p, ctypes.c_size_t,
]
_lib.permuteP.restype = None

_lib.spchol.argtypes = [
    ctypes.c_size_t, ctypes.c_size_t, c_size_t_p,
    c_size_t_p, c_size_t_p, c_size_t_p, c_double_p,
    c_size_t_p, c_double_p, c_double_p, c_size_t_p,
    ctypes.c_double, ctypes.c_double, ctypes.c_double,
    c_size_t_p, c_size_t_p,
    ctypes.c_size_t, c_size_t_p, ctypes.c_size_t, c_double_p,
]
_lib.spchol.restype = ctypes.c_size_t

_SIZE_T_ERROR = (1 << 64) - 1  # (mwIndex)-1 sentinel: spchol/blkLDL's
# "insufficient workspace" return value, wrapped around through mwIndex
# being unsigned (size_t) -- ctypes.c_size_t surfaces it as this value
# rather than -1.


def _compute_snode(xsuper, m):
    """Mirrors the small "map each column to its supernode" loop that
    appears identically in choltmpsiz.c and blkchol.c's spchol():
        j = xsuper[0]
        for jsup in range(nsuper): while j < xsuper[jsup+1]: snode[j++] = jsup
    """
    import numpy as np

    nsuper = len(xsuper) - 1
    counts = np.diff(np.asarray(xsuper, dtype=np.int64))
    return np.repeat(np.arange(nsuper, dtype=np.int64), counts).astype(np.uintp)


def _as_double_array(x):
    import numpy as np

    arr = np.ascontiguousarray(x, dtype=np.float64)
    ptr = arr.ctypes.data_as(c_double_p)
    return arr, ptr


def realdot(x, y) -> float:
    """r = sum(x_i * y_i), via the BLAS-backed C kernel (sdmauxRdot.c)."""
    xa, xp = _as_double_array(x)
    ya, yp = _as_double_array(y)
    if xa.shape != ya.shape:
        raise ValueError(f"shape mismatch: {xa.shape} vs {ya.shape}")
    return _lib.realdot(xp, yp, xa.size)


def realssqr(x) -> float:
    """r = sum(x_i^2), via the BLAS-backed C kernel (sdmauxRdot.c)."""
    xa, xp = _as_double_array(x)
    return _lib.realssqr(xp, xa.size)


def scalarmul(alpha: float, x):
    """r = alpha * x, via the BLAS-backed C kernel (sdmauxScalarmul.c)."""
    import numpy as np

    xa, xp = _as_double_array(x)
    out = np.empty_like(xa)
    outp = out.ctypes.data_as(c_double_p)
    _lib.scalarmul(outp, float(alpha), xp, xa.size)
    return out


def addscalarmul(r, alpha: float, x):
    """r += alpha * x, via the BLAS-backed C kernel (sdmauxScalarmul.c)."""
    ra, rp = _as_double_array(r)
    xa, xp = _as_double_array(x)
    _lib.addscalarmul(rp, float(alpha), xp, xa.size)
    return ra


def _as_index_array(x):
    import numpy as np

    arr = np.ascontiguousarray(x, dtype=np.uintp)
    ptr = arr.ctypes.data_as(c_size_t_p)
    return arr, ptr


def fwsolve(L_csc, xsuper, y):
    """Forward-solve y := L \\ y in place, where L is the unit-lower-
    triangular factor from SeDuMi's supernodal Cholesky (blkchol), stored
    exactly like `scipy.sparse.csc_matrix` (indptr/indices/data), and
    `xsuper` marks each supernode's first column (length nsuper+1, as
    produced by symbchol). L's stored diagonal entries are never read --
    only their presence in the sparsity pattern matters, per fwblkslv.c.

    This wraps fwsolve() in fwblkslv.c (sedumi's forward substitution
    kernel) with no MATLAB/Octave/MEX layer at all.
    """
    Ljc, Ljc_p = _as_index_array(L_csc.indptr)
    Lir, Lir_p = _as_index_array(L_csc.indices)
    Lpr, Lpr_p = _as_double_array(L_csc.data)
    xs, xs_p = _as_index_array(xsuper)
    ya, y_p = _as_double_array(y)

    nsuper = xs.size - 1
    m = L_csc.shape[0]
    fwork = ctypes.create_string_buffer(8 * m)  # generous upper bound
    fwork_p = ctypes.cast(fwork, c_double_p)

    _lib.fwsolve(y_p, Ljc_p, Lir_p, Lpr_p, xs_p, nsuper, fwork_p)
    return ya


def bwsolve(L_csc, xsuper, y):
    """Backward-solve y := L' \\ y in place -- see fwsolve()'s docstring;
    wraps bwsolve() in bwblkslv.c."""
    Ljc, Ljc_p = _as_index_array(L_csc.indptr)
    Lir, Lir_p = _as_index_array(L_csc.indices)
    Lpr, Lpr_p = _as_double_array(L_csc.data)
    xs, xs_p = _as_index_array(xsuper)
    ya, y_p = _as_double_array(y)

    nsuper = xs.size - 1
    m = L_csc.shape[0]
    fwork = ctypes.create_string_buffer(8 * m)  # generous upper bound
    fwork_p = ctypes.cast(fwork, c_double_p)

    _lib.bwsolve(y_p, Ljc_p, Lir_p, Lpr_p, xs_p, nsuper, fwork_p)
    return ya


def _to_fortran_adjacency(A_csc):
    """Mirrors getadj() in ordmmdmex.c/symfctmex.c exactly: converts a
    0-indexed CSC sparsity pattern (diagonal entries included or not, both
    fine) into the 1-indexed (xadj, adjncy) adjacency-list form Liu's
    Fortran ordmmd_/symfct_ expect, dropping diagonal entries. `A_csc`
    must be symmetric (only one triangle need be nonzero for the sparsity
    pattern, but symbchol.m always passes a symmetric ADA_sedumi_) and is
    sorted by row index within each column first, to match MATLAB/
    Octave's own sparse storage -- this determines the exact adjacency
    order ordmmd_/symfct_ see, which affects tie-breaking, so must match
    bit-for-bit to reproduce the same ordering as the MEX build.
    """
    import numpy as np

    A = A_csc.copy()
    A.sort_indices()
    n = A.shape[0]
    cjc = A.indptr
    cir = A.indices

    xadj = np.empty(n + 1, dtype=np.intp)
    adjncy = np.empty(cjc[n], dtype=np.intp)  # upper bound; diag entries excluded
    inz = 0
    for j in range(n):
        xadj[j] = inz + 1
        for ix in range(cjc[j], cjc[j + 1]):
            i = cir[ix]
            if i != j:
                adjncy[inz] = i + 1
                inz += 1
    xadj[n] = inz + 1
    return xadj, adjncy[:inz].copy()


def ordmmd(A_csc):
    """Multiple minimum degree ordering (Liu's genmmd, via ordmmd.c) --
    the exact same algorithm and C source SeDuMi's ordmmdmex() MEX
    function calls, just without any MATLAB/Octave/MEX in the calling
    path. `A_csc` must be a square, symmetric scipy.sparse.csc_matrix (as
    symbchol.m always passes -- only the sparsity pattern matters).

    Returns a 0-indexed permutation array `perm` (Python/NumPy
    convention); MATLAB/Octave's ordmmdmex returns the 1-indexed version
    of the exact same array.
    """
    import numpy as np

    n = A_csc.shape[0]
    if A_csc.shape[0] != A_csc.shape[1]:
        raise ValueError("A_csc must be square")

    xadj, adjncy = _to_fortran_adjacency(A_csc)
    perm = np.zeros(n, dtype=np.intp)
    invp = np.zeros(n, dtype=np.intp)
    iwsiz = 4 * n
    iwork = np.zeros(max(iwsiz, 1), dtype=np.intp)

    neqns = ctypes.c_ssize_t(n)
    iwsiz_c = ctypes.c_ssize_t(iwsiz)
    nofsub = ctypes.c_ssize_t(0)
    iflag = ctypes.c_ssize_t(0)

    def p(arr):
        return arr.ctypes.data_as(c_ssize_t_p)

    _lib.ordmmd_(
        ctypes.byref(neqns),
        p(xadj),
        p(adjncy),
        p(invp),
        p(perm),
        ctypes.byref(iwsiz_c),
        p(iwork),
        ctypes.byref(nofsub),
        ctypes.byref(iflag),
    )
    if iflag.value == -1:
        raise RuntimeError("ordmmd: insufficient working storage (iwsiz too small)")
    return perm - 1


def symbolic_cholesky(A_csc, perm0):
    """Symbolic block-sparse Cholesky factorization: mirrors
    `L = symfctmex(X, perm)` (symfctmex.c, Liu's SPARSPAK-A sfinit_/
    symfct_) exactly -- same C/Fortran source, no MATLAB/Octave/MEX in
    the calling path. This is the real (non-dense-fallback) branch of
    symbchol.m.

    Parameters
    ----------
    A_csc : scipy.sparse.csc_matrix, symmetric m x m sparsity pattern
        (as symbchol.m's ADA_sedumi_).
    perm0 : 0-indexed initial permutation (e.g. from ordmmd()).

    Returns
    -------
    dict with:
      "L"      -- scipy.sparse.csc_matrix, the symbolic factor's
                  sparsity pattern (all data entries are 1.0 -- this is
                  a symbolic factorization, not a numeric one), matching
                  the MATLAB/Octave L.L field.
      "perm"   -- 0-indexed final permutation (matching L.perm; sfinit_
                  can refine the input ordmmd() permutation to merge
                  same-pattern columns into supernodes, so this is not
                  always identical to perm0).
      "xsuper" -- 0-indexed supernode boundaries, length nsuper+1
                  (matching L.xsuper).
    """
    import numpy as np

    m = A_csc.shape[0]
    if A_csc.shape[0] != A_csc.shape[1]:
        raise ValueError("A_csc must be square")

    xadj, adjncy = _to_fortran_adjacency(A_csc)
    nnza = A_csc.nnz  # matches Xjc[m] in symfctmex.c: X's own total stored
    # nnz (diagonal included), used only as an upper bound for workspace
    # sizing, not literally len(adjncy) (which excludes the diagonal).

    perm = (np.asarray(perm0, dtype=np.intp) + 1).copy()  # 1-indexed, mutable:
    # sfinit_ updates (perm, invp) in place to an equivalent ordering, so
    # this buffer's *final* contents (after both Fortran calls below) is
    # the true output permutation -- exactly what symfctmex.c reads back.
    invp = np.zeros(m, dtype=np.intp)
    for i in range(m):
        invp[perm[i] - 1] = i + 1

    colcnt = np.zeros(m, dtype=np.intp)
    snode = np.zeros(m, dtype=np.intp)
    xsuper = np.zeros(m + 1, dtype=np.intp)
    iwsiz = 7 * m + 3
    iwork = np.zeros(max(iwsiz, 1), dtype=np.intp)

    def p(arr):
        return arr.ctypes.data_as(c_ssize_t_p)

    m_c = ctypes.c_ssize_t(m)
    nnza_c = ctypes.c_ssize_t(nnza)
    nnzl = ctypes.c_ssize_t(0)
    nsub = ctypes.c_ssize_t(0)
    nsuper_c = ctypes.c_ssize_t(0)
    iwsiz_c = ctypes.c_ssize_t(iwsiz)
    flag = ctypes.c_ssize_t(0)

    _lib.sfinit_(
        ctypes.byref(m_c), ctypes.byref(nnza_c), p(xadj), p(adjncy),
        p(perm), p(invp), p(colcnt),
        ctypes.byref(nnzl), ctypes.byref(nsub), ctypes.byref(nsuper_c),
        p(snode), p(xsuper), ctypes.byref(iwsiz_c), p(iwork), ctypes.byref(flag),
    )
    if flag.value == -1:
        raise RuntimeError("sfinit: insufficient working storage (iwsiz too small)")

    xlindx = np.zeros(m + 1, dtype=np.intp)
    # Lir/Ljc are allocated once and reused for two different purposes in
    # sequence -- exactly as symfctmex.c does: symfct_ first fills them as
    # (lindx, xlnz) -- a compact, per-supernode, 1-indexed representation
    # -- then expandsub() overwrites the same buffers in place with the
    # standard, per-column, 0-indexed CSC representation (Ljc capacity
    # m+1 is already enough for both; Lir's nnzl capacity from sfinit_ is
    # an upper bound valid for both the compact and the expanded form).
    Lir = np.zeros(max(nnzl.value, 1), dtype=np.intp)
    Ljc = np.zeros(m + 1, dtype=np.intp)

    flag2 = ctypes.c_ssize_t(0)
    _lib.symfct_(
        ctypes.byref(m_c), ctypes.byref(nnza_c), p(xadj), p(adjncy),
        p(perm), p(invp), p(colcnt),
        ctypes.byref(nsuper_c), p(xsuper), p(snode),
        ctypes.byref(nsub), p(xlindx), p(Lir), p(Ljc),
        ctypes.byref(iwsiz_c), p(iwork), ctypes.byref(flag2),
    )
    if flag2.value == -1:
        raise RuntimeError("symfct: insufficient working storage (iwsiz too small)")
    if flag2.value == -2:
        raise RuntimeError("symfct: input error")

    def p_size_t(arr):
        return arr.ctypes.data_as(c_size_t_p)

    _lib.expandsub(m, nsuper_c.value, p_size_t(xsuper), p_size_t(xlindx),
                    p_size_t(Ljc), p_size_t(Lir))

    nnz_L = int(Ljc[m])
    import scipy.sparse

    L_csc = scipy.sparse.csc_matrix(
        (np.ones(nnz_L, dtype=np.float64), Lir[:nnz_L].astype(np.int64),
         Ljc.astype(np.int64)),
        shape=(m, m),
    )

    final_xsuper = xsuper[: nsuper_c.value + 1] - 1
    snode_for_tmpsiz = _compute_snode(final_xsuper, m).copy()
    tmpsiz = _lib.gettmpsiz(
        Ljc.astype(np.uintp).ctypes.data_as(c_size_t_p),
        Lir[:nnz_L].astype(np.uintp).ctypes.data_as(c_size_t_p),
        final_xsuper.astype(np.uintp).ctypes.data_as(c_size_t_p),
        nsuper_c.value,
        snode_for_tmpsiz.ctypes.data_as(c_size_t_p),
    )

    return {
        "L": L_csc,
        "perm": perm - 1,
        "xsuper": final_xsuper,
        "tmpsiz": int(tmpsiz),
    }


def numeric_cholesky(sym: dict, X_csc, pars: dict | None = None, absd=None) -> dict:
    """Numeric block sparse LDL' factorization: mirrors
    `[L.L, L.d, skip, diagadd] = blkchol(L, X, pars, absd)` (blkchol.c's
    permuteP + spchol/blkLDL) exactly, no MATLAB/Octave/MEX in the
    calling path.

    Parameters
    ----------
    sym : dict from symbolic_cholesky() -- needs "L" (pattern), "perm"
        (0-indexed), "xsuper" (0-indexed), "tmpsiz".
    X_csc : scipy.sparse.csc_matrix, the numeric matrix to factor (only
        its lower triangle is read, matching P(perm,perm) -- SeDuMi
        always builds X to already be symmetric with only tril stored).
    pars : optional dict overriding canceltol (1e-12), maxu (5e2),
        abstol (1e-20), delay (False) -- same names/defaults as blkchol.c.
    absd : optional length-m array of "before cancellation" diagonal
        magnitudes (pars.absd in the .m API); defaults to X's own
        diagonal.

    Returns dict with "L" (scipy.sparse.csc_matrix, same pattern as
    sym["L"], numeric values), "d" (length-m diagonal of D, with
    d[skip]==0 as blkchol always reports it), "skip" (0-indexed columns
    where the pivot was too unstable and got replaced with a unit
    column -- matching `L.d(find(L.skip)) = inf` in blkchol.m's own
    solve recipe), "diagadd" (values added to the diagonal at the
    OTHER unstable pivots that were stabilized instead of skipped).
    """
    import numpy as np
    import scipy.sparse

    m = X_csc.shape[0]
    L_pattern = sym["L"].tocsc()
    perm = np.ascontiguousarray(sym["perm"], dtype=np.uintp)
    xsuper = np.ascontiguousarray(sym["xsuper"], dtype=np.uintp)
    nsuper = xsuper.size - 1
    tmpsiz = sym["tmpsiz"]

    pars = pars or {}
    canceltol = float(pars.get("canceltol", 1e-12))
    maxu = float(pars.get("maxu", 5e2))
    abstol = max(float(pars.get("abstol", 1e-20)), 0.0)
    use_delay = bool(pars.get("delay", False))

    def p_size_t(arr):
        return arr.ctypes.data_as(c_size_t_p)

    def p_double(arr):
        return arr.ctypes.data_as(c_double_p)

    Ljc = np.ascontiguousarray(L_pattern.indptr, dtype=np.uintp)
    Lir_original = np.ascontiguousarray(L_pattern.indices, dtype=np.uintp)
    Lir = Lir_original.copy()  # spchol uses this as scratch, restored after
    Lpr = np.zeros(int(Ljc[-1]), dtype=np.float64)

    Pj = np.zeros(max(m, 1), dtype=np.float64)  # permuteP's own scratch space
    X = X_csc.tocsc()
    Pjc = np.ascontiguousarray(X.indptr, dtype=np.uintp)
    Pir = np.ascontiguousarray(X.indices, dtype=np.uintp)
    Ppr = np.ascontiguousarray(X.data, dtype=np.float64)

    _lib.permuteP(
        p_size_t(Ljc), p_size_t(Lir), p_double(Lpr),
        p_size_t(Pjc), p_size_t(Pir), p_double(Ppr),
        p_size_t(perm), p_double(Pj), m,
    )

    if absd is not None:
        absd_arr = np.ascontiguousarray(absd, dtype=np.float64)
        orgd = absd_arr[perm.astype(np.int64)].copy()
    else:
        orgd = np.array([Lpr[Ljc[j]] for j in range(m)], dtype=np.float64)

    snode = np.zeros(max(m, 1), dtype=np.uintp)
    xlindx = np.zeros(m + 1, dtype=np.uintp)
    d = np.zeros(m, dtype=np.float64)
    skip = np.zeros(max(m, 1), dtype=np.uintp)
    nadd_c = ctypes.c_size_t(0)

    iwsiz = max(2 * (m + nsuper), 1)
    fwsiz = max(tmpsiz, 1)
    iwork = np.zeros(iwsiz, dtype=np.uintp)
    fwork = np.zeros(fwsiz, dtype=np.float64)

    nskip = _lib.spchol(
        m, nsuper, p_size_t(xsuper),
        p_size_t(snode), p_size_t(xlindx), p_size_t(Lir), p_double(orgd),
        p_size_t(Ljc), p_double(Lpr), p_double(d), p_size_t(perm),
        abstol, canceltol, maxu,
        p_size_t(skip), ctypes.byref(nadd_c),
        iwsiz, p_size_t(iwork), fwsiz, p_double(fwork),
    )
    if nskip == _SIZE_T_ERROR:
        raise RuntimeError("spchol: insufficient working storage (iwsiz/fwsiz too small)")
    nadd = nadd_c.value

    # spchol used Lir as scratch (the "compress subscripts" step turns it
    # into a per-supernode compact array); the *sparsity pattern* of the
    # output L is unchanged from the input, so restore it -- exactly the
    # memcpy(L.ir, LINir, ...) in blkchol.c's mexFunction.
    Lir[:] = Lir_original

    skip = skip[:nskip]
    diagadd_idx = np.zeros(nadd, dtype=np.uintp)
    diagadd_val = np.zeros(nadd, dtype=np.float64)

    skip_out = []
    skip_val = []
    for j in range(nskip):
        i = int(skip[j])
        if use_delay:
            skip_val.append(1.0)
        else:
            skip_val.append(Lpr[Ljc[i]])
            Lpr[Ljc[i]] = 1.0
            Lpr[Ljc[i] + 1 : Ljc[i + 1]] = 0.0
        skip_out.append(i)

    # iwork[:nadd] holds the diagadd indices, written by spchol.
    for j in range(nadd):
        i = int(iwork[j])
        diagadd_idx[j] = i
        diagadd_val[j] = orgd[i]

    L_csc = scipy.sparse.csc_matrix(
        (Lpr, Lir.astype(np.int64), Ljc.astype(np.int64)), shape=(m, m)
    )

    return {
        "L": L_csc,
        "d": d,
        "skip": np.array(skip_out, dtype=np.int64),
        "skip_values": np.array(skip_val, dtype=np.float64),
        "diagadd_index": diagadd_idx.astype(np.int64),
        "diagadd": diagadd_val,
    }


_lib.ddotxj.argtypes = [c_double_p, c_double_p, c_double_p, c_size_t_p, ctypes.c_size_t]
_lib.ddotxj.restype = None

_lib.blkmul.argtypes = [c_double_p, c_double_p, c_double_p, c_size_t_p, ctypes.c_size_t, ctypes.c_size_t]
_lib.blkmul.restype = ctypes.c_int

_lib.vecsymPSD.argtypes = [c_double_p, c_double_p, ctypes.c_size_t, ctypes.c_size_t, c_double_p]
_lib.vecsymPSD.restype = None

_lib.rquaddadd.argtypes = [c_double_p, ctypes.c_double, ctypes.c_double, ctypes.c_double]
_lib.rquaddadd.restype = ctypes.c_double


def ddot(d, X, blkstart):
    """ddot(d, X, blkstart) -- dense-X path of ddot.c/ddotxj: for each
    column of X and each Lorentz block k (spanning blkstart[k]:blkstart[k+1]
    in 0-indexed, half-open convention), computes d[k]'*X[block,column].
    Wraps ddotxj() directly (no MATLAB/Octave/MEX); the sparse-X path
    (spddotxj) is not yet wrapped -- Phase 3 will add it if/when the .m
    port actually needs ddot on sparse X.
    """
    import numpy as np

    d = np.ascontiguousarray(d, dtype=np.float64)
    X = np.ascontiguousarray(X, dtype=np.float64)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    blkstart = np.ascontiguousarray(blkstart, dtype=np.uintp)
    nblk = blkstart.size - 1
    nrows, ncols = X.shape

    qDim = blkstart[-1] - blkstart[0]
    if d.size != qDim:
        d = d[int(blkstart[0]) :]

    out = np.empty((nblk, ncols), dtype=np.float64, order="F")
    bs = (blkstart - blkstart[0]).astype(np.uintp)  # ddotxj asserts blkstart[0]==0
    dptr = d.ctypes.data_as(c_double_p)
    bsptr = bs.ctypes.data_as(c_size_t_p)
    for j in range(ncols):
        col = np.ascontiguousarray(X[:, j])
        _lib.ddotxj(
            out[:, j].ctypes.data_as(c_double_p), dptr,
            col.ctypes.data_as(c_double_p), bsptr, nblk,
        )
    return out.squeeze(axis=1) if ncols == 1 else out


def blkmul(mu, d, nL):
    """y[block k] = mu[k] * d[block k], blocks given by nL (block LENGTHS,
    not offsets -- see blkmul.m). Wraps blkmul() (blkmul.c) directly."""
    import numpy as np

    mu = np.ascontiguousarray(mu, dtype=np.float64).ravel()
    d = np.ascontiguousarray(d, dtype=np.float64).ravel()
    nL_arr = np.ascontiguousarray(nL, dtype=np.uintp).ravel()
    kappa = mu.size
    n = d.size
    y = np.zeros(n, dtype=np.float64)

    remaining = _lib.blkmul(
        y.ctypes.data_as(c_double_p), mu.ctypes.data_as(c_double_p),
        d.ctypes.data_as(c_double_p), nL_arr.ctypes.data_as(c_size_t_p),
        kappa, n,
    )
    if remaining != 0:
        raise ValueError("blkmul: nL size mismatch (sum(nL) != len(d))")
    return y


def qblkmul(mu, d, blkstart):
    """y[block k] = mu[k] * d[block k], blocks given by blkstart
    (1-indexed, as in the .m/MEX convention -- see qblkmul.m). qblkmul.c
    has no separable core function (the whole computation lives in its
    mexFunction), so this ports that logic directly to NumPy rather than
    binding a C function that doesn't exist as such."""
    import numpy as np

    mu = np.ascontiguousarray(mu, dtype=np.float64).ravel()
    d = np.ascontiguousarray(d, dtype=np.float64).ravel()
    blkstart = np.ascontiguousarray(blkstart, dtype=np.int64).ravel() - 1
    nblk = mu.size
    span = int(blkstart[-1] - blkstart[0])

    if d.size != span:
        if d.size == nblk + span:
            d = d[nblk:]
        else:
            d = d[int(blkstart[0]) :]

    out = np.empty(span, dtype=np.float64)
    pos = 0
    for k in range(nblk):
        nk = int(blkstart[k + 1] - blkstart[k])
        out[pos : pos + nk] = mu[k] * d[pos : pos + nk]
        pos += nk
    return out


def vecsym(x, K: dict):
    """y = vecsym(x, K): copies the LP+SOCP part of x unchanged, then
    symmetrizes each real PSD block ((Xk+Xk')/2) and Hermitianizes each
    complex one, via vecsymPSD() (vecsym.c) directly."""
    import numpy as np

    cK = cone_from_dict(K)
    x = np.ascontiguousarray(x, dtype=np.float64).ravel()
    lqDim = cK.lpN + cK.qDim
    lenfull = lqDim + cK.rDim + cK.hDim
    if x.size != lenfull:
        raise ValueError(f"x must have length {lenfull}, got {x.size}")

    y = x.copy()
    sdpNL = cK._keepalive[2]  # the "s" array cone_from_dict built cK from
    _lib.vecsymPSD(
        y[lqDim:].ctypes.data_as(c_double_p),
        x[lqDim:].ctypes.data_as(c_double_p),
        cK.rsdpN, cK.sdpN,
        sdpNL.ctypes.data_as(c_double_p) if cK.sdpN else None,
    )
    return y


def quadadd(xhi, xlo, y):
    """(zhi, zlo) = quadadd(xhi, xlo, y): extended-precision (double-
    double style) addition xhi+xlo+y, elementwise, via rquaddadd()
    (quadadd.c) directly -- not reimplemented in Python, since the whole
    point of this kernel is the specific extended-precision arithmetic
    sequence, which is easy to get subtly wrong by "simplifying"."""
    import numpy as np

    xhi = np.ascontiguousarray(xhi, dtype=np.float64).ravel()
    xlo = np.ascontiguousarray(xlo, dtype=np.float64).ravel()
    y = np.ascontiguousarray(y, dtype=np.float64).ravel()
    m = xhi.size
    zhi = np.empty(m, dtype=np.float64)
    zlo = np.empty(m, dtype=np.float64)
    for i in range(m):
        lo = ctypes.c_double(0.0)
        zhi[i] = _lib.rquaddadd(ctypes.byref(lo), xhi[i], xlo[i], y[i])
        zlo[i] = lo.value
    return zhi, zlo


class KeyDouble(ctypes.Structure):
    """Mirrors blksdp.h's `keydouble` (double r; mwIndex k;)."""

    _fields_ = [("r", ctypes.c_double), ("k", ctypes.c_size_t)]


c_ubyte_p = ctypes.POINTER(ctypes.c_ubyte)  # for `char*`/`bool*` buffers --
# deliberately not ctypes.c_char_p, which has Python-string marshaling
# semantics that are the wrong fit for a plain output byte buffer.

_lib.fwprodform.argtypes = [
    c_double_p, c_size_t_p, c_size_t_p, c_double_p, c_double_p, c_size_t_p,
    c_ubyte_p, ctypes.c_size_t,
]
_lib.fwprodform.restype = None

_lib.bwprodform.argtypes = [
    c_double_p, c_size_t_p, c_size_t_p, c_double_p, c_double_p, c_size_t_p,
    c_ubyte_p, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t,
]
_lib.bwprodform.restype = None

_lib.prodformfact.argtypes = [
    c_double_p, c_size_t_p, c_double_p, c_size_t_p,
    c_double_p, c_ubyte_p, c_size_t_p,
    c_size_t_p, c_size_t_p,
    c_double_p, ctypes.c_size_t, c_size_t_p, c_size_t_p,
    ctypes.c_double, c_double_p, ctypes.POINTER(KeyDouble),
]
_lib.prodformfact.restype = None


def _dpr1_apply(direction, Lden: dict, b):
    """Shared implementation for fwdpr1()/bwdpr1(): PROD_k L(pk,betak) *
    ynew = yold (forward) or PROD_k L(pk,betak)' * ynew = yold (backward),
    where L(p,beta) = eye(m) + tril(p*beta',-1). Wraps fwprodform/
    bwprodform (fwdpr1.c/bwdpr1.c) directly.

    Lden needs "betajc" (1-indexed, length nden+1; nden==0 means no dense
    columns at all -- y is returned unchanged, exactly like the MEX
    build's early return), "p", "beta", "pivperm", "dopiv", and "dz" (a
    sparse matrix whose .indptr is used as the per-column cumulative
    "reach" xsuper and whose .indices give the row-compaction mapping --
    see dpr1fact()'s docstring for what this "dz" scheme actually means).
    """
    import numpy as np

    betajc_1indexed = np.ascontiguousarray(Lden["betajc"], dtype=np.int64).ravel()
    nden = betajc_1indexed.size - 1
    b = np.ascontiguousarray(b, dtype=np.float64)
    if nden == 0:
        return b.copy()

    single_col = b.ndim == 1
    B = b.reshape(-1, 1) if single_col else b
    m = B.shape[0]

    betajc = (betajc_1indexed - 1).astype(np.uintp)
    p = np.ascontiguousarray(Lden["p"], dtype=np.float64)
    beta = np.ascontiguousarray(Lden["beta"], dtype=np.float64)
    ordered = np.ascontiguousarray(Lden["dopiv"], dtype=np.uint8).ravel()
    # pivperm is an opaque, internal 0-indexed array private to the
    # dpr1fact()<->fwdpr1()/bwdpr1() pairing -- never interpreted as a
    # MATLAB-facing 1-indexed permutation anywhere, including by the
    # original C mexFunctions (they round-trip it unchanged), so no
    # +-1 conversion happens here either.
    pivperm = np.ascontiguousarray(Lden["pivperm"], dtype=np.uintp).ravel()
    dz_jc = np.ascontiguousarray(Lden["dz"].indptr, dtype=np.uintp)
    dz_ir = np.ascontiguousarray(Lden["dz"].indices, dtype=np.uintp)
    dznnz = int(dz_jc[nden])

    Y = B.copy()
    fwork = np.empty(max(dznnz, 1), dtype=np.float64)
    for j in range(Y.shape[1]):
        col = Y[:, j]
        for i in range(dznnz):
            fwork[i] = col[dz_ir[i]]
        if direction == "fw":
            _lib.fwprodform(
                fwork.ctypes.data_as(c_double_p), dz_jc.ctypes.data_as(c_size_t_p),
                pivperm.ctypes.data_as(c_size_t_p), p.ctypes.data_as(c_double_p),
                beta.ctypes.data_as(c_double_p), betajc.ctypes.data_as(c_size_t_p),
                ordered.ctypes.data_as(c_ubyte_p), nden,
            )
        else:
            # bwprodform additionally needs the *total* lengths of p and
            # pivperm up front (it walks backward, decrementing into
            # them), unlike fwprodform which only needs cumulative
            # offsets it can derive from dz_jc as it goes forward.
            _lib.bwprodform(
                fwork.ctypes.data_as(c_double_p), dz_jc.ctypes.data_as(c_size_t_p),
                pivperm.ctypes.data_as(c_size_t_p), p.ctypes.data_as(c_double_p),
                beta.ctypes.data_as(c_double_p), betajc.ctypes.data_as(c_size_t_p),
                ordered.ctypes.data_as(c_ubyte_p), nden, p.size, pivperm.size,
            )
        for i in range(dznnz):
            col[dz_ir[i]] = fwork[i]
    return Y[:, 0] if single_col else Y


def fwdpr1(Lden: dict, b):
    """y = fwdpr1(Lden, b): solve PROD_k L(pk,betak) * y = b. See
    _dpr1_apply()'s docstring for Lden's fields."""
    return _dpr1_apply("fw", Lden, b)


def bwdpr1(Lden: dict, b):
    """y = bwdpr1(Lden, b): solve PROD_k L(pk,betak)' * y = b. See
    _dpr1_apply()'s docstring for Lden's fields."""
    return _dpr1_apply("bw", Lden, b)


def dpr1fact(x, d, Lsym: dict, smult, maxu: float):
    """[Lden, d_out] = dpr1fact(x, d, Lsym, smult, maxu): factors
    diag(d) + x*diag(smult)*x' = (PROD_k L(pk,betak)) * diag(d_out) *
    (PROD_k L(pk,betak))', wrapping prodformfact() (dpr1fact.c) directly
    -- the same C computation the MEX build uses for SeDuMi's
    "dense column" handling in the PCG preconditioner.

    x : scipy.sparse.csc_matrix, m x n (n = number of dense columns).
    d : length-m array, diagonal to update.
    Lsym : dict with "dz" (scipy.sparse.csc_matrix -- see below), "perm"
        (1-indexed length-n column order, as symfctmex-style outputs
        use), "first" (1-indexed length-n array).
    smult : length-n array of per-column multipliers (x*diag(smult)*x').
    maxu : stability threshold -- a column gets pivoted (reordered) if a
        pivot magnitude ratio would otherwise exceed this.

    Returns a dict shaped like Lden (see _dpr1_apply()'s docstring) plus
    the updated diagonal, ready to feed to fwdpr1()/bwdpr1() after also
    copying over Lsym's dz/perm/first fields (as deninfac.m does):
        Lden["dz"], Lden["first"], Lden["perm"] = Lsym["dz"], Lsym["first"], Lsym["perm"]
    """
    import numpy as np
    import scipy.sparse

    X = x.tocsc()
    m, n = X.shape
    dz = Lsym["dz"].tocsc()
    dz_jc = np.ascontiguousarray(dz.indptr, dtype=np.uintp)
    dz_ir = np.ascontiguousarray(dz.indices, dtype=np.uintp)
    dznnz = int(dz_jc[n])
    if dznnz > m:
        raise ValueError("Lsym.dz size mismatch: more compact rows than m")

    colperm = (np.ascontiguousarray(Lsym["perm"], dtype=np.int64).ravel() - 1).astype(np.uintp)
    firstpiv = (np.ascontiguousarray(Lsym["first"], dtype=np.int64).ravel() - 1).astype(np.uintp)

    pnnz = int(sum(int(dz_jc[j + 1]) for j in range(n)))
    d_compact = np.empty(max(dznnz, 1), dtype=np.float64)
    lab = np.ascontiguousarray(d, dtype=np.float64).copy()
    for i in range(dznnz):
        d_compact[i] = lab[dz_ir[i]]

    dep = np.zeros(dznnz + 1, dtype=np.uintp)
    ndep = 0
    for i in range(dznnz):
        if d_compact[i] <= 0.0:
            dep[ndep] = i
            ndep += 1
    dep[ndep] = m

    invrowperm = np.zeros(max(m, 1), dtype=np.uintp)
    for i in range(dznnz):
        invrowperm[dz_ir[i]] = i

    p = np.zeros(max(pnnz + m, 1), dtype=np.float64)
    pos = 0
    for j in range(n):
        pos += int(dz_jc[j])
        permj = int(colperm[j])
        for i in range(X.indptr[permj], X.indptr[permj + 1]):
            p[pos + int(invrowperm[X.indices[i]])] = X.data[i]

    p = p[:pnnz].copy() if pnnz > 0 else np.zeros(0, dtype=np.float64)
    beta = np.zeros(max(pnnz, 1), dtype=np.float64)
    betajc = np.zeros(n + 1, dtype=np.uintp)
    ordered = np.zeros(max(n, 1), dtype=np.uint8)
    pivperm = np.zeros(max(pnnz, 1), dtype=np.uintp)
    fwork = np.zeros(max(dznnz, 1), dtype=np.float64)
    kdwork = (KeyDouble * max(dznnz, 1))()
    ndep_c = ctypes.c_size_t(ndep)
    smult_arr = np.ascontiguousarray(smult, dtype=np.float64)

    _lib.prodformfact(
        p.ctypes.data_as(c_double_p), pivperm.ctypes.data_as(c_size_t_p),
        beta.ctypes.data_as(c_double_p), betajc.ctypes.data_as(c_size_t_p),
        d_compact.ctypes.data_as(c_double_p), ordered.ctypes.data_as(c_ubyte_p),
        dz_jc.ctypes.data_as(c_size_t_p),
        colperm.ctypes.data_as(c_size_t_p), firstpiv.ctypes.data_as(c_size_t_p),
        smult_arr.ctypes.data_as(c_double_p), n, dep.ctypes.data_as(c_size_t_p),
        ctypes.byref(ndep_c),
        maxu, fwork.ctypes.data_as(c_double_p), kdwork,
    )

    for i in range(dznnz):
        lab[dz_ir[i]] = d_compact[i]

    # permnnz = sum{dz.jc[j+1] | ordered[j]==1} -- exactly dpr1fact.c's
    # mexFunction; pivperm[:permnnz] is meaningful, the rest is scratch.
    permnnz = 0
    for i in range(n):
        if ordered[i]:
            permnnz += int(dz_jc[i + 1])

    Lden = {
        "betajc": (betajc[: n + 1].astype(np.int64) + 1),  # 1-indexed, .m-facing
        "beta": beta[: int(betajc[n])].copy(),
        "p": p,
        "dopiv": ordered[:n].copy(),
        "pivperm": pivperm[:permnnz].copy(),  # opaque, see _dpr1_apply()
    }
    return Lden, lab


def cone_from_dict(K: dict) -> ConeK:
    """Build a ConeK from a plain dict shaped like SeDuMi's K struct, e.g.
    {"f": 2, "l": 3, "q": [4], "s": [2, 3]}. Mirrors what conepars() does
    for the MATLAB/Octave K struct, with no mxArray/MATLAB/Octave layer
    anywhere in the path.
    """
    import numpy as np

    raw = SedumiKRaw()
    raw.f = float(K.get("f", 0.0))
    raw.l = float(K.get("l", 0.0))

    # Keep the backing numpy arrays alive for the duration of the call by
    # returning them alongside; conepars_raw() only reads them, so this
    # scope is enough (it doesn't retain the pointers past the call).
    q = np.ascontiguousarray(K.get("q", []), dtype=np.float64)
    r = np.ascontiguousarray(K.get("r", []), dtype=np.float64)
    s = np.ascontiguousarray(K.get("s", []), dtype=np.float64)
    raw.q = q.ctypes.data_as(c_double_p) if q.size else None
    raw.qN = q.size
    raw.r = r.ctypes.data_as(c_double_p) if r.size else None
    raw.rN = r.size
    raw.s = s.ctypes.data_as(c_double_p) if s.size else None
    raw.sN = s.size

    if "rsdpN" in K:
        raw.rsdpNgiven = b"\x01"
        raw.rsdpN = float(K["rsdpN"])
    else:
        raw.rsdpNgiven = b"\x00"

    raw.statsGiven = b"\x00"  # always recompute; the precomputed-stats
    # path exists to mirror the MEX adapter, not needed from Python

    cone = ConeK()
    _lib.conepars_raw(ctypes.byref(raw), ctypes.byref(cone))
    # cone.{lorNL,rconeNL,sdpNL} are just copies of raw.{q,r,s}, i.e.
    # pointers into q/r/s above -- keep them alive as long as `cone` is,
    # or they'd dangle the moment this function returns and q/r/s get
    # garbage collected.
    cone._keepalive = (q, r, s)
    return cone
