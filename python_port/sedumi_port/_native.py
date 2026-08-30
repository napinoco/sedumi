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

    return {
        "L": L_csc,
        "perm": perm - 1,
        "xsuper": xsuper[: nsuper_c.value + 1] - 1,
    }


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
    return cone
