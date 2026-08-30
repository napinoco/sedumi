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
