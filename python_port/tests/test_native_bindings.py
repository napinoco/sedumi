"""Phase 2: verify the ctypes bindings over libsedumi.so (built by Phase 1
with -DSEDUMI_STANDALONE, no MATLAB/Octave/MEX in the loop at all) produce
correct results, by comparing against plain NumPy reference computations.
"""

import numpy as np
import pytest

sedumi_port = pytest.importorskip("sedumi_port")
from sedumi_port import _native  # noqa: E402


def test_realdot_matches_numpy():
    rng = np.random.default_rng(0)
    x = rng.standard_normal(37)
    y = rng.standard_normal(37)
    assert _native.realdot(x, y) == pytest.approx(np.dot(x, y))


def test_realssqr_matches_numpy():
    rng = np.random.default_rng(1)
    x = rng.standard_normal(23)
    assert _native.realssqr(x) == pytest.approx(np.dot(x, x))


def test_scalarmul_matches_numpy():
    rng = np.random.default_rng(2)
    x = rng.standard_normal(11)
    out = _native.scalarmul(3.5, x)
    np.testing.assert_allclose(out, 3.5 * x)


def test_addscalarmul_matches_numpy():
    rng = np.random.default_rng(3)
    r = rng.standard_normal(9)
    x = rng.standard_normal(9)
    expected = r + 2.0 * x
    out = _native.addscalarmul(r, 2.0, x)
    np.testing.assert_allclose(out, expected)
    np.testing.assert_allclose(r, expected)  # in place, like the C kernel


def test_cone_from_dict_lp_and_soc():
    # K.f=2 free vars, K.l=3 LP vars, K.q=[4] one 4-dim SOC block.
    cone = _native.cone_from_dict({"f": 2, "l": 3, "q": [4]})
    assert cone.frN == 2
    assert cone.lpN == 3
    assert cone.lorN == 1
    assert cone.sdpN == 0
    assert cone.qDim == 4


def test_cone_from_dict_sdp():
    # K.s=[2, 3]: one 2x2 and one 3x3 PSD block, both real (no rsdpN given
    # -> defaults to all-real, matching conepars()'s MEX behavior).
    cone = _native.cone_from_dict({"s": [2, 3]})
    assert cone.sdpN == 2
    assert cone.rsdpN == 2
    assert cone.rDim == 4 + 9  # svec'd dims of a 2x2 and a 3x3 block


def test_cone_from_dict_defaults_to_empty_cone():
    cone = _native.cone_from_dict({})
    assert cone.frN == 0
    assert cone.lpN == 0
    assert cone.lorN == 0
    assert cone.sdpN == 0
