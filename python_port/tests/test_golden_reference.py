"""
Phase 0 self-check for the golden reference data.

This does NOT yet test the Python port (it doesn't exist yet) -- it only
confirms that tests/golden/*.mat was generated correctly and is internally
consistent, so later phases (C kernel extraction, Python translation) have
a trustworthy oracle to compare against.

Once the Python port exists (Phase 3+), a companion test module will load
the same golden files and compare them against `sedumi_port`'s own output
(objective value and DIMACS error vector `info["err"]`, within tolerance).
"""

from pathlib import Path

import numpy as np
import pytest
import scipy.io
import scipy.sparse


def _scalar(value):
    """Octave saves 1x1 results of sparse-times-dense products as a sparse
    matrix rather than a plain double; normalize either form to a float.
    Problems with complex data (e.g. quantum.mat) yield a complex c'*x /
    b'*y whose imaginary part is numerical noise -- only the real part is
    the objective value."""
    if scipy.sparse.issparse(value):
        value = value.toarray()
    value = np.asarray(value).squeeze()
    if np.iscomplexobj(value):
        value = value.real
    return float(value)

GOLDEN_DIR = Path(__file__).parent / "golden"

EXPECTED_PROBLEMS = {
    "arch0_golden.mat": -5.665170e-01,
    "control07_golden.mat": -2.062510e01,
    "nb_golden.mat": -5.070309e-02,
    "OH_2Pi_STO-6GN9r12g1T2_golden.mat": 7.946708e01,
    "trto3_golden.mat": -1.279999e04,
    "quantum_golden.mat": -0.75395345,
}

TOL = 1e-6


def _golden_files():
    if not GOLDEN_DIR.exists():
        return []
    return sorted(GOLDEN_DIR.glob("*_golden.mat"))


@pytest.mark.skipif(not _golden_files(), reason="golden reference not generated yet")
@pytest.mark.parametrize("filename", sorted(EXPECTED_PROBLEMS))
def test_golden_matches_known_optimum(filename):
    path = GOLDEN_DIR / filename
    if not path.exists():
        pytest.skip(f"{filename} not generated yet")

    data = scipy.io.loadmat(path)
    expected = EXPECTED_PROBLEMS[filename]

    cx = _scalar(data["cx"])
    by = _scalar(data["by"])

    assert abs(cx - expected) / abs(expected) < TOL
    assert abs(by - expected) / abs(expected) < TOL

    info = data["info"]
    numerr = int(np.asarray(info["numerr"].item()).squeeze())
    pinf = int(np.asarray(info["pinf"].item()).squeeze())
    dinf = int(np.asarray(info["dinf"].item()).squeeze())

    assert numerr in (0, 1), f"solver reported serious numerical error ({numerr})"
    assert pinf == 0
    assert dinf == 0


def test_at_least_one_golden_file_present():
    files = _golden_files()
    if not files:
        pytest.skip(
            "no golden reference generated yet -- run "
            "`octave-cli --no-gui --eval \"cd python_port/tools; generate_golden\"` "
            "from the repository root"
        )
    assert len(files) > 0
