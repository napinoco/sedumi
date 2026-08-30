"""Phase 3 integration check: the full documented usage pattern from
blkchol.m --

    [L.L,L.d,L.skip,L.add] = blkchol(symbchol(X),X);
    L.d(find(L.skip)) = inf;
    y = sparbwslv(L, sparfwslv(L,b) ./ L.d);

solves X*y=b -- chained end to end through ordmmd -> symbolic_cholesky ->
numeric_cholesky -> fwsolve -> bwsolve, with no MATLAB/Octave/MEX
anywhere in the path, checked against a plain dense solve. This is the
exact sequence every interior-point iteration in sedumi.m relies on, so
it is the most direct evidence yet that the ported kernels compose
correctly, not just individually.
"""

from pathlib import Path

import numpy as np
import pytest
import scipy.io
import scipy.sparse

sedumi_port = pytest.importorskip("sedumi_port")
from sedumi_port import _native  # noqa: E402

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "blkchol"


def _fixture_names():
    if not FIXTURE_DIR.exists():
        return []
    return sorted(p.stem.replace("_spd", "") for p in FIXTURE_DIR.glob("*_spd.mat"))


def _solve_via_cholesky(X_csc, b):
    perm0 = _native.ordmmd(X_csc)
    sym = _native.symbolic_cholesky(X_csc, perm0)
    fact = _native.numeric_cholesky(sym, X_csc)

    perm = sym["perm"]
    xsuper = sym["xsuper"]  # fwsolve/bwsolve index into y/L with these
    # directly (0-indexed, matching symbolic_cholesky's own convention;
    # see _native.fwsolve's docstring).

    d = fact["d"].copy()
    if fact["skip"].size:
        d[fact["skip"]] = np.inf

    rhs = np.asarray(b, dtype=np.float64)[perm].copy()
    z = _native.fwsolve(fact["L"], xsuper, rhs)
    z = z / d
    y_permuted = _native.bwsolve(fact["L"], xsuper, z)

    y = np.empty_like(y_permuted)
    y[perm] = y_permuted
    return y


@pytest.mark.skipif(not _fixture_names(), reason="blkchol fixtures not generated")
@pytest.mark.parametrize("name", _fixture_names())
def test_full_pipeline_solves_Xy_equals_b(name):
    data = scipy.io.loadmat(FIXTURE_DIR / f"{name}_spd.mat")
    X = data["X"].tocsc()
    n = X.shape[0]

    rng = np.random.default_rng(hash(name) % (2**32))
    b = rng.standard_normal(n)

    y = _solve_via_cholesky(X, b)

    X_dense = X.toarray()
    X_dense = np.tril(X_dense) + np.tril(X_dense, -1).T  # symmetrize, as
    # the fixture stores both triangles but this makes the check
    # independent of any assumption about which one "counts"
    expected_y = np.linalg.solve(X_dense, b)

    np.testing.assert_allclose(y, expected_y, rtol=1e-8, atol=1e-8)
