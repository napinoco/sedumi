"""Phase 2: verify the fwsolve/bwsolve ctypes bindings (sparse, unit lower
triangular, supernodal) against a plain SciPy dense triangular solve.

L is stored exactly as SeDuMi's blkchol produces it: unit lower
triangular, with the (unused) diagonal entry explicitly present in the
sparsity pattern as the first entry of each CSC column, matching
scipy.sparse.csc_matrix's indptr/indices/data layout directly. xsuper here
uses the simplest (but valid) case of one column per supernode.
"""

import numpy as np
import pytest
import scipy.linalg
import scipy.sparse

sedumi_port = pytest.importorskip("sedumi_port")
from sedumi_port import _native  # noqa: E402


def _make_unit_lower_triangular_csc(L_dense: np.ndarray):
    """Build a CSC matrix whose sparsity pattern always includes the
    diagonal (even though fwsolve/bwsolve never read its value), matching
    what SeDuMi's own Cholesky factor storage looks like."""
    m = L_dense.shape[0]
    L = L_dense.copy()
    np.fill_diagonal(L, 1.0)  # value is a placeholder; never read
    return scipy.sparse.csc_matrix(np.tril(L))


def _reference_L():
    L = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.5, 1.0, 0.0, 0.0],
            [0.25, -0.5, 1.0, 0.0],
            [0.0, 0.75, 0.1, 1.0],
        ]
    )
    return L


def test_fwsolve_matches_dense_triangular_solve():
    L_dense = _reference_L()
    L_csc = _make_unit_lower_triangular_csc(L_dense)
    xsuper = np.arange(L_dense.shape[0] + 1)  # every column its own supernode

    rng = np.random.default_rng(42)
    b = rng.standard_normal(L_dense.shape[0])

    y = _native.fwsolve(L_csc, xsuper, b.copy())
    expected = scipy.linalg.solve_triangular(
        L_dense, b, lower=True, unit_diagonal=True
    )
    np.testing.assert_allclose(y, expected, atol=1e-12)


def test_bwsolve_matches_dense_triangular_solve():
    L_dense = _reference_L()
    L_csc = _make_unit_lower_triangular_csc(L_dense)
    xsuper = np.arange(L_dense.shape[0] + 1)

    rng = np.random.default_rng(43)
    b = rng.standard_normal(L_dense.shape[0])

    y = _native.bwsolve(L_csc, xsuper, b.copy())
    # bwsolve solves L' * y = b.
    expected = scipy.linalg.solve_triangular(
        L_dense.T, b, lower=False, unit_diagonal=True
    )
    np.testing.assert_allclose(y, expected, atol=1e-12)


def test_fwsolve_then_bwsolve_recovers_original_via_normal_equations():
    # (L L') y = b  <=>  fwsolve then bwsolve, since L is unit lower
    # triangular: this is exactly the pattern SeDuMi uses every iteration
    # to solve with a Cholesky factor.
    L_dense = _reference_L()
    L_csc = _make_unit_lower_triangular_csc(L_dense)
    xsuper = np.arange(L_dense.shape[0] + 1)

    rng = np.random.default_rng(44)
    b = rng.standard_normal(L_dense.shape[0])

    z = _native.fwsolve(L_csc, xsuper, b.copy())
    y = _native.bwsolve(L_csc, xsuper, z.copy())

    LLt = L_dense @ L_dense.T
    expected = np.linalg.solve(LLt, b)
    np.testing.assert_allclose(y, expected, atol=1e-10)
