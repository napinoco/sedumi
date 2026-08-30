"""Port of getDAtm.m: computes DAt.q[k,j] = d[k]'*Aj[k] for each Lorentz
block k and constraint column j -- the per-iteration "scaled A" quantity
loopPcg.m/PopK.m/sddir.m need for the Lorentz part, independent of any
dense-column handling.

SCOPE NOTE: this only implements the DAt.q computation, which is needed
for ANY problem with Lorentz cones (not just ones with dense columns).
The DAt.denq correction (dense.q nonempty) requires adendotd -- part of
the "dense columns" subsystem deferred since Phase 2's cluster 4/5 (see
_native.py's own docstrings on getada1/getada2/incorder/iswnbr/
symbfwblk). Rather than silently give a wrong answer, getDAtm() here
raises NotImplementedError if dense.q is nonempty; getdense.m's own
threshold logic means this is empty for the large majority of problems
(dense-column handling is a performance optimization for A matrices
with a handful of unusually dense columns, not a correctness
requirement), so this is a real but narrow gap.

Implemented directly with NumPy/SciPy sparse arithmetic rather than by
wrapping extractA()/ddot()'s C kernels: ddot()'s existing binding
(_native.py, cluster 1) only wraps ddotxj() (the dense-X path);
getDAtm.m's own `ddot(d.q2, A, K.qblkstart, Ablkjc)` call uses the
sparse-X path (spddotxj(), not yet bound) precisely because A is the
full sparse constraint matrix here -- reimplementing the same reduction
via sparse matrix algebra avoids needing that binding at all.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp


def getDAtm(A, dense: dict, d: dict, K: dict) -> dict:
    """DAt = getDAtm(A,dense,d,K): A is the internal-format At (N x m,
    SeDuMi's own storage convention: rows are the N cone variables,
    columns are the m constraints)."""
    lorN = len(K.get("q", []))
    m = A.shape[1]
    if lorN == 0:
        return {"q": np.zeros((0, m)), "denq": np.zeros(0)}

    dense_q = np.asarray(dense.get("q", np.zeros(0, dtype=np.int64))).ravel()
    if dense_q.size:
        raise NotImplementedError(
            "getDAtm: dense.q (dense Lorentz-block) correction (DAt.denq via "
            "adendotd) is not implemented -- see this module's docstring."
        )

    A = A.tocsr() if sp.issparse(A) else sp.csr_matrix(A)
    ix = np.asarray(K["mainblks"], dtype=np.int64).ravel()
    i1, i2 = int(ix[0]), int(ix[1])
    trace_rows = np.arange(i1 - 1, i2 - 1)

    qblkstart = np.asarray(K["qblkstart"], dtype=np.int64).ravel()
    vec_start = int(qblkstart[0]) - 1
    vec_end = int(qblkstart[-1]) - 1

    A_trace = A[trace_rows, :]
    DAt_q = sp.diags(np.asarray(d["q1"]).ravel()) @ A_trace

    if vec_end > vec_start:
        vec_rows = np.arange(vec_start, vec_end)
        block_lens = np.diff(qblkstart)
        block_of = np.repeat(np.arange(lorN), block_lens)
        A_vec = A[vec_rows, :]
        W = sp.csr_matrix(
            (np.asarray(d["q2"]).ravel(), (block_of, np.arange(vec_rows.size))),
            shape=(lorN, vec_rows.size),
        )
        DAt_q = DAt_q + W @ A_vec

    DAt_q_dense = np.asarray(DAt_q.todense())
    return {"q": DAt_q_dense, "denq": np.zeros(0)}
