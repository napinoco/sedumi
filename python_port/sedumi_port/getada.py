"""Port of getada.m: builds ADA = A*P(d)*A' for LP(+Lorentz) problems
directly via dense/sparse linear algebra -- no C kernels, pure
NumPy/SciPy. Used only by optstep.m's LP-optimality-projection path
(sum(K.s)==0), which in real usage is the ONLY path optstep.m ever
takes: sedumi.m only calls optstep.m when `lponly` (K.l==length(c)),
which forces K.q and K.s empty too -- so the Lorentz-block loop below
is exercised faithfully (it's real code in getada.m) but is dead code
in practice, exactly like optstep.m's own getada1/getada2/getada3
branch is for the sum(K.s)!=0 case that can never actually be reached.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp


def getada(A, K: dict, d: dict, DAt: dict):
    """absd = getada(A,K,d,DAt): also returns the freshly built ADA
    matrix (unlike the .m file, which stores it into the global
    ADA_sedumi_ instead of returning it) -- callers here get it back as
    a normal return value: (ADA, absd)."""
    A = A.tocsc() if sp.issparse(A) else sp.csc_matrix(A)
    m = A.shape[1]

    ix3 = int(np.asarray(K["mainblks"]).ravel()[2])
    Alq = A[: ix3 - 1, :]

    lorN = len(K.get("q", []))
    ix2 = int(np.asarray(K["mainblks"]).ravel()[1]) if lorN else ix3
    Kl = int(K["l"])
    scalingvector = np.concatenate([d["l"], -d["det"], np.zeros(ix3 - ix2)])
    if lorN:
        qblkstart = np.asarray(K["qblkstart"], dtype=np.int64).ravel()
        for i in range(lorN):
            lo, hi = int(qblkstart[i]), int(qblkstart[i + 1])
            scalingvector[lo - 1 : hi - 1] = d["det"][i]

    DAt_q = np.asarray(DAt["q"])
    ADA = DAt_q.T @ DAt_q
    ADA = ADA + (Alq.T @ sp.diags(scalingvector) @ Alq).toarray()
    ADA = sp.csc_matrix(ADA)
    absd = np.asarray(ADA.diagonal()).ravel()
    return ADA, absd
