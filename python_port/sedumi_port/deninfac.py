"""Port of deninfac.m: factors the dense columns into the PCG
preconditioner. SCOPE NOTE: only the "no dense columns" (dense.cols
empty) branch is implemented -- the dense-column branch needs
adenscale/dpr1fact orchestration this port hasn't wired up end to end
yet; raises NotImplementedError rather than silently answering wrong."""

from __future__ import annotations

import numpy as np


def deninfac(symLden, L: dict, dense: dict, DAt: dict, d: dict, absd, qblkstart, pars: dict):
    """[Lden,Ld] = deninfac(symLden,L,dense,DAt,d,absd,qblkstart,pars)"""
    cols = np.asarray(dense.get("cols", np.zeros(0))).ravel()
    if cols.size:
        raise NotImplementedError(
            "deninfac: dense-column factoring (dense.cols nonempty) is not implemented."
        )

    Lden = {"betajc": np.array([0]), "rowperm": np.arange(1, L["d"].size + 1)}
    Ld = L["d"].copy()

    maxuden = pars.get("maxuden", 10)  # noqa: F841 -- unused in the no-dense-cols path, kept for signature fidelity

    skip = np.asarray(L["skip"], dtype=np.int64).ravel()
    if skip.size:
        perm = np.asarray(L["perm"], dtype=np.int64).ravel()
        dtol = pars["canceltol"] * np.asarray(absd)[perm[skip]]
        dtol = np.maximum(dtol, pars["abstol"])
        mask = Ld[skip] <= dtol
        Ld[skip[mask]] = 1.0

    return Lden, Ld
