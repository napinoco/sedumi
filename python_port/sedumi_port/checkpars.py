"""Port of checkpars.m: fills in defaults for missing fields in the
`pars` options dict. Pure dict manipulation, no cone math or C kernels
-- a direct, literal translation."""

from __future__ import annotations


def checkpars(pars: dict | None = None) -> dict:
    """pars = checkpars(pars): fills in defaults for missing fields in
    `pars`. Does not mutate the input dict (returns a new one), unlike
    checkpars.m which works on a copy-on-write MATLAB struct anyway."""
    pars = dict(pars) if pars else {}

    # ---- algorithm selection parameters ----
    if pars.get("alg") not in (0, 1, 2):
        pars["alg"] = 2
    if "beta" not in pars:
        pars["beta"] = 0.5
    elif pars["beta"] > 0.9:
        pars["beta"] = 0.9
    elif pars["beta"] < 0.1:
        pars["beta"] = 0.1
    if "theta" not in pars:
        pars["theta"] = 1 if pars["alg"] == 0 else 0.25
    elif pars["theta"] > 1.0:
        pars["theta"] = 1.0
    elif pars["theta"] < 0.01:
        pars["theta"] = 0.01
    pars.setdefault("stepdif", 2)
    w = pars.get("w")
    if w is None:
        pars["w"] = [1.0, 1.0]
    elif len(w) != 2:
        pars["w"] = [1.0, 1.0]
    else:
        pars["w"] = [max(w[0], 1e-8), max(w[1], 1e-8)]

    # ---- preprocessing ----
    pars.setdefault("free", 1)
    pars.setdefault("sdp", 1)

    # ---- initialization ----
    if pars.get("mu", 0) <= 0:
        pars["mu"] = 1

    # ---- stopping and reporting criteria ----
    pars.setdefault("fid", 1)
    pars.setdefault("eps", 1e-8)
    pars.setdefault("bigeps", 1e-3)
    pars.setdefault("maxiter", 150)

    # ---- debugging and algorithmic/diagnostic analysis ----
    pars.setdefault("vplot", 0)
    pars.setdefault("stopat", -1)
    pars.setdefault("errors", 1)
    pars.setdefault("prep", 1)

    # ---- dense column handling ----
    pars.setdefault("denq", 0.75)
    pars.setdefault("denf", 10)

    # ---- numerical control ----
    pars.setdefault("numtol", 5e-7)
    pars.setdefault("bignumtol", 0.9)
    pars.setdefault("numlvl", 0)

    cholpars = dict(pars.get("chol") or {})
    cholpars.setdefault("skip", 1)
    cholpars.setdefault("abstol", 1e-20)
    cholpars.setdefault("canceltol", 1e-12)
    cholpars.setdefault("maxu", 5e5)
    cholpars.setdefault("maxuden", 5e2)
    pars["chol"] = cholpars

    cgpars = dict(pars.get("cg") or {})
    cgpars.setdefault("qprec", 1)
    cgpars.setdefault("restol", 5e-3)
    cgpars.setdefault("stagtol", 5e-14)
    cgpars.setdefault("maxiter", 49)
    cgpars.setdefault("refine", 1)
    pars["cg"] = cgpars

    return pars
