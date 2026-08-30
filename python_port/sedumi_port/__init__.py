"""sedumi_port: MATLAB/Octave-free port of SeDuMi.

Phase 2 (this module): ctypes bindings over libsedumi.so, the standalone
C kernel library built in Phase 1 (see ../README.md). Phase 3 will add
the interior-point method itself on top of these bindings.
"""

from . import _native  # noqa: F401
