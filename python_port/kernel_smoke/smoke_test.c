/* Phase 1 smoke test: prove sdmauxCone.c's conepars_raw() (and the
   sedumi_platform.h SEDUMI_STANDALONE path in general) compiles and
   runs with zero MEX/MATLAB/Octave dependency -- no mex.h anywhere. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../../blksdp.h"

void sedumi_fatal(const char *msg) {
    fprintf(stderr, "FATAL: %s\n", msg);
    abort();
}
void sedumi_warn(const char *msg) {
    fprintf(stderr, "WARN: %s\n", msg);
}

int main(void) {
    /* Mirrors K.f=2, K.l=3, K.q=[4], K.s=[2,3] (no rsdpN/no stats given,
       exactly what happens for a hand-built problem with no MEX/MATLAB
       struct in the picture at all). */
    double q[1] = {4};
    double s[2] = {2, 3};
    sedumiKRaw raw;
    coneK K;

    memset(&raw, 0, sizeof(raw));
    raw.f = 2; raw.l = 3;
    raw.q = q; raw.qN = 1;
    raw.s = s; raw.sN = 2;

    conepars_raw(&raw, &K);

    printf("frN=%zu lpN=%zu lorN=%zu sdpN=%zu rsdpN=%zu qDim=%zu rDim=%zu hDim=%zu\n",
           (size_t)K.frN, (size_t)K.lpN, (size_t)K.lorN, (size_t)K.sdpN,
           (size_t)K.rsdpN, (size_t)K.qDim, (size_t)K.rDim, (size_t)K.hDim);

    if (K.frN != 2 || K.lpN != 3 || K.lorN != 1 || K.sdpN != 2 || K.rsdpN != 2) {
        fprintf(stderr, "MISMATCH\n");
        return 1;
    }

    /* Also exercise a BLAS-backed kernel (realdot -> FORT(ddot)) to prove
       the standalone BLAS prototypes in sedumi_platform.h link correctly
       against the system BLAS, with no MATLAB/Octave blas.h/f77blas.h. */
    {
        double x[3] = {1.0, 2.0, 3.0};
        double y[3] = {4.0, 5.0, 6.0};
        double d = realdot(x, y, 3);
        printf("realdot = %g (expected 32)\n", d);
        if (d != 32.0) { fprintf(stderr, "BLAS MISMATCH\n"); return 1; }
    }

    printf("OK: standalone build (SEDUMI_STANDALONE) works with no mex.h.\n");
    return 0;
}
