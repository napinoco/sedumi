/* ************************************************************
   HEADER sedumi_platform.h

   Platform shim that lets the SeDuMi C kernels build either as before
   (MATLAB/Octave MEX files, via mex.h) or as part of a standalone,
   MATLAB/Octave-free build (define SEDUMI_STANDALONE).

   This is part of the ongoing MATLAB-free port (see python_port/README.md).
   The MEX build path is untouched -- this file only adds a second,
   parallel path that avoids mex.h entirely.
   ************************************************************ */

#if !defined(SEDUMI_PLATFORM_H)
#define SEDUMI_PLATFORM_H

#if defined(SEDUMI_STANDALONE)

#include <stddef.h>
#include <stdbool.h>
#include <stdlib.h>

/* Plain replacements for the MEX API's index/size types. mwIndex and
   mwSize are both unsigned and the same width on 32-bit-index MATLAB/
   Octave builds; size_t matches that here. */
typedef size_t mwIndex;
typedef size_t mwSize;

/* Reference BLAS / OpenBLAS default to 32-bit Fortran integers (LP64,
   not ILP64) on Linux/macOS/Windows, unlike MATLAB's mwSignedIndex-sized
   blasint. Matches the FORT()-wrapped calls in sdmauxScalarmul.c,
   sdmauxRdot.c and blkchol2.c (dcopy/dscal/daxpy/ddot/idamax). */
typedef int blasint;

#if defined(SEDUMI_BLAS_NO_UNDERSCORE)
#define FORT(x) x
#else
#define FORT(x) x##_
#endif

extern void FORT(dcopy)(const blasint *n, const double *x, const blasint *incx,
                         double *y, const blasint *incy);
extern void FORT(dscal)(const blasint *n, const double *alpha, double *x,
                         const blasint *incx);
extern void FORT(daxpy)(const blasint *n, const double *alpha, const double *x,
                         const blasint *incx, double *y, const blasint *incy);
extern double FORT(ddot)(const blasint *n, const double *x, const blasint *incx,
                          const double *y, const blasint *incy);
extern blasint FORT(idamax)(const blasint *n, const double *x, const blasint *incx);

/* mexErrMsgTxt/mexPrintf/mxAssert equivalents: a standalone build can't pop
   a MATLAB error dialog, so report to stderr and abort instead. Kernels
   ported to be called from Python should prefer returning an error code
   over reaching this, but the legacy call sites (ported largely unchanged
   from the MEX sources) still expect a "this never returns" error path. */
void sedumi_fatal(const char *msg);
#define mexErrMsgTxt(msg) sedumi_fatal(msg)
#define mexWarnMsgTxt(msg) sedumi_warn(msg)
#define mexPrintf printf
#define mxAssert(cond, msg) do { if (!(cond)) sedumi_fatal(msg); } while (0)

void sedumi_warn(const char *msg);

#define SEDUMI_ASSERT(cond, msg) do { if (!(cond)) sedumi_fatal(msg); } while (0)

#else /* !SEDUMI_STANDALONE: original MEX build, unchanged */

#include "mex.h"
#ifdef OCTAVE
#include "f77blas.h"  /* defines "blasint" data type */
#define FORT(x) BLASFUNC(x)
#else /* Matlab */
#include "blas.h"
typedef ptrdiff_t blasint;
/**
 * For Matlab R2019a (probably before) and newer, when including
 * "blas.h" the respective BLAS identifiers are already defined,
 * e.g. for "dcopy":
 *
 *   #define dcopy FORTRAN_WRAPPER(dcopy)
 *
 * thus calling FORT(dcopy) == FORTRAN_WRAPPER(dcopy) inside SeDuMi
 * would result in "dcopy__" and already resulted in some bug reports.
 *
 * Compiling with -DFWRAPPER restores the previous behavior.
 */
#ifdef FWRAPPER
#define FORT(x) FORTRAN_WRAPPER(x)
#else
#define FORT(x) x
#endif
#endif

#define SEDUMI_ASSERT(cond, msg) mxAssert(cond, msg)

#endif /* SEDUMI_STANDALONE */

#endif /* SEDUMI_PLATFORM_H */
