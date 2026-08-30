/* ************************************************************
   MODULE sedumi_platform.c

   Default implementations of the error-reporting hooks declared in
   sedumi_platform.h's SEDUMI_STANDALONE path (sedumi_fatal/sedumi_warn),
   used in place of MATLAB/Octave's mexErrMsgTxt/mexWarnMsgTxt.

   Both are __attribute__((weak)): a consumer of libsedumi (e.g. the
   Phase 2 Python bindings) can link its own strong definition -- to
   raise a Python exception instead of aborting the process, for
   instance -- and it will be used instead of the default below with no
   further changes needed on either side.
   ************************************************************ */

#if defined(SEDUMI_STANDALONE)

#include <stdio.h>
#include <stdlib.h>

__attribute__((weak)) void sedumi_fatal(const char *msg)
{
  fprintf(stderr, "sedumi: fatal error: %s\n", msg);
  abort();
}

__attribute__((weak)) void sedumi_warn(const char *msg)
{
  fprintf(stderr, "sedumi: warning: %s\n", msg);
}

#endif /* SEDUMI_STANDALONE */
