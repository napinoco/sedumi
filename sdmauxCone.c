/* ************************************************************
   MODULE sdmaux*.c  -- Several low-level subroutines for the
   mex-files in the Self-Dual-Minimization package.

% This file is part of SeDuMi 1.1 by Imre Polik and Oleksandr Romanko
% Copyright (C) 2005 McMaster University, Hamilton, CANADA  (since 1.1)
%
% Copyright (C) 2001 Jos F. Sturm (up to 1.05R5)
%   Dept. Econometrics & O.R., Tilburg University, the Netherlands.
%   Supported by the Netherlands Organization for Scientific Research (NWO).
%
% Affiliation SeDuMi 1.03 and 1.04Beta (2000):
%   Dept. Quantitative Economics, Maastricht University, the Netherlands.
%
% Affiliations up to SeDuMi 1.02 (AUG1998):
%   CRL, McMaster University, Canada.
%   Supported by the Netherlands Organization for Scientific Research (NWO).
%
% This program is free software; you can redistribute it and/or modify
% it under the terms of the GNU General Public License as published by
% the Free Software Foundation; either version 2 of the License, or
% (at your option) any later version.
%
% This program is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
% GNU General Public License for more details.
%
% You should have received a copy of the GNU General Public License
% along with this program; if not, write to the Free Software
% Foundation, Inc.,  51 Franklin Street, Fifth Floor, Boston, MA
% 02110-1301, USA

 ************************************************************ */

#include <string.h>
#include "blksdp.h"
/* ============================================================
   CONE-K STATISTICS
   ============================================================ */
/* ************************************************************
   PROCEDURE conepars_raw - Read cone K parameters from a plain-C
     sedumiKRaw struct. This is the MATLAB/Octave-free core: it never
     touches an mxArray, so it can be called directly by a standalone
     (non-MEX) build.
   INPUT
     rawK  -  plain-C mirror of the "K" struct's fields (see blksdp.h).
   OUTPUT
     *pK - struct where cone K parameters get stored.
   ************************************************************ */
void conepars_raw(const sedumiKRaw *rawK, coneK *pK)
{
 mwIndex idummy, nblk;

 pK->frN = (mwSize) rawK->f;
 pK->lpN = (mwSize) rawK->l;

 pK->lorN = rawK->qN;                              /* K.q */
 pK->lorNL = rawK->q;
 if(pK->lorN == 1 && pK->lorNL[0] == 0.0)           /* K.q=0 -> lorN = 0 */
   pK->lorN = 0;

 pK->rconeN = rawK->rN;                            /* K.r */
 pK->rconeNL = rawK->r;
 if(pK->rconeN == 1 && pK->rconeNL[0] == 0.0)       /* K.r=0 -> rconeN = 0 */
   pK->rconeN = 0;

 pK->sdpN = rawK->sN;                              /* K.s */
 pK->sdpNL = rawK->s;
 if(pK->sdpN == 1 && pK->sdpNL[0] == 0.0)           /* K.s=0 -> sdpN = 0 */
   pK->sdpN = 0;

 if(!rawK->rsdpNgiven)                             /* K.rsdpN */
   pK->rsdpN = pK->sdpN;                           /* default to all real */
 else
   pK->rsdpN = (mwSize) rawK->rsdpN;
 SEDUMI_ASSERT(pK->rsdpN <= pK->sdpN, "K.rsdpN mismatches K.s");

 /* --------------------------------------------------
    GET STATISTICS: try to read from K, otherwise compute them.
    -------------------------------------------------- */
 if(rawK->statsGiven){
   const double *blkstartPr;
   pK->rLen = (mwSize) rawK->rLen;
   pK->hLen = (mwSize) rawK->hLen;
   pK->qMaxn = (mwSize) rawK->qMaxn;
   pK->rMaxn = (mwSize) rawK->rMaxn;
   pK->hMaxn = (mwSize) rawK->hMaxn;
   nblk = 1 + pK->lorN + pK->sdpN;
   SEDUMI_ASSERT(rawK->blkstartN == nblk + 1, "Size mismatch K.blkstart.");
   blkstartPr = rawK->blkstart;
   pK->qDim = (mwSize) blkstartPr[pK->lorN+1] - (mwSize) blkstartPr[0];
   blkstartPr += pK->lorN+1;
   pK->rDim = (mwSize) blkstartPr[pK->rsdpN] - (mwSize) blkstartPr[0];
   pK->hDim = (mwSize) blkstartPr[pK->sdpN] - (mwSize) blkstartPr[pK->rsdpN];
 } else {
   someStats(&(pK->qMaxn), &(pK->qDim), &idummy, pK->lorNL, pK->lorN);
   someStats(&(pK->rMaxn), &(pK->rLen), &(pK->rDim), pK->sdpNL, pK->rsdpN);
   someStats(&(pK->hMaxn), &(pK->hLen), &(pK->hDim), pK->sdpNL+pK->rsdpN,
	     (pK->sdpN) - (pK->rsdpN));
   pK->hDim *= 2;
 }
}

#ifndef SEDUMI_STANDALONE
/* ************************************************************
   PROCEDURE conepars - Read cone K parameters from the MATLAB/Octave
     K-structure. Thin mxArray adapter around conepars_raw(); kept so
     every existing call site (conepars(K_IN, &cK) in the mexFunctions)
     needs no change.
   INPUT
     mxK  -  the Matlab structure "K", as passes as input argument "K_IN".
   OUTPUT
     *pK - struct where cone K parameters get stored.
   ************************************************************ */
void conepars(const mxArray *mxK, coneK *pK)
{
 const mxArray *K_FIELD;
 sedumiKRaw rawK;
 memset(&rawK, 0, sizeof(rawK));

 mxAssert(mxIsStruct(mxK), "Parameter `K' should be a structure.");
 if( (K_FIELD = mxGetField(mxK,(mwIndex)0,"f")) != NULL)      /* K.f */
   rawK.f = mxGetScalar(K_FIELD);
 if( (K_FIELD = mxGetField(mxK,(mwIndex)0,"l")) != NULL)      /* K.l */
   rawK.l = mxGetScalar(K_FIELD);
 if( (K_FIELD = mxGetField(mxK,(mwIndex)0,"q")) != NULL){     /* K.q */
   rawK.qN = mxGetM(K_FIELD) * mxGetN(K_FIELD);
   rawK.q = mxGetPr(K_FIELD);
 }
 if( (K_FIELD = mxGetField(mxK,(mwIndex)0,"r")) != NULL){     /* K.r */
   rawK.rN = mxGetM(K_FIELD) * mxGetN(K_FIELD);
   rawK.r = mxGetPr(K_FIELD);
 }
 if( (K_FIELD = mxGetField(mxK,(mwIndex)0,"s")) != NULL){     /* K.s */
   rawK.sN = mxGetM(K_FIELD) * mxGetN(K_FIELD);
   rawK.s = mxGetPr(K_FIELD);
 }
 if( (K_FIELD = mxGetField(mxK,(mwIndex)0,"rsdpN")) != NULL){ /* K.rsdpN */
   rawK.rsdpNgiven = 1;
   rawK.rsdpN = mxGetScalar(K_FIELD);
 }

 rawK.statsGiven = 1;
 if( (K_FIELD = mxGetField(mxK,(mwIndex)0,"rLen")) != NULL){      /* K.rLen */
   rawK.rLen = mxGetScalar(K_FIELD);
   if( (K_FIELD = mxGetField(mxK,(mwIndex)0,"hLen")) != NULL){      /* K.hLen */
     rawK.hLen = mxGetScalar(K_FIELD);
     if( (K_FIELD = mxGetField(mxK,(mwIndex)0,"qMaxn")) != NULL){      /* K.qMaxn */
       rawK.qMaxn = mxGetScalar(K_FIELD);
       if( (K_FIELD = mxGetField(mxK,(mwIndex)0,"rMaxn")) != NULL){      /* K.rMaxn */
	 rawK.rMaxn = mxGetScalar(K_FIELD);
	 if( (K_FIELD = mxGetField(mxK,(mwIndex)0,"hMaxn")) != NULL){    /* K.hMaxn */
	   rawK.hMaxn = mxGetScalar(K_FIELD);
	   if( (K_FIELD = mxGetField(mxK,(mwIndex)0,"blkstart"))!=NULL){ /*K.blkstart*/
	     mxAssert(!mxIsSparse(K_FIELD), "K.blkstart must be a full vector.");
	     rawK.blkstart = mxGetPr(K_FIELD);
	     rawK.blkstartN = mxGetM(K_FIELD) * mxGetN(K_FIELD);
	     goto gotthem;
	   }
	 }
       }
     }
   }
 }
 rawK.statsGiven = 0;
gotthem:

 conepars_raw(&rawK, pK);
}
#endif /* !SEDUMI_STANDALONE */

/* ************************************************************
   PROCEDURE someStats  --  Computes maximum, sum and sum of squares
   INPUT
   x, n - length n vector
   OUTPUT
   xmax, xsum, xssqr - Maximum, sum total and sum of squares
   IMPORTANT: this routine is especially designed for use with the
    blk.s structure, which contains nonneg integers stored as doubles.
   ************************************************************ */
void someStats(mwIndex *pxmax, mwIndex *pxsum, mwIndex *pxssqr,
	       const double *x, const mwIndex n)
{
 mwIndex xi, xmax, xsum, xssqr;
 mwIndex i;

 xmax = 0;             /* assume that all integers are nonnegative */
 xsum = 0; xssqr = 0;
 for(i = 0; i < n; i++){
   xi = (mwIndex) x[i];
   xmax = MAX(xmax, xi);
   xsum += xi;
   xssqr += SQR(xi);
 }
 *pxmax = xmax;
 *pxsum = xsum;
 *pxssqr = xssqr;
}
