/*
 * include/double2.h
 *
 * This work was supported by the Director, Office of Science, Division
 * of Mathematical, Information, and Computational Sciences of the
 * U.S. Department of Energy under contract number DE-AC03-76SF00098.
 *
 * Copyright (c) 2000-2007
 *
 * Double-double precision (>= 106-bit significand) floating point
 * arithmetic package based on David Bailey's Fortran-90 double-double
 * package, with some changes. See
 *
 *   http://www.nersc.gov/~dhbailey/mpdist/mpdist.html
 *
 * for the original Fortran-90 version.
 *
 * Overall structure is similar to that of Keith Brigg's C++ double-double
 * package.  See
 *
 *   http://www-epidem.plansci.cam.ac.uk/~kbriggs/doubledouble.html
 *
 * for more details.  In particular, the fix for x86 computers is borrowed
 * from his code.
 *
 * Yozo Hida
 */

#ifndef _DD_REAL_H
#define _DD_REAL_H

#include <float.h>
#include <limits.h>
#include <math.h>

#include "_c99compat.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Some configuration defines */

/* If fast fused multiply-add is available, define to the correct macro for
   using it.  It is invoked as DD_FMA(a, b, c) to compute fl(a * b + c).
   If correctly rounded multiply-add is not available (or if unsure),
   keep it undefined. */
#ifndef DD_FMA
#ifdef FP_FAST_FMA
#define DD_FMA(A, B, C) fma((A), (B), (C))
#endif
#endif

/* Same with fused multiply-subtract */
#ifndef DD_FMS
#ifdef FP_FAST_FMA
#define DD_FMS(A, B, C) fma((A), (B), (-C))
#endif
#endif

/* Define this macro to be the isfinite(x) function. */
#define DD_ISFINITE sc_isfinite

/* Define this macro to be the isinf(x) function. */
#define DD_ISINF sc_isinf

/* Define this macro to be the isnan(x) function. */
#define DD_ISNAN sc_isnan

/* Set the following to 1 to set commonly used function
   to be inlined.  This should be set to 1 unless the compiler
   does not support the "inline" keyword, or if building for
   debugging purposes. */
#if defined(__STDC__) && defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 199901L)
#define DD_TRY_INLINE 1
#endif

// Define one of: DD_INLINE_IS_INLINE, DD_INLINE_IS_STATIC_INLINE and DD_INLINE_IS_EXTERN
#ifdef DD_TRY_INLINE
/* For C11 conformant compilers, declare inline in definition, extern inline in dd_real.c.
  For older MSVC compilers, use inline (C++) or __inline (C).
  Otherwise declare static inline.
 */
 #if defined(__STDC__) && defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L
  #define DD_INLINE_IS_INLINE
  #define DD_INLINE inline
  #define DD_EXTERN_INLINE extern inline
 #else
  #define DD_EXTERN_INLINE extern
  #define DD_INLINE_IS_STATIC_INLINE
  #if defined(_MSC_VER) && !defined(__cplusplus)
   #define DD_INLINE static __inline
  #else
   #define DD_INLINE static inline
  #endif /* _MSC_VER */
 #endif /* __STDC_VERSION__ */
#else /* ! DD_TRY_INLINE */
 #define DD_INLINE_IS_EXTERN
 #define DD_INLINE
 #define DD_EXTERN_INLINE extern
#endif /* DD_TRY_INLINE */

#ifdef __cplusplus
#define DD_STATIC_CAST(T, X) (static_cast<T>(X))
#else
#define DD_STATIC_CAST(T, X) ((T)(X))
#endif

/* double2 struct defintion, some external always-present double2 constants.
*/
typedef struct double2
{
    double x[2];
} double2;

extern const double DD_C_EPS;
extern const double DD_C_MIN_NORMALIZED;
extern const double2 DD_C_MAX;
extern const double2 DD_C_SAFE_MAX;
extern const int DD_C_NDIGITS;

extern const double2 DD_C_2PI;
extern const double2 DD_C_PI;
extern const double2 DD_C_3PI4;
extern const double2 DD_C_PI2;
extern const double2 DD_C_PI4;
extern const double2 DD_C_PI16;
extern const double2 DD_C_E;
extern const double2 DD_C_LOG2;
extern const double2 DD_C_LOG10;
extern const double2 DD_C_ZERO;
extern const double2 DD_C_ONE;

#if defined(__STDC__) && defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 199901L) && defined(NAN)
#define DD_C_NAN_IS_CONST
extern const double2 DD_C_NAN;
extern const double2 DD_C_INF;
extern const double2 DD_C_NEGINF;
#else
#define DD_C_NAN (dd_create(NPY_NAN, NPY_NAN))
#define DD_C_INF (dd_create(NPY_INFINITY, NPY_INFINITY))
#define DD_C_NEGINF (dd_create(-NPY_INFINITY, -NPY_INFINITY))
#endif


/* Include either the inline definitions or declarations of these same functions */

#if defined(DD_INLINE_IS_INLINE) || defined(DD_INLINE_IS_STATIC_INLINE)
#include "dd_real_idefs.h"
#else
#include "dd_real_idecls.h"
#endif  /* DD_INLINE_IS_INLINE ||  DD_INLINE_IS_STATIC_INLINE */

/* Non-inline functions */

/********** Exponentiation **********/
double2 dd_npwr(const double2 a, int n);

/*********** Transcendental Functions ************/
double2 dd_exp(const double2 a);
double2 dd_log(const double2 a);
double2 dd_expm1(const double2 a);
double2 dd_log1p(const double2 a);
double2 dd_log10(const double2 a);
double2 dd_log_d(double a);

/* Returns the exponent of the double precision number.
   Returns INT_MIN is x is zero, and INT_MAX if x is INF or NaN. */
int get_double_expn(double x);

/*********** Random number generator ************/
extern double2 dd_rand(void);


#ifdef __cplusplus
}
#endif


#endif /* _DD_REAL_H */
