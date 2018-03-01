/* File altered for inclusion in cephes module for Python:
 * Main loop commented out.... */
/*  Travis Oliphant Nov. 1998 */

/* Re Kolmogorov statistics, here is Birnbaum and Tingey's (actually it was already present
 * in Smirnov's paper) formula for the
 * distribution of D+, the maximum of all positive deviations between a
 * theoretical distribution function P(x) and an empirical one Sn(x)
 * from n samples.
 *
 *     +
 *    D  =         sup     [P(x) - S (x)]
 *     n     -inf < x < inf         n
 *
 *
 *                  [n(1-d)]
 *        +            -                    v-1              n-v
 *    Pr{D   > d} =    >    C    d (d + v/n)    (1 - d - v/n)
 *        n            -   n v
 *                    v=0
 *
 * (also equals the following sum, but note the terms may be large and alternating in sign)
 * See Smirnov 1944, Dwass 1959
 *                         n
 *                         -                         v-1              n-v
 *                =  1 -   >         C    d (d + v/n)    (1 - d - v/n)
 *                         -        n v
 *                       v=[n(1-d)]+1
 *
 * [n(1-d)] is the largest integer not exceeding n(1-d).
 * nCv is the number of combinations of n things taken v at a time.

 * The algorithms used below for the Smirnov distibution appear in
 * "Computing the Cumulative Distribution Function and Quantiles of the One-sided Kolmogorov-Smirnov Statistic"
 * http://arxiv.org/abs/1802.06966

 */

#include "mconf.h"
#include <float.h>
#include <math.h>


/* ************************************************************************ */
/* Algorithm Configuration */

/* Kolmogorov Two-sided */
/* Switchover between the two series to compute K(x) */
#define KOLMOG_CUTOVER 0.82


/* Smirnov One-sided */
/* N larger than this will result in an approximation */
const int SMIRNOV_MAX_COMPUTE_N = 1000000;

/* Use the upper sum formula, if the number of terms is at most SM_UPPER_MAX_TERMS,
 * and n is at least SM_UPPERSUM_MIN_N
 * Don't use the upper sum if lots of terms are involved as the series alternates
 *  sign and the terms get much bigger than 1.
 */
#define SM_UPPER_MAX_TERMS 3
#define SM_UPPERSUM_MIN_N 10

/* ************************************************************************ */
/* ************************************************************************ */

// Assuming LOW and HIGH are constants.
#define CLIP(X, LOW, HIGH) ((X) < LOW ? LOW : MIN(X, HIGH))
#ifndef MIN
#define MIN(a,b) (((a) < (b)) ? (a) : (b))
#endif
#ifndef MAX
#define MAX(a,b) (((a) < (b)) ? (b) : (a))
#endif

extern double MAXLOG;  // from cephes constants
extern double MINLOG;   // from cephes constants

static const int MIN_EXPABLE = (-708 - 38); // exp() of anything below this returns 0

#ifndef NPY_SQRT2PI
#define NPY_SQRT2PI 2.5066282746310005024157652848110453
#define NPY_LOGSQRT2PI 0.91893853320467274178032973640561764
#endif

typedef struct ThreeProbs {
    double sf;
    double cdf;
    double pdf;
} ThreeProbs;

#define RETURN_3PROBS(PSF, PCDF, PDF) \
    ret.cdf = (PCDF);                 \
    ret.sf = (PSF);                   \
    ret.pdf = (PDF);                  \
    return ret;

static const double _xtol = DBL_EPSILON;
static const double _rtol = 2*DBL_EPSILON;

static int
_within_tol(double x, double y, double atol, double rtol)
{
    double diff = fabs(x-y);
    int result = (diff <=  (atol + rtol * fabs(y)));
    return result;
}

#include "dd_real.h"

// Shorten some of the double-double names for readibility
#define valueD dd_to_double
#define add_dd dd_add_d_d
#define sub_dd dd_sub_d_d
#define mul_dd dd_mul_d_d
#define neg_D  dd_neg
#define div_dd dd_div_d_d
#define add_DD dd_add
#define sub_DD dd_sub
#define mul_DD dd_mul
#define div_DD dd_div
#define add_Dd dd_add_dd_d
#define add_dD dd_add_d_dd
#define sub_Dd dd_sub_dd_d
#define sub_dD dd_sub_d_dd
#define mul_Dd dd_mul_dd_d
#define mul_dD dd_mul_d_dd
#define div_Dd dd_div_dd_d
#define div_dD dd_div_d_dd
#define frexpD dd_frexp
#define ldexpD dd_ldexp
#define logD   dd_log
#define log1pD dd_log1p


/* ************************************************************************ */
/* Kolmogorov : Two-sided                      **************************** */
/* ************************************************************************ */

static ThreeProbs
_kolmogorov(double x)
{
    double P = 1.0;
    double D = 0;
    double sf, cdf, pdf;
    ThreeProbs ret;

    if (npy_isnan(x)) {
        RETURN_3PROBS(NPY_NAN, NPY_NAN, NPY_NAN);
    }
    if (x <= 0) {
        RETURN_3PROBS(1.0, 0.0, 0);
    }
    // x <= 0.040611972203751713
    if (x <= NPY_PI/sqrt(-MIN_EXPABLE * 8)) {
        RETURN_3PROBS(1.0, 0.0, 0);
    }

    P = 1.0;
    if (x <= KOLMOG_CUTOVER) {
        // u = e^(-pi^2/(8x^2))
        // w = sqrt(2pi)/x
        // p = w*u(1+u^8+u^24+u^48...)
        double w = sqrt(2 * NPY_PI)/x;
        double logu8 = -NPY_PI * NPY_PI/(x * x); // log(u^8)
        double u = exp(logu8/8);
        if (u == 0) {
            // P = w*u, but u < 1e-308, and w > 1, so compute as logs, then exponentiate
            double logP = logu8/8 + log(w);
            P = exp(logP);
        } else {
           // Just unroll the loop
            double u8 = exp(logu8);
            double u8cub = pow(u8, 3);
            P = 1 + u8cub * P;
            D = 5*5 + u8cub * D;
            P = 1 + u8*u8 * P;
            D = 3*3 + u8*u8 * D;
            P = 1 + u8 * P;
            D = 1*1 + u8 * D;

            D = NPY_PI * NPY_PI/4/(x*x) * D - P;
            D *=  w * u/x;
            P = w * u * P;
        }
        cdf = P;
        sf = 1-P;
        pdf = D;
    }
    else {
        /* v = e^(-2x^2)
           p = 2 (v - v^4 + v^9...)
             = 2v(1 - v^3*(1 - v^5*(1 - v^7*(1 - ...))) */
        /* Want q^((2k-1)^2)(1-q^(4k-1)) / q(1-q^3) < epsilon */
        double logv = -2*x*x;
        double v = exp(logv);
        // Unroll the loop, k<= 4
        double vsq = v*v;
        double v3 = pow(v, 3);
        double vpwr;

        vpwr = v3*v3*v;   // v**7
        P = 1 - vpwr * P; // 1 - (1-v**(2k-1)) * P
        D = 3*3 - vpwr * D;

        vpwr = v3*vsq;
        P = 1 - vpwr * P;
        D = 2*2 - vpwr * D;

        vpwr = v3;
        P = 1 - vpwr * P;
        D = 1*1 - vpwr * D;

        P = 2 * v * P;
        D = 8 * v * x * D;
        sf = P;
        cdf = 1 - sf;
        pdf = D;
    }
    pdf = MAX(0, pdf);
    cdf = CLIP(cdf, 0, 1);
    sf = CLIP(sf, 0, 1);
    RETURN_3PROBS(sf, cdf, pdf);
}


/* Find x such kolmogorov(x)=psf, kolmogc(x)=pcdf */
static double
_kolmogi(double psf, double pcdf)
{
    double x, t;
    double xmin = 0;
    double xmax = NPY_INFINITY;
    double fmin = 1 - psf;
    double fmax = pcdf - 1;
    int iterations;
    double a = xmin, b = xmax;
    double fa = fmin, fb = fmax;

    if (!(psf >= 0.0 && pcdf >= 0.0 && pcdf <= 1.0 && psf <= 1.0)) {
        mtherr("kolmogi", DOMAIN);
        return (NPY_NAN);
    }
    if (fabs(1.0 - pcdf - psf) >  4* DBL_EPSILON) {
        mtherr("kolmogi", DOMAIN);
        return (NPY_NAN);
    }
    if (pcdf == 0.0) {
        return 0.0;
    }
    if (psf == 0.0) {
        return NPY_INFINITY;
    }

    if (pcdf <= 0.5) {
        /* p ~ (sqrt(2pi)/x) *exp(-pi^2/8x^2).  Generate lower and upper bounds  */
        double logpcdf = log(pcdf);
        const double SQRT2 = NPY_SQRT2;
        const double LOGSQRT2 = NPY_LOGSQRT2PI;
        /* 1 >= x >= sqrt(p) */
        // Iterate twice: x -> pi/(sqrt(8) sqrt(log(sqrt(2pi)) - log(x) - log(pdf)))
        a = NPY_PI / (2 * SQRT2 * sqrt(-(logpcdf + logpcdf/2 - LOGSQRT2)));
        b = NPY_PI / (2 * SQRT2 * sqrt(-(logpcdf + 0 - LOGSQRT2)));
        a = NPY_PI / (2 * SQRT2 * sqrt(-(logpcdf + log(a) - LOGSQRT2)));
        b = NPY_PI / (2 * SQRT2 * sqrt(-(logpcdf + log(b) - LOGSQRT2)));
        x =  (a + b) / 2.0;
    }
    else {
        /* Based on the approximation p ~ 2 exp(-2x^2)
           Found that needed to replace psf with a slightly smaller number in the second element
           as otherwise _kolmogorov(b) came back as a very small number but with
           the same sign as _kolmogorov(a)
           kolmogi(0.5) = 0.82757355518990772
           so (1-q^(-(4-1)*2*x^2)) = (1-exp(-6*0.8275^2) ~ (1-exp(-4.1)*/
        const double jiggerb = 256 * DBL_EPSILON;
        double pba = psf/(1.0 - exp(-4))/2, pbb = psf * (1 - jiggerb)/2;
        double q0;
        a = sqrt(-0.5 * log(pba));
        b = sqrt(-0.5 * log(pbb));
        /* Use inversion of
            p = q - q^4 + q^9 - q^16 + ...:
            q = p + p^4 + 4p^7 - p^9 + 22p^10  - 13p^12 + 140*p^13 ...
         */
        {
            double p = psf/2.0;
            double p2 = p*p;
            double p3 = p*p*p;
            q0 = 1 + p3 * (1 + p3 * (4 + p2 *(-1 + p*(22 + p2* (-13 + 140 * p)))));
            q0 *= p;
        }
        x = sqrt(-log(q0) / 2);
        if (x < a || x > b) {
            x = (a+b)/2;
        }
    }
    assert(a <= b);

    iterations = 0;
    do {
        double x0 = x;
        ThreeProbs probs = _kolmogorov(x0);
        double df = ((pcdf < 0.5) ? (pcdf - probs.cdf) : (probs.sf - psf));
        double dfdx;

        if (fabs(df) == 0) {
            break;
        }
        /* Update the bracketing interval */
        if (df > 0 && x > a) {
            a = x;
            fa = df;
        } else if (df < 0 && x < b) {
            b = x;
            fb = df;
        }

        dfdx = -probs.pdf;
        if (fabs(dfdx) <= 0.0) {
            x = (a+b)/2;
            t = x0 - x;
        } else {
            t = df/dfdx;
            x = x0 - t;
        }

        /* Check out-of-bounds.
        Not expecting this to happen often --- kolmogorov is convex near x=infinity and
        concave near x=0, and we should be approaching from the correct side.
        If out-of-bounds, replace x with a midpoint of the bracket. */
        if (x >= a && x <= b)  {
            if (_within_tol(x, x0, _xtol, _rtol)) {
                break;
            }
            if ((x == a) || (x == b)) {
                x = (a + b) / 2.0;
                /* If the bracket is already so small ... */
                if (x == a || x == b) {
                    break;
                }
            }
        } else {
            x = (a + b) / 2.0;
            if (_within_tol(x, x0, _xtol, _rtol)) {
                break;
            }
        }

        if (++iterations > MAXITER) {
            mtherr("kolmogi", TOOMANY);
            break;
        }
    } while(1);
    return (x);
}


double
kolmogorov(double x)
{
    if (npy_isnan(x)) {
        return NPY_NAN;
    }
    return _kolmogorov(x).sf;
}

double
kolmogc(double x)
{
    if (npy_isnan(x)) {
        return NPY_NAN;
    }
    return _kolmogorov(x).cdf;
}

double
kolmogp(double x)
{
    if (npy_isnan(x)) {
        return NPY_NAN;
    }
    if (x <= 0) {
        return -0.0;
    }
    return -_kolmogorov(x).pdf;
}

/* Functional inverse of Kolmogorov survival statistic for two-sided test.
 * Finds x such that kolmogorov(x) = p.
 */
double
kolmogi(double p)
{
    if (npy_isnan(p)) {
        return NPY_NAN;
    }
    return _kolmogi(p, 1-p);
}

/* Functional inverse of Kolmogorov cumulative statistic for two-sided test.
 * Finds x such that kolmogc(x) = p = (or kolmogorov(x) = 1-p).
 */
double
kolmogci(double p)
{
    if (npy_isnan(p)) {
        return NPY_NAN;
    }
    return _kolmogi(1-p, p);
}



/* ************************************************************************ */
/* ********** Smirnov : One-sided ***************************************** */
/* ************************************************************************ */

static double
nextPowerOf2(double x)
{
    double q = ldexp(x, 1-DBL_MANT_DIG);
    double L = fabs(q+x);
    if (L == 0) {
        L = fabs(x);
    } else {
        int Lint = (int)(L);
        if (Lint == L) {
            L = Lint;
        }
    }
    return L;
}

static double
modnx(int n, double x, int *pnxfloor, double *pnx)
{
    double2 alphaD, nxD, nxfloorD;
    int nxfloor;
    double alpha;

    nxD = mul_dd(n, x);
    nxfloorD = dd_floor(nxD);
    alphaD = sub_DD(nxD, nxfloorD);
    alpha = dd_hi(alphaD);
    nxfloor = dd_to_int(nxfloorD);
    assert(alpha >= 0);
    assert(alpha <= 1);
    if (alpha == 1) {
        nxfloor += 1;
        alpha = 0;
    }
    assert(alpha < 1.0);
    *pnx = dd_to_double(nxD);
    *pnxfloor = nxfloor;
    return alpha;
}

static void
updateBinomial(double2 *Cman, int *Cexpt, int n, int j)
{
    int expt;
    assert (!dd_is_zero(*Cman));
    double2 rat = div_dd(n - j, j + 1.0);
    double2 CD2 = mul_DD(*Cman, rat);
    double2 manD = frexpD(CD2, &expt);
    *Cexpt += expt;
    *Cman = manD;
}


static double2
powsimp_D(double2 a,  int m)
{
    /* Using dd_npwr() here would be quite time-consuming.  Tradeoff accuracy-time by using pow().*/
    double ans;
    if (m <= 0) {
        if (m == 0) {
            return DD_C_ONE;
        }
        return dd_inv(powsimp_D(a, -m));
    }
    if (dd_is_zero(a)) {
        return DD_C_ZERO;
    }
    ans = pow(a.x[0], m);
    double r = a.x[1]/a.x[0];
    double adj = m*r;
    if (fabs(adj) > 1e-8) {
        if (fabs(adj) < 1e-4) {
           // Take 1st two terms of Tayler Series for (1+r)^m
            adj += (m*r) * ((m-1)/2.0 * r);
        } else {
            // Take exp of scaled log
            adj = expm1(m*log1p(r));
        }
    }
    return dd_add_d_d(ans, ans*adj);
}

static double
pow2(double a, double b,  int m)
{
    return dd_to_double(powsimp_D(add_dd(a, b), m));
}

#define _MAX_EXPONENT 960
//# Not 1024 as too big.  Want < 1023-52 so as to keep ansD[1] normalized

#define RETURN_M_E(MAND, EXPT) \
    *pExponent = EXPT;\
    return MAND;


static double2
pow2Scaled_D(double2 a, int m, int *pExponent)
{
    double2 ansD;
    int ansE;
    if (m <= 0)
    {
        int L, L2;
        if (m == 0) {
            RETURN_M_E(DD_C_ONE, 0);
        }
        double2 ans = pow2Scaled_D(a, -m, &L);
        ansD = frexpD(div_dD(1.0, ans), &L2);
        ansE = -L + L2;
        RETURN_M_E(ansD, ansE);
    }
    int xL;
    double2 xD = frexpD(a, &xL);
    if (m == 1) {
        RETURN_M_E(xD, xL);
    }
    // xD ^ maxExpt >= 2^{-1022}
    // =>  maxExpt = 1022 / log2(xD.x[0]) = 708 / log(xD.x[0])
    //            = 708/((1-xD.x[0] + xD.x[0]^2/2 - ...) <= 708/(1-xD.x[0])
    int maxExpt = _MAX_EXPONENT;
    if (m*(xD.x[0]-1) / xD.x[0] < -_MAX_EXPONENT * NPY_LOGE2) {
        double lg2x = log(xD.x[0])/NPY_LOGE2;
        double lgAns = m*lg2x;
        if (lgAns <= -_MAX_EXPONENT) {
            maxExpt = (int)(nextPowerOf2(-_MAX_EXPONENT / lg2x + 1)/2);
        }
    }
    if (m <= maxExpt)
    {
#if 1
        double2 ans1 = powsimp_D(xD, m);
        ansD = frexpD(ans1, &ansE);
        ansE += m * xL;
        RETURN_M_E(ansD, ansE);
#else
        double ans1 = pow2(xD.x[0], xD.x[1], m);
        ans1 = frexp(ans1, &ansE);
        ansE += m * xL;
        RETURN_M_E(add_dd(ans1, 0), ansE);
#endif
    }
    int q = m / maxExpt;
    int r = m % maxExpt;
    int rE, qE, q512E;
    double2 ansr = pow2Scaled_D(xD, r, &rE);       // x^r
    double2 ansq = pow2Scaled_D(xD, maxExpt, &qE); // x^512
    double2 ansq512 = pow2Scaled_D(ansq, q, &q512E);     //  (x^512)^q
    q512E += qE * q;
    ansD = frexpD(mul_DD(ansr, ansq512), &ansE);
    ansE += q512E + rE;
    ansE += m*xL;
    RETURN_M_E(ansD, ansE);
}


static double2
pow4_D(double a, double b, double c, double d, int m)
{
    double2 A, C, X;
    if (m <= 0){
        if (m == 0) {
            return DD_C_ONE;
        }
        return pow4_D(c, d, a, b, -m);
    }
    A = add_dd(a, b);
    C = add_dd(c, d);
    if (dd_is_zero(A)) {
        return (dd_is_zero(C) ? DD_C_NAN : DD_C_ZERO);
    }
    if (dd_is_zero(C)) {
        return (dd_is_negative(A) ? DD_C_NEGINF : DD_C_INF);
    }
    X = div_DD(A, C);
    return powsimp_D(X, m);
}


static double
pow4(double a, double b, double c, double d, int m)
{
    double2 ret = pow4_D(a, b, c, d, m);
    return dd_to_double(ret);
}

static double2
logpow4_D(double a, double b, double c, double d, int m)
{
    double2 ans;
    double2 A, C, X;
    if (m == 0) {
        return DD_C_ZERO;
    }
    A = add_dd(a, b);
    C = add_dd(c, d);
    if (dd_is_zero(A)) {
        return (dd_is_zero(C) ? DD_C_ZERO : DD_C_NEGINF);
    }
    if (dd_is_zero(C)) {
        return DD_C_INF;
    }
    X = div_DD(A, C);
    assert(X.x[0] >= 0);
    if (0.5 <= X.x[0] && X.x[0] <= 1.5) {
        double2 A1 = sub_DD(A, C);
        double2 X1 = div_DD(A1, C);
        ans = log1pD(X1);
    } else {
        ans = logD(X);
    }
    ans = mul_Dd(ans, m);
    return ans;
}

static double
logpow4(double a, double b, double c, double d, int m)
{
    double2 ans = logpow4_D(a, b, c, d, m);
    return dd_to_double(ans);
}

/* Compute a single term in the summation  */
static void
computeAjnx(int n, double x, int v, double2 Cman, int Cexpt,
            double2 *pt1, double2 *pt2, double2 *pAi)
{
    int L1, L2, Lans;
    double2 Ai;
    double2 t2x = sub_Dd(div_dd(n - v, n), x);  //  1 - x - v/n
    double2 t2 = pow2Scaled_D(t2x, n-v, &L2);
    double2 t1x = add_Dd(div_dd(v, n), x);   // x + v/n
    double2 t1 = pow2Scaled_D(t1x, v-1, &L1);
    double2 ans = mul_DD(t1, t2);
    ans = mul_DD(ans, Cman);
    Lans = Cexpt + L1 + L2;
    Ai = ldexpD(ans, Lans);
    *pAi = Ai;
    *pt1 = t1;
    *pt2 = t2;
}

static ThreeProbs
_smirnov_all_sd(int n, double x)
{
    double nx, alpha;
    double2 AjSum = DD_C_ZERO;
    double2 dAjSum = DD_C_ZERO;
    double cdf, sf, pdf;

    int bUseUpperSum;
    int nxfl, nxceil, n1mxfl, n1mxceil;
    ThreeProbs ret;

    if (!(n > 0 && x >= 0.0 && x <= 1.0)) {
        RETURN_3PROBS(NPY_NAN, NPY_NAN, NPY_NAN);
    }
    if (n == 1) {
        RETURN_3PROBS(1-x, x, 1.0);
    }
    if (x == 0.0) {
        RETURN_3PROBS(1.0, 0.0, 1.0);
    }
    if (x == 1.0) {
        RETURN_3PROBS(0.0, 1.0, 0.0);
    }

    alpha = modnx(n, x, &nxfl, &nx);
    nxceil = nxfl + (alpha == 0 ? 0: 1);
    n1mxfl = n - nxfl - (alpha == 0 ? 0 : 1);
    n1mxceil = n - nxfl;
    // If alpha is 0, don't actually want to include the last term
    // in either the lower or upper summations
    if (alpha == 0) {
        n1mxfl -= 1;
        n1mxceil += 1;
    }

    if (nxfl == 0 || (nxfl == 1 && alpha == 0)) {
        double t = pow2(1, x, n-1);
        pdf = (nx + 1) * t / (1+x);
        cdf = x * t;
        sf = 1 - cdf;
        if (nxfl == 1) {  // Adjust if x=1/n *exactly*
            assert(alpha == 0);
            pdf -= 0.5;
        }
        RETURN_3PROBS(sf, cdf, pdf);
    }
    if (-2 * n * x*x < MINLOG) {
        RETURN_3PROBS(0, 1, 0);
    }
    if (nxfl >= n-1) {
        sf = pow2(1, -x, n);
        cdf = 1 - sf;
        pdf = n * sf/(1-x);
        RETURN_3PROBS(sf, cdf, pdf);
    }
    if (n > SMIRNOV_MAX_COMPUTE_N) {
        // p ~ e^(-(6nx+1)^2 / 18n)
        double logp = -pow(6*n*x+1.0, 2)/18.0/n;
        sf = exp(logp);
        cdf = 1 - sf;
        pdf = (6 * nx + 1) * 2 * sf/3;
        RETURN_3PROBS(sf, cdf, pdf);
    }
    {
        // Use the upper sum if n is large enough, and x is small enough and
        // the number of terms is going to be small enough.
        // Otherwise it just drops accuracy, about 1.6bits * nUpperTerms
        int nUpperTerms = n - n1mxceil + 1;
        bUseUpperSum = (nUpperTerms <= 1 && x < 0.5);
        bUseUpperSum = (bUseUpperSum ||
                        ((n >= SM_UPPERSUM_MIN_N)
                          && (nUpperTerms <= SM_UPPER_MAX_TERMS)
                          && (x <= 0.5 / sqrt(n))));
    }

    {
        int start=0, step=1, nTerms=n1mxfl+1;
        int j, firstJ = 0;
        int vmid = n/2;
        double2 Cman = DD_C_ONE;
        int Cexpt = 0;
        double2 Aj, dAj, t1, t2, dAjCoeff;
        double2 oneOverX = div_dd(1, x);

        if (bUseUpperSum) {
            start = n;
            step = -1;
            nTerms = n - n1mxceil + 1;

            t1 = pow4_D(1, x, 1, 0, n - 1);
            t2 = DD_C_ONE;
            Aj = t1;

            dAjCoeff = div_dD(n - 1, add_dd(1, x));
            dAjCoeff = add_DD(dAjCoeff, oneOverX);
        } else {
            t1 = oneOverX;
            t2 = pow4_D(1, -x, 1, 0, n);
            Aj = div_Dd(t2, x);

            dAjCoeff = div_DD(sub_dD(-1, mul_dd(n - 1, x)), sub_dd(1, x));
            dAjCoeff = div_Dd(dAjCoeff, x);
            dAjCoeff = add_DD(dAjCoeff, oneOverX);
        }

        dAj = mul_DD(Aj, dAjCoeff);
        AjSum = add_DD(AjSum, Aj);
        dAjSum = add_DD(dAjSum, dAj);

        updateBinomial(&Cman, &Cexpt, n, 0);
        firstJ ++;

        for (j = firstJ; j < nTerms; j += 1) {
            int v = start + j * step;

            computeAjnx(n, x, v, Cman, Cexpt, &t1, &t2, &Aj);

            if (dd_isfinite(Aj) && !dd_is_zero(Aj)) {
                dAjCoeff = sub_DD(div_dD((n * (v - 1)), add_dd(nxfl + v, alpha)),
                            div_dD(((n - v) * n), sub_dd(n - nxfl - v, alpha)));
                dAjCoeff = add_DD(dAjCoeff, oneOverX);
                dAj = mul_DD(Aj, dAjCoeff);

                assert(dd_isfinite(Aj));
                AjSum = add_DD(AjSum, Aj);
                dAjSum = add_DD(dAjSum, dAj);
            }
            if (!dd_is_zero(Aj)) {
               if ((4*(nTerms-j) * fabs(dd_to_double(Aj)) < DBL_EPSILON * dd_to_double(AjSum))
                     && (j != nTerms - 1)) {
                   break;
                }
            }
            else if (j > vmid) {
                assert(dd_is_zero(Aj));
                break;
            }

            updateBinomial(&Cman, &Cexpt, n, j);
        }
        assert(dd_isfinite(AjSum));
        assert(dd_isfinite(dAjSum));
        {
            double2 derivD = mul_dD(x, dAjSum);
            double2 probD = mul_Dd(AjSum, x);
            double deriv = dd_to_double(derivD);
            double prob = dd_to_double(probD);

            assert (nx != 1 || alpha > 0);
            if (step < 0) {
                cdf = prob;
                sf = 1-prob;
                pdf = deriv;
            } else {
                cdf = 1-prob;
                sf = prob;
                pdf = -deriv;
            }
        }
    }

    pdf = MAX(0, pdf);
    cdf = CLIP(cdf, 0, 1);
    sf = CLIP(sf, 0, 1);
    RETURN_3PROBS(sf, cdf, pdf);
}

/* Functional inverse of Smirnov distribution
 * finds x such that smirnov(n, x) = psf; smirnovc(n, x) = pcdf).  */
static double
_smirnovi(int n, double psf, double pcdf)
{
    /* Need to use a bracketing NR algorithm here and be very careful about the starting point */
    double x, logpcdf;
    int iterations = 0;
    int function_calls = 0;
    double a=0, b=1;
    double fa=pcdf, fb=-psf;
    double maxlogpcdf, psfrootn;
    double dx, dxold;

    if (!(n > 0 && psf >= 0.0 && pcdf >= 0.0 && pcdf <= 1.0 && psf <= 1.0)) {
        mtherr("smirnovi", DOMAIN);
        return (NPY_NAN);
    }
    if (fabs(1.0 - pcdf - psf) >  4* DBL_EPSILON) {
        mtherr("smirnovi", DOMAIN);
        return (NPY_NAN);
    }
    // STEP 1: Handle psf, or pcdf == 0
    if (pcdf == 0.0) {
        return 0.0;
    }
    if (psf == 0.0) {
        return 1.0;
    }
    // STEP 2: Handle n=1
    if (n == 1) {
        return pcdf;
    }

    // STEP 3 Handle x close to 1
    /* Handle psf *very* close to 0.  Correspond to (n-1)/n < x < 1  */
    psfrootn = pow(psf, 1.0 / n);
    if (n < 150 && n*psfrootn <= 1) { // xmin > 1 - 1.0 / n
        // Solve exactly.
        x = 1  - psfrootn;
        return x;
    }

    logpcdf = (pcdf < 0.5 ? log(pcdf) : log1p(-psf));

    // STEP 4 Find bracket and initial estimate for use in N-R
    // 4(a)  x close to 1
    /* Handle 0 < x <= 1/n:   pcdf = x * (1+x)^*(n-1)  */
    maxlogpcdf = logpow4(1, 0.0, n, 0, 1) + logpow4(n, 1, n, 0, n - 1);
    if (logpcdf <= maxlogpcdf) {
        double xmin = pcdf / NPY_El;
        double xmax = pcdf;
        double P1 = pow4(n, 1, n, 0, n - 1) / n;
        double R = pcdf/P1;
        double z0 = R;
        /*
         Do one iteration of N-R solving: z*e^(z-1) = R, with z0=pcdf/P1
         z <- z - (z exp(z-1) - pcdf)/((z+1)exp(z-1))
         If z_0 = R, z_1 = R(1-exp(1-R))/(R+1)
        */
        if (R >= 1) {
            // R=1 is OK; R>1 can happen due to truncation error for x = (1-1/n)+-epsilon
            R = 1;
            x = R/n;
            return x;
        }
        z0 = (z0*z0 + R * exp(1-z0))/(1+z0);
        x = z0/n;
        a = xmin*(1 - 4 * DBL_EPSILON);
        a = MAX(a, 0);
        b = xmax * (1 + 4 * DBL_EPSILON);
        b = MIN(b, 1.0/n);
        x = CLIP(x, a, b);
    }
    else
    {
        // 4(b) : 1/n < x < (n-1)/n
        double xmin = 1  - psfrootn;
        double logpsf = (psf < 0.5 ? log(psf) : log1p(-pcdf));
        double xmax = sqrt(-logpsf / (2.0L * n));
        double xmax6 = xmax - 1.0L / (6 * n);
        a = xmin;
        b = xmax;
        /* Allow for a little rounding error */
        a *= 1 - 4 * DBL_EPSILON;
        b *= 1 + 4 * DBL_EPSILON;
        a = MAX(xmin, 1.0/n);
        b = MIN(xmax, 1-1.0/n);
        x = xmax6;
    }
    if (x < a || x > b) {
        x = (a + b)/2;
    }

    assert (x < 1);

    fa = pcdf;
    fb = -psf; // fa, fb have the correct sign but are really f(0.0), f(1.0), not f(a), f(b)

    // STEP 5 Run N-R
    dxold = b - a;
    dx = dxold;

    /* smirnov should be well-enough behaved for NR starting at this location */
    /* Use smirnov(n, x)-psf, or pcdf - smirnovc(n, x), whichever has smaller p */
    do {
        double dfdx, x0 = x, deltax, df;
        assert(x < 1);
        assert(x > 0);
        {
            ThreeProbs probs =  _smirnov_all_sd(n, x0);
            ++function_calls;
            df = ((pcdf < 0.5) ? pcdf - probs.cdf : probs.sf - psf);
            dfdx = -probs.pdf;
        }
        if (df == 0) {
            return x;
        }
        /* Update the bracketing interval */
        if (df > 0 &&  x > a) {
            a = x;
            fa = df;
        } else if (df < 0 && x < b) {
            b = x;
            fb = df;
        }

        if (dfdx == 0) {
            /* x was not within tolerance, but now we hit a 0 derivative.
            This implies that x >> 1/sqrt(n), and even then |smirnovp| >= |smirnov|
            so this condition is unexpected. */
            /* This would be a problem for pure N-R, but we are bracketed,
            so can just do a bisection step. */
            x = (a+b)/2;
            deltax = x0 - x;
        }  else {
            deltax = df / dfdx;
            x = x0 - deltax;
        }
        /* Check out-of-bounds.
           Not expecting this to happen ofen --- smirnov is convex near x=1 and
           concave near x=0, and we should be approaching from the correct side.
           If out-of-bounds, replace x with a midpoint of the brck.
           Also check fast enough convergence.
            */
        if ((a <= x) && (x <= b) && (fabs(2 * deltax) <= fabs(dxold) || fabs(dxold) < 256 * DBL_EPSILON)) {
            dxold = dx;
            dx = deltax;
        } else {
            dxold = dx;
            dx = dx / 2;
            x = (a + b) / 2;
            deltax = x0 - x;
        }
        /* Note that if psf is close to 1, f(x) -> 1, f'(x) -> -1.
           => abs difference |x-x0| is approx |f(x)-p| >= DBL_EPSILON,
           => |x-x0|/x >= DBL_EPSILON/x.
           => cannot use a purely relative criteria as it will fail for x close to 0.
        */
        if (_within_tol(x, x0, (psf < 0.5 ? 0 : _xtol), _rtol)) {
            break;
        }
        if (++iterations > MAXITER) {
            mtherr("smirnovi", TOOMANY);
            return (x);
        }
    } while (1);
    return x;
}


double
smirnov(int n, double x)
{
    if (npy_isnan(x)) {
        return NPY_NAN;
    }
    ThreeProbs probs =  _smirnov_all_sd(n, x);
    return probs.sf;
}

double
smirnovc(int n, double x)
{
    if (npy_isnan(x)) {
        return NPY_NAN;
    }
    ThreeProbs probs =  _smirnov_all_sd(n, x);
    return probs.cdf;
}


/* Derivative of smirnov(n, x)
   One point of discontinuity at x=1/n
*/
double
smirnovp(int n, double x)
{
    /* This comparison should assure returning NaN whenever
     * x is NaN itself.  */
    if (!(n > 0 && x >= 0.0 && x <= 1.0)) {
        return (NPY_NAN);
    }
    if (n == 1) {
         /* Slope is always -1 for n ==1, even at x = 1.0 */
         return -1.0;
    }
    if (x == 1.0) {
        return -0.0;
    }
    /* If x is 0, the derivative is discontinuous, but approaching
       from the right the limit is -1 */
    if (x == 0.0) {
        return -1.0;
    }
    ThreeProbs probs =  _smirnov_all_sd(n, x);
    return -probs.pdf;
}


double
smirnovi(int n, double p)
{
    if (npy_isnan(p)) {
        return NPY_NAN;
    }
    return _smirnovi(n, p, 1-p);
}

double
smirnovci(int n, double p)
{
    if (npy_isnan(p)) {
        return NPY_NAN;
    }
    return _smirnovi(n, 1-p, p);
}
