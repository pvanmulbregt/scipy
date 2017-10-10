/* File altered for inclusion in cephes module for Python:
 * Main loop commented out.... */
/*  Travis Oliphant Nov. 1998 */

/* Re Kolmogorov statistics, here is Birnbaum and Tingey's formula for the
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
 *                         n
 *                         -                         v-1              n-v
 *                =  1 -   >         C    d (d + v/n)    (1 - d - v/n)
 *                         -        n v
 *                       v=[n(1-d)]+1
 *
 * [n(1-d)] is the largest integer not exceeding n(1-d).
 * nCv is the number of combinations of n things taken v at a time.
 */


#include "mconf.h"
#include "float.h"
#include "math.h"
extern double MAXLOG;

#ifndef MIN
#define MIN(a,b) (((a) < (b)) ? (a) : (b))
#endif

// Assuming LOW and HIGH are constants.  It still evaluates X twice...
#define CLIP(X, LOW, HIGH) ((X) < LOW ? LOW : MIN(X, HIGH))

static const int MIN_EXPABLE = (-708 - 38); // exp() of anything below this returns 0

static const double _xtol = DBL_EPSILON;
static const double _rtol = 2*DBL_EPSILON;

static int _within_tol(double x, double y, double atol, double rtol)
{
    double diff = fabs(x-y);
    int result = (diff <=  (atol + rtol * fabs(y)));
    return result;
}

/* Kolmogorov's limiting distribution of two-sided test, returns
 * probability that sqrt(n) * max deviation > x,
 * or that max deviation > x/sqrt(n).
 * The approximation is useful for the tail of the distribution
 * when n is large.
 * If cdf is non-zero, return 1-p == Pr(max deviation <= x/sqrt(n))
 */

 /* Two series for kolmogorov(x), a Jacobi theta function
  *  sum (-1)^k exp(-2k^2 x^2)   (sum over all integer k); or
  *  sqrt(2pi)/x * sum exp((2k-1)^2pi^2/(8x^2))   (sum over positive integer k)
  *  The second is good for x close to 1, the first for x nearer to 1 (and above)
 */

#define KOLMOG_RTOL (DBL_EPSILON)

#define KOLMOG_CUTOVER 1.0
#define KOLMOG_JIGGER 0.1
#define KOLMOG_EPSILON (DBL_EPSILON / 2)

static double _kolmogorov(double x, int cdf)
{
    double P = 1.0;

    if (npy_isnan(x)) {
        return NPY_NAN;
    }
    if (x <= 0) {
        return (cdf ? 0.0 : 1.0);
    }
    // x <= 0.040611972203751713
    if (x <= NPY_PI/sqrt(-MIN_EXPABLE * 8)) {
        return (cdf ? 0.0 : 1.0);
    }

    P = 1.0;
    if (x <= KOLMOG_CUTOVER) {
        double w = sqrt(2 * NPY_PI)/x;
        double logu8 = -NPY_PI * NPY_PI/(x * x); // log(u^8)
        double u = exp(logu8/8);
        if (u == 0) {
            // P = w*u, but u < 1e-308, and w > 1, so compute as logs, then exponentiate
            double logP = logu8/8 + log(w);
            P = exp(logP);
        } else {
            // Look for u**(2k-1)**2 / u**1 < eps
            double Nmax = 8 * log(KOLMOG_EPSILON) / logu8;
            // Nmax  "=" (2k-1)**2 - 1
            int k = (int)(floor((sqrt(Nmax + 1) + 1)/2 + KOLMOG_JIGGER));
            int r = k - 1;
            assert (r <= 2); // Additional terms needed as x approximately crosses: 0.52, 0.9, 1.29, 1.65
            for(; r>0; r--) {
                P = 1 + exp(logu8 * r) * P;  // (1 + u**(8r) * P)
            }
            P = w * u * P;
        }
        if (!cdf) {
            P = 1 - P;
        }
    }
    else {
        /* Want q^((2k-1)^2)(1-q^(4k-1)) / q(1-q^3) < epsilon */
        double logv = -2*x*x;
        double v = exp(logv);
        double Nmax = log(KOLMOG_EPSILON)/logv;
        double f0 = 2*sqrt(Nmax+1)/3 + 1;
        double delta1 = log(f0)/logv;
        Nmax += delta1 * (1.05);
        // Nmax  "=" k**2 - 1
        int k = (int)(floor(2*sqrt(Nmax + 1)/3 + 1 + KOLMOG_JIGGER));
        assert (k <= 4); // Additional terms neeed as x crosses: 4.2, 1.5, 0.88,...
        // If k > 10, rework as sum with consecutive terms paired.
        for( ; k>1; k--) {
            // P = 1 - pow(v, 2*k-1) * P; // 1 - (1-v**(2r-1)) * P
            P = -expm1(logv * (2*k - 1) + log(P));
        }
        P = 2 * v * P;
        if (cdf) {
            P = 1 - P;
        }
    }
    return CLIP(P, 0, 1);
}


double kolmogorov(double x)
{
    if (npy_isnan(x)) {
        return NPY_NAN;
    }
    return _kolmogorov(x, 0);
}

double kolmogc(double x)
{
    if (npy_isnan(x)) {
        return NPY_NAN;
    }
    return _kolmogorov(x, 1);
}

double kolmogp(double x)
{
    double pp;

    if (npy_isnan(x)) {
        return NPY_NAN;
    }
    if (x <= 0) {
        return -0.0;
    }

    if (x <= KOLMOG_CUTOVER) {
        /* kolmog(x) = sqrt(2pi) * f(x)/x). Use Product Rule to differentiate */
        // double w = sqrt(2 * NPY_PI)/x;
        double logu = -NPY_PI * NPY_PI/(8 * x * x);
        // double u = exp(logu);
        double Nmax = log(KOLMOG_EPSILON) / logu ;
        const double SQRT2PI = sqrt(2*NPY_PI);
        int k = (int)(floor((sqrt(Nmax + 1)+1)/2 + KOLMOG_JIGGER));
        double p0 = 0.0, p1 = 0.0;
        assert (k <= 4);
        for(; k>0; k--) {
            int r = 2*k - 1;
            int r2 = r*r;
            double qn = exp(logu * r2);
            p0 += qn;
            p1 += r2 * qn;
        }
        p0 /= x*x;
        p1 *= pow(NPY_PI, 2);
        p1 /= 4;
        p1 /= pow(x, 4);
        pp = (-p0 + p1)  * SQRT2PI;
        /* pp is derivative of CDF, want derivative of SF */
        pp = -pp;
    }
    else {
        double logv = -2 * x*x;
        //  double v = exp(logv);
        double Nmax = log(KOLMOG_EPSILON)/logv;
        // Nmax  "=" (2r-1)**2 - 1
        int k = (int)(floor((sqrt(Nmax + 1)+1) + KOLMOG_JIGGER));
        assert (k <= 4);
        pp = 0;
        for( ; k>0; k--) {
            int k2 = k*k;
            //  pp = k2 * pow(v, k2) - pp;
            pp = k2 * exp(logv * k2) - pp;
        }
        pp *= (-8*x);
    }
    return pp;
}

/* Find x such kolmogorov(x)=psf, kolmogc(x)=pcdf */
static double _kolmogi(double psf, double pcdf)
{
    double x, t;
    double xmin = 0;
    double xmax = NPY_INFINITY;
    double fmin = 1 - psf;
    double fmax = pcdf - 1;
    int iterations;
    double a = xmin, b = xmax;
    double fa = fmin, fb = fmax;
    const double SQRT2PI = sqrt(2*NPY_PI);

    if (!(psf >= 0.0 && pcdf >= 0.0 && pcdf <= 1.0 && psf <= 1.0)) {
        mtherr("smirnovi", DOMAIN);
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
        const double logSQRT2PI = log(SQRT2PI);
        const double SQRT2 = sqrt(2.0);
        /* 1 >= x >= p.  */
        /* Even better: 1 >= x >= sqrt(p) */
        // Iterate twice: x -> pi/(sqrt(8) sqrt(log(sqrt(2pi)) - log(x) - log(pdf)))
        // a = NPY_PI / (2 * SQRT2 * sqrt(-(logpcdf + logpcdf - logSQRT2PI)));
        /* First time */
        a = NPY_PI / (2 * SQRT2 * sqrt(-(logpcdf + logpcdf/2 - logSQRT2PI)));
        b = NPY_PI / (2 * SQRT2 * sqrt(-(logpcdf + 0 - logSQRT2PI)));
        /* Iterate once more */
        a = NPY_PI / (2 * SQRT2 * sqrt(-(logpcdf + log(a) - logSQRT2PI)));
        b = NPY_PI / (2 * SQRT2 * sqrt(-(logpcdf + log(b) - logSQRT2PI)));
        x =  (a + b) / 2.0;
    }
    else {
        // Based on the approximation p ~ 2 exp(-2x^2)
        //  Found that needed to replace psf with a slightly smaller number in the second element
        //   as otherwise _kolmogorov(b) came back as a very small number but with
        //  the same sign as _kolmogorov(a)
        //  kolmogi(0.5) = 0.82757355518990772
        //  so (1-q^(-(4-1)*2*x^2)) = (1-exp(-6*0.8275^2) ~ (1-exp(-4.1)
        const double jiggerb = 256 * DBL_EPSILON;
        double pba = psf/(1.0 - exp(-4))/2, pbb = psf * (1 - jiggerb)/2;
        double p2, q0;
        a = sqrt(-0.5 * log(pba));
        b = sqrt(-0.5 * log(pbb));
        // Use inversion of p = q - q^4 + q^9 - ...:
        //                 q = p + p^4 + 4p^7 - p^9 + 22p^10 ...
        p2 = psf/2.0;
        q0 = p2 + pow(p2, 4) + 4 * pow(p2, 7);
        x = sqrt(-log(q0) / 2);
        if (x < a || x > b) {
            x = (a+b)/2;
        }
    }
    assert(a <= b);

    iterations = 0;
    do {
        double x0 = x;
        double df = ((pcdf < 0.5) ? (pcdf - _kolmogorov(x0, 1)) : (_kolmogorov(x0, 0) - psf));
        double dfdx;

        if (fabs(df) == 0) {
            break;
        }
        /* Update the bracketing interval */
        if (df > 0) {
            if (x > a) {
                a = x;
                fa = df;
            }
        } else if (df < 0) {
            if (x < b) {
                b = x;
                fb = df;
            }
        }

        dfdx = kolmogp(x0);
        if (fabs(dfdx) <= 0.0) {
            // This is a problem for pure N-R, but we are bracketed, so can just do a bisection step!
            if (0) {
                /* Check if the value of df is already so small,
                so that we can't do any better no matter how hard we try */
                int krexp;
                frexp(df, &krexp);
                if (krexp > -1000) {
                    break;
                }
                mtherr("kolmogi", UNDERFLOW);
                return 0.0;
            }
            x = (a+b)/2;
            t = x0 - x;
        }
        else {
            t = df/dfdx;
            x = x0 - t;
        }

        /* Check out-of-bounds.
        Not expecting this to happen often --- kolmogorov is convex near x=infinity and
        concave near x=0, and we should be approaching from the correct side.
        If out-of-bounds, replace x with a midpoint of the bracket. */
        if (x >= a && x <= b)
        {
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
        }
        else {
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


/* Functional inverse of Kolmogorov survival statistic for two-sided test.
 * Finds x such that kolmogorov(x) = p.
 */
double kolmogi(double p)
{
    if (npy_isnan(p)) {
        return NPY_NAN;
    }
    return _kolmogi(p, 1-p);
}

/* Functional inverse of Kolmogorov cumulative statistic for two-sided test.
 * Finds x such that kolmogc(x) = p (or kolmogorov(x) = 1-p).
 */
double kolmogci(double p)
{
    if (npy_isnan(p)) {
        return NPY_NAN;
    }
    return _kolmogi(1-p, p);
}


/* Now the Smirnov functions for one-sided testing. */

/* For n bigger than 1028, nCv is no longer always representable in
   64bit floating point. Back off a bit further to account for
   the other factors in each term.
   */
#define SM_POLY_MAX_N 1013

/* Don't bother using the upper sum for small N, as not enough savings */
#define SM_UPPERSUM_MIN_N 10
/* Don't use the upper sum if lots of terms are involved as the series alternates
   sign and the terms get much bigger than 1 */
#define SM_UPPER_MAX_TERMS 3

/* Exact Smirnov statistic, for one-sided test.  */
static double _smirnov(int n, double d, int cdf)
{
    int nn;
    double p;
    int bUseUpperSum;
    int nTerms;

    /* This comparison should assure returning NaN whenever
     * d is NaN itself.  In original || form it would proceed */
    if (!(n > 0 && d >= 0.0 && d <= 1.0)) {
        return (NPY_NAN);
    }
    if (d == 0.0) {
        return (cdf ? 0.0 : 1.0);
    }
    if (d == 1.0) {
        return (cdf ? 1.0 : 0.0);
    }
    if (n == 1) {
        return (cdf ? d : 1-d);
    }

    bUseUpperSum = ((n*d <= 1)
                    || ((n >= SM_UPPERSUM_MIN_N) && (n*d <= SM_UPPER_MAX_TERMS)
                        && (d <= 0.5/sqrt(n))));

    if (bUseUpperSum) {
        nn = (int) (floor((double) n * (1.0 - d)));
        nTerms = n - nn;
    } else {
        nn = (int) (floor((double) n * (1.0 - d)));
        nn = MIN(nn, n-1);
        nTerms = nn+1;
    }

    p = 0.0;
    if (n < SM_POLY_MAX_N) {
        double c = 1.0;
        double p0 = 0;
        int i = 0;
        if (!bUseUpperSum) {
            p0 = exp(log1p(-d) * n); //  pow(1-d, (double)(n));
            c *= n;
            i++;
        }
        for (; i < nTerms; i++) {
            int v = (bUseUpperSum ? n - i : i);
            double evn = d + ((double) v) / n;
            double omevn = 1.0-evn;
            double aomevn = fabs(omevn);
            int sign = 1;
            double t;

            if (!bUseUpperSum && omevn <= 0) {
                continue;
            }
            if (omevn <= 0) {
                t = c * pow(evn, (double)(v - 1)) * pow(aomevn, (double)(n - v));
            } else {
                t = c * pow(evn, (double)(v - 1)) * exp(log1p(-evn) * (n - v));
            }
            if (npy_isnan(t)  || t == 0) {
                /* Can happen if one of the two pow()'s above is very small so 0 is returned.
                  Even though the full product would be non-zero, the order matters.
                  Redo but include the pow()s half at a time.*/
                double v2 = ((double)(v-1))/2;
                double nmv2 = ((double)(n-v))/2;
                double pwhalf = pow(evn, (double)v2) * pow(aomevn, (double)nmv2) * sqrt(c);
                if (npy_isnan(pwhalf)) {
                    return NPY_NAN;
                }
                if (pwhalf != 0) {
                    t = pwhalf * pwhalf;
                }
            }
            if (bUseUpperSum && (omevn < 0) && (n-v)%2) {
                sign = -1;
            }
            p += sign * t;
            /* Next update the combinatorial term  */
            if (bUseUpperSum) {
                c *= ((double) (v)) / (n - v + 1);
            } else {
                c *= ((double) (n - v)) / (v + 1);
            }
        }
        p = p*d + p0;
        if ((cdf && !bUseUpperSum) || (!cdf && bUseUpperSum)) {
            p = 1 - p;
        }
    }
    else {
        double lgamnp1 = lgam((double) (n + 1));
        double lt, lc = 0.0;
        int i = 0;
        for (i=0; i < nTerms; i++) {
            int v = (bUseUpperSum ? n - i : i);
            double evn = d + ((double) v) / n;
            double omevn = 1.0-evn;
            double aomevn = fabs(omevn);
            int sign = 1;
            if (!bUseUpperSum && omevn <= 0) {
                continue;
            }
            lc = lgamnp1 - lgam((double)(v + 1)) - lgam((double)(n - v + 1));
            lt = lc + (v - 1) * log(evn)  + (n - v) * log(aomevn); // log1p(-evn)
            if (bUseUpperSum && (omevn < 0) && (n-v)%2) {
                sign = -1;
            }
            if (lt >= -MAXLOG) {
                p += sign * exp(lt);
            }
        }
        p *= d;
        if ((cdf && !bUseUpperSum) || (!cdf && bUseUpperSum)) {
            p = 1 - p;
        }
    }
    // Ensure within [0, 1]
    p = CLIP(p, 0, 1);
    return p;
}


double smirnov(int n, double d)
{
    if (npy_isnan(d)) {
        return NPY_NAN;
    }
    return _smirnov(n, d, 0);
}

double smirnovc(int n, double d)
{
    if (npy_isnan(d)) {
        return NPY_NAN;
    }
    return _smirnov(n, d, 1);
}


/* Derivative of smirnov(n, x)

    smirnov(n, x) = sum f_v(x)  over {v : 0 <= v <= (n-1)*x}
    f_v  = nCv * x * pow(xvn, v-1) * pow(omxvn, n-v)     [but special case v=0,1,n below]
    f_v' = nCv *     pow(xvn, v-2) * pow(omxvn, n-v-1) * (xvn*omxvn + (v-1)*x*omxvn -(n-v)*x*xvn)
      where xvn = 1+v/n, omxvn = 1-xvn = 1-x-v/n
    Specializations for v=0,1,n
    f_0 = pow(omxvn, n)
    f_1 = n * x * pow(omxvn, n-1)
    f_n = x * pow(xvn, n-1)
    f_v(x) has a zero of order (n-v) at x=(1-v/n) for v=0,1,...n-1;
      and f_n(x) has a simple zero at x=0
    As the zero of f_(n-1) at x=1/n is only a simple zero, there is a
      discontinuity in smirnovp at x=1/n, and a fixable one at x=0.
    smirnovp is continuous at all other values of x.

    Note that the terms in these products can have wildly different magnitudes.
    If one of the pow()s in the product is "zero", recalculate the product
    in pieces and different order to avoid underflow.
*/

/* Use for small values of n, where nCv is representable as a double */
static double smirnovp_pow(int n, double x)
{
    int v, nn;
    double xvn, omxvn, pp, t;
    double c = 1.0; /* Holds the nCv combinations value */

    assert(n > 1);
    assert(x > 0);
    assert(x < 1);

    nn = (int)floor(n * (1.0 - x));
    /* The f_n term is not needed, and would introduce an inconvenient discontuity at x = 0.
    Should only appear if x == 0 (or x is less than DBL_EPSILON, so that 1-x==1) */
    nn = MIN(nn, n-1);
    pp = 0.0;

    /* Pull out v=0 */
    pp = c * pow(1-x, n-1) * (-n);
    c *= n;
    for (v = 1; v <= nn; v++) {
        xvn = x + ((double) v) / n;
        omxvn = 1.0 - xvn;
        if (omxvn <= 0.0) {  /* rounding issue */
            continue;
        }
        if (v == 1) {
            double coeff = (omxvn - (n-1)*x);
            if (coeff != 0) {
                t = c * pow(omxvn, n-2) * coeff;
                if (t == 0) {
                    double nmv2 = (n-2)/2.0;
                    double powhf = pow(omxvn, nmv2);
                    if (powhf != 0) {
                        t = (powhf * c) * coeff * powhf;
                    }
                }
                pp += t;
            }
        }
        else {
            double coeff = (xvn * omxvn + (v-1)*x*omxvn - (n-v) * x * xvn);
            if (coeff != 0) {
                t = c * pow(omxvn, n - v -1) * pow(xvn, v - 2) * coeff;
                if (t == 0) {
                    double v2 = (v-2)/2.0;
                    double nmv2 = (n-v-1)/2.0;
                    double pwhalf = pow(xvn, v2) * pow(omxvn, nmv2);
                    if (pwhalf != 0) {
                        t = (pwhalf * c) * coeff * pwhalf;
                    }
                }
                pp += t;
            }
        }
        c *= ((double) (n - v)) / (v + 1);
    }
    if (pp > 0) {
        pp = -0.0;
    }
    return pp;
}

/* Needed for larger values of n.
   nCv is not always representable as a double if N >= 1028.
   Use log(gamma(m+1)) and exponentiate later */
static double smirnovp_gamma(int n, double x)
{
    int v, nn;
    double xvn, omxvn, pp, t;

    assert(n > 1);
    assert(x > 0);
    assert(x < 1);

    nn = (int)floor(n * (1.0 - x));
    /* The f_n term is not needed, and would introduce an additional
        discontinuity at x = 0, so drop it.
       nn=n  <==>  x == 0 (or x is less than DBL_EPSILON, so that 1-x=1) */
    nn = MIN(nn, n-1);
    pp = 0.0;
    {
        double logn = log((double) (n));
        double lgamnp1 = lgam((double) (n + 1));
        /* Pull out v=0 */
        t = (n-1) * log1p(-x) + logn;
        if (t > -MAXLOG) {
            pp = -exp(t);
        }

        for (v = 1; v <= nn; v++) {
            int signcoeff = 1;
            xvn = x + ((double) v) / n;
            omxvn = 1.0 - xvn;
            if (fabs(omxvn) <= 0.0) {
                continue;
            }
            if (v == 1) {
                /* f_1' = nC1 * pow(omxvn, n-2) * (omxvn-(n-1)x)*/
                double coeff = (omxvn - (n-1)*x);
                if (coeff == 0) {
                    continue;
                }
                if (coeff < 0) {
                    signcoeff = -1;
                    coeff = -coeff;
                }
                t = (n-2) * log1p(-xvn) + log(coeff)+ logn;
            }
            else {
                /*  f_v' = nCv * pow(xvn, v-2) * pow(omxvn, n-v-1) *
                        (xvn*omxvn + (v-1)*x*omxvn -(n-v)*x*xvn)   */
                double lc;
                double coeff = (xvn * omxvn) + (v-1)*x*omxvn - (n-v) * x * xvn;
                if (coeff == 0) {
                    continue;
                }
                if (coeff < 0) {
                    signcoeff = -1;
                    coeff = -coeff;
                }
                lc = (lgamnp1 - lgam((double)(v+1)) - lgam((double)(n-v+1)));
                t = (v-2) * log(xvn) + (n-v-1) * log1p(-xvn);
                t += lc + log(coeff);
            }
            if (t > -MAXLOG) {
                pp += signcoeff * exp(t);
            }
        }
    }
    if (pp > 0) {
        pp = -0.0;
    }
    return pp;
}


/* Derivative of smirnov(n, x)
   One point of discontinuity at x=1/n
*/
double smirnovp(int n, double x)
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
    /* If x is very small, use the alternate sum (one term only) to evaluate */
    if (n * x <= 1) {
        double pp = -exp((n - 2) * log1p(x)) * (1 + n * x);
        return pp;
    }

    /* If n is too big, calculate using logs */
    if (n < SM_POLY_MAX_N) {
        return smirnovp_pow(n, x);
    }
    return smirnovp_gamma(n, x);
}


/* Solve smirnov(n, x) == p for p in [0, 1] with constraint 0<=x<=1.

    Discussion:
    smirnov(n, _) is easily invertible except for
    values of p close to the endpoints, where it can be quite brutal.
    Useful Asymptotic formula:
        As n -> infinity,  smirnov(n, x*sqrt(n)) ~ exp(-2 * x**2).
    Using this approximation to generate a starting point, and then
    approximating the derivative with the derivative of the limit
    sometimes (often?) succeeds, but also often results in
    slow convergence or non-convergence (indicated by a return of NPY_NAN.)
    Either the original estimate isn't a good starting point
    for Newton-Raphson(NR), or the derivative of the asymptote isn't close
    enough to the actual derivative.

    Current Algorithm:
    1. First handle the two endpoints: p=0, or p=1.
    2. Handle n==1: smirnov(1,x) = 1-x
    3. Exactly handle the case of extremely small p with a formula.
       (corresponds to (n-1)/n < x < 1.0).
    4a. Generate an initial bracket [a,b] and starting value x0 using the asymptotic limit.
    4b. Use Newton-Raphson to find the root, ensuring estimates stay within the interval [a,b].
        [Instrumentation suggest it converges in ~6-10 iterations for n<=100.]

    Note on 3.
    Small p (extremely small!):
     smirnov(n, x) = (1-x)**n,     if n*(1-x) <= 1, so
     smirnovi(n, p) = 1-p**(1/n),  if p is small enough, log(p) < -n*log(n).
    Asymptotically this cutoff handles fewer and fewer values of p,
    but smirnov(n, x) is not smooth at (1-1/n), so it is still useful.

    Note on 4b.  The estimate from the asymptotic is always > correct value of x.
      This can present a big problem if the true value lies on
      the "sloped" part of the graph, but the initial estimate lies on
      the "flat" part of the graph, (or if the estimate is bigger than 1.)
      [In general, if g(x)*g''(x) < 0, the N-R iteration overshoots the
       correct value.  If g(x)*g''(x) > 0, it undershoots.  Undershooting
       is good here as it keeps the iterations within the interval [0,1].]
     f(x) ~ exp(-2 n x^2) - p, so f'(x) ~ exp(-2 n x^2)*(16n x^2-4n),
     hence |f'(x)| maxxes out where f''(x)~0,  at x=0.5/sqrt(n).  Choose
     the endpoint of the interval so that f(x)*f''(x) < 0. 

    Alternative approaches considered.
    1. Only use bisection.
       Pros: Convergence guaranteed.
       Cons: Many iterations required to be within same tolerance.
    2. Use a smarter bracketing algorithm, such as a modified false position or brentq.
       Pros: Faster convergence than bisection.
       Cons: Exists elsewhere in scipy (scipy.optimize) but not easily
             callable from here, so an implementation needed.
       [This was tested from the Python side, and brentq had fewer correct digits with
       the same number of iterations than did this combination of bisection & NR.]

    Using the derivative results in needing fewer iterations
    to achieve same (or better) accuracy.
*/


/* Functional inverse of Smirnov distribution
 * finds x such that smirnov(n, x) = psf; smirnovc(n, x) = pcdf).  */
static double _smirnovi(int n, double psf, double pcdf)
{
    /* Need to use a bracketing NR algorithm here and be very careful about the starting point */
    double x, logpsf, logpcdf;
    int iterations = 0;
    int function_calls = 0;
    double a=0, b=1;
    double fa=pcdf, fb = -psf;
    double xmin, xmax, xmax6;
    double maxlogpcdf;

    if (!(n > 0 && psf >= 0.0 && pcdf >= 0.0 && pcdf <= 1.0 && psf <= 1.0)) {
        mtherr("smirnovi", DOMAIN);
        return (NPY_NAN);
    }
    if (fabs(1.0 - pcdf - psf) >  4* DBL_EPSILON) {
        mtherr("smirnovi", DOMAIN);
        return (NPY_NAN);
    }
    if (pcdf == 0.0) {
        return 0.0;
    }
    if (psf == 0.0) {
        return 1.0;
    }
    if (n == 1) {
        return pcdf;
    }

    /* Handle psf *very* close to 0.  Correspond to (n-1)/n < x < 1  */
    logpsf = log(psf);
    logpcdf = log(pcdf);

    xmin = - expm1(logpsf / n);  // =  1.0 - exp(lp / n)
    xmax = sqrt(-logpsf / (2.0 * n));
    xmax6 = xmax - 1.0 / (6 * n);

    if (logpsf <= -n * log(n)) { // xmin > 1 - 1.0 / n
        // Solve exactly.
        x = xmin;
        return x;
    }

    /* Handle 0 < x <= 1/n:   pcdf = x * (1+x)^*(n-1)  */
    maxlogpcdf = -log(n) + (n - 1) * log1p(1.0 / n);
    if (logpcdf < maxlogpcdf) {
        double P1 = exp(maxlogpcdf);
        double R = pcdf/P1;
        // Do one iteration of N-R solving: z*e^(z-1) = R, with z0=pcdf/P1
        // z <- z - (z exp(z-1) - pcdf)/((z+1)exp(z-1))
        // If z_0 = R, z_1 = R(1-exp(1-R))/(R+1)
        double z0 = R;

        xmin = pcdf / NPY_E;
        xmax = pcdf;

        z0 = (z0*z0 + R * exp(1-z0))/(1+z0);
        if (0) {
            // Do we need a 2nd iteration?
            z0 = (z0*z0 + R * exp(1-z0))/(1+z0);
        }
        x = z0/n;
        // The estimate can be quite good, especially if n>>1 and n*x << 1, but needs some N-R steps to tighten it up
        // For small n, sometimes the alpha estimate does not lie in the interval [pcdf/e, pcdf],
        // which is specific to a particular n.   If so, replace with a better estimate.
        x = CLIP(x, xmin, xmax);
    }
    else
    {
        x = xmax6;
        if (xmax6 < xmin) {
            x = (xmax + xmin)/2;
        }
    }
    assert (x < 1);

    a = xmin;
    b = xmax;
    fa = pcdf;
    fb = -psf; // Has the correct sign

    /* smirnov should be well-enough behaved for NR starting at this location */
    /* Use smirnov(n, x)-psf, or pcdf - smirnovc(n, x), whichever has smaller p */
    do {
        double dfdx, x0 = x, deltax, df;
        assert(x < 1);
        assert(x > 0);

        df = ((pcdf < 0.5) ? (pcdf - _smirnov(n, x0, 1)) :  (_smirnov(n, x0, 0) - psf));
        ++function_calls;
        if (df == 0) {
            return x;
        }
        /* Update the bracketing interval */
        if (df > 0) {
            if (x > a) {
                a = x;
                fa = df;
            }
        } else if (df < 0) {
             if (x < b) {
                 b = x;
                 fb = df;
             }
        }
        dfdx = smirnovp(n, x0);
        ++function_calls;
        if (dfdx == 0) {
            /* x was not within tolerance, but now we hit a 0 derivative.
            This implies that x >> 1/sqrt(n), and even then |smirnovp| >= |smirnov|
            so this condition is unexpected. */

            // This would be a problem for pure N-R, but we are bracketed, so can just do a bisection step.
            if (0) {
                /* Check if the value of df is already so small,
                so that we can't do any better no matter how hard we try */
                int krexp;
                frexp(df, &krexp);
                if (krexp > -1000) {
                    break;
                }
                mtherr("smirnovi", UNDERFLOW);
                return (NPY_NAN);
            }
            x = (a+b)/2;
            deltax = x0 - x;
        }  else {
            deltax = df / dfdx;
            x = x0 - deltax;
        }
        /* Check out-of-bounds.
           Not expecting this to happen ofen --- smirnov is convex near x=1 and
           concave near x=0, and we should be approaching from the correct side.
           If out-of-bounds, replace x with a midpoint of the brck. */
        if (x <= a || x >= b)
        {
            x = (a + b) / 2.0;
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

double smirnovi(int n, double p)
{
    if (npy_isnan(p)) {
        return NPY_NAN;
    }
    return _smirnovi(n, p, 1-p);
}

double smirnovci(int n, double p)
{
    if (npy_isnan(p)) {
        return NPY_NAN;
    }
    return _smirnovi(n, 1-p, p);
}


/* Type in a number.  */
/* void
 * getnum (s, px)
 * char *s;
 * double *px;
 * {
 * char str[30];
 * 
 * printf (" %s (%.15e) ? ", s, *px);
 * gets (str);
 * if (str[0] == '\0' || str[0] == '\n')
 * return;
 * sscanf (str, "%lf", px);
 * printf ("%.15e\n", *px);
 * }
 */
/* Type in values, get answers.  */
/*
 * void
 * main ()
 * {
 * int n;
 * double d, p, ps, pk, ek, y;
 * 
 * n = 5;
 * d = 0.0;
 * p = 0.1;
 * loop:
 * ps = n;
 * getnum ("n", &ps);
 * n = ps;
 * if (n <= 0)
 * {
 * printf ("? Operator error.\n");
 * goto loop;
 * }
 */
  /*
   * getnum ("d", &d);
   * ps = smirnov (n, d);
   * y = sqrt ((double) n) * d;
   * printf ("y = %.4e\n", y);
   * pk = kolmogorov (y);
   * printf ("Smirnov = %.15e, Kolmogorov/2 = %.15e\n", ps, pk / 2.0);
   */
/*
 * getnum ("p", &p);
 * d = smirnovi (n, p);
 * printf ("Smirnov d = %.15e\n", d);
 * y = kolmogi (2.0 * p);
 * ek = y / sqrt ((double) n);
 * printf ("Kolmogorov d = %.15e\n", ek);
 * goto loop;
 * }
 */
