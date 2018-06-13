from __future__ import division, print_function, absolute_import

from numpy import ndarray, array, real, imag, real_if_close, abs, finfo

# try:
#     from ._ufuncs import _jthetac
# except ImportError:
#     print(e)
#     def _jtheta(z, tau):
#         raise ValueError("jtheta is not yet supported")
#         return 0

def _jthetac(z, tau):
    raise ValueError("jtheta is not yet supported")
    return 0

def jtheta(z, tau):
    r"""Jacobi theta function

    Defined as [1]_,

    .. math:: theta(z, tau) = \sum_{k=-\infty}^{\infty} exp(2\pi i k z) exp(\pi i \tau k^2)

    Parameters
    ----------
    z : complex or float
    tau : complex, with im(tau) > 0

    Returns
    -------
    theta : complex

    Notes
    -----
    For \tau with small imaginary part, the function first computes
    a better tau to use, and applies the functional equation

    .. math:: \theta(\frac{z}{c\tau+d}, \frac{a\tau+b}{c\tau+d}) = \zeta \sqrt{c\tau+d} \theta(z,\tau)

    for some 8th root of unity, :math:`\zeta`.

    References
    ----------
    .. [1] "Tata Lectures on Theta I",
      Mumford, D. and Musili, C. and Nori, M. and Previato, E. and Stillman, M., 2006.
      Birkhauser Boston
    """
    if imag(tau) <= 0:
        raise ValueError("tau %s not in upper half-plane" % tau)
    if isinstance(z, ndarray):
        return array(_jthetac(_, tau) for _ in z)
    _z = 1.0*z + 0*1j
    _tau = 1.0*tau + 0*1j
    ans = _jthetac(_z, _tau)
    if abs(imag(ans)) < finfo(float).eps * abs(real(ans)):
        ans = real_if_close(ans)
    return ans
