from __future__ import division, print_function, absolute_import

# import warnings

import numpy as np

import scipy.special as sc

from . import _stats
from ._distn_infrastructure import (get_distribution_names,
                                    rv_circular)


class uniformc_gen(rv_circular):
    r"""A Uniform continuous random variable.

    %(before_notes)s

    Notes
    -----
    `x` is assumed to be an angle in radians,

    The probability density function for `uniform` is:

    .. math::

        f(x) = \frac{ 1 }{ 2 \pi }

    %(after_notes)s


    %(example)s

    """
    def _pdf(self, x):
        return 1.0 / (2*np.pi)

    def _cdf(self, x):
        return (x - self.a) / (self.b - self.a)


uniformc = uniformc_gen(name='uniformc')


class vonmisesc_gen(rv_circular):
    r"""A Von Mises continuous random variable.

    %(before_notes)s

    Notes
    -----
    Circular distribution, defined on [-\pi, \pi]

    The probability density function for `vonmises` is:

    .. math::

        f(x, \kappa) = \frac{ \exp(\kappa \cos(x)) }{ 2 \pi I[0](\kappa) }

    for :math:`\kappa > 0`.

    `vonmises` takes :math:`\kappa` as a shape parameter.

    %(after_notes)s

    See Also
    --------
    vonmises_line : The same distribution as a continuous distribution,
            defined on a [-\pi, \pi] segment of the real line.

    %(example)s

    """
    def _rvs(self, kappa):
        return self._random_state.vonmises(0.0, kappa, size=self._size)

    def _pdf(self, x, kappa):
        return np.exp(kappa * np.cos(x)) / (2*np.pi*sc.i0(kappa))

    def _cdf(self, x, kappa):
        return _stats.von_mises_cdf(kappa, x)

    def _entropy(self, kappa):
        return (-kappa * sc.i1(kappa) / sc.i0(kappa) +
                np.log(2 * np.pi * sc.i0(kappa)))

vonmisesc = vonmisesc_gen(momtype=0, name='vonmisesc')

# Collect names of classes and objects in this module.
pairs = list(globals().items())
_distn_names, _distn_gen_names = get_distribution_names(pairs, rv_circular)

__all__ = _distn_names + _distn_gen_names
