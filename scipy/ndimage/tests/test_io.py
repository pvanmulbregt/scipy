from __future__ import division, print_function, absolute_import

from numpy.testing import assert_array_equal, dec, run_module_suite
from scipy._lib._numpy_compat import suppress_warnings
import scipy.ndimage as ndi

import os

try:
    from PIL import Image
    pil_missing = False
except ImportError:
    pil_missing = True


@dec.skipif(pil_missing, msg="The Python Image Library could not be found.")
def test_imread():
    lp = os.path.join(os.path.dirname(__file__), 'dots.png')
    with suppress_warnings() as sup:
        # PIL causes a Py3k ResourceWarning
        sup.filter(message="unclosed file")
        img = ndi.imread(lp, mode="RGB")
    assert_array_equal(img.shape, (300, 420, 3))

    with suppress_warnings() as sup:
        # PIL causes a Py3k ResourceWarning
        sup.filter(message="unclosed file")
        img = ndi.imread(lp, flatten=True)
    assert_array_equal(img.shape, (300, 420))

    with open(lp, 'rb') as fobj:
        img = ndi.imread(fobj, mode="RGB")
        assert_array_equal(img.shape, (300, 420, 3))


if __name__ == "__main__":
    run_module_suite()
