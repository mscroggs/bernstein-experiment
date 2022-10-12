import numpy as np
import scipy.special
import symfem
import sympy
import pytest

import bernstein


@pytest.mark.parametrize("px", range(4))
@pytest.mark.parametrize("py", range(4))
@pytest.mark.parametrize("n", range(1, 4))
def test_integrals_triangle(px, py, n):
    x, y = sympy.Symbol("x"), sympy.Symbol("y")

    b = symfem.elements.bernstein.bernstein_polynomials(n, 2)

    def f(x, y):
        return x ** px * y ** py

    integrals1 = [(i *  f(x, y)).integrate([x, 0, 1-y], [y, 0, 1]) for i in b]
    integrals1 = [float(i) for i in integrals1]

    integrals2 = bernstein.compute_moments_triangle(n, f, px + py)
    d = len(integrals2)
    integrals2 = [integrals2[j, i] for i in range(d) for j in range(d - i)]

    assert np.allclose(integrals1, integrals2)
