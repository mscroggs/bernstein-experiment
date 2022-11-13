import numpy as np
import symfem
import sympy
import pytest

import bernstein

x, y, z = sympy.symbols("x y z")


@pytest.mark.parametrize("px", range(4))
@pytest.mark.parametrize("py", range(4))
@pytest.mark.parametrize("n", range(1, 4))
def test_integrals_triangle(px, py, n):
    def f(x, y):
        return x ** px * y ** py

    b = symfem.elements.bernstein.bernstein_polynomials(n, 2)

    integrals1 = [float(
        (i * f(x, y)).integrate([x, 0, 1-y], [y, 0, 1])
    ) for i in b]
    integrals1 = [float(i) for i in integrals1]

    integrals2 = bernstein.compute_moments_triangle(n, f, px + py)
    d = len(integrals2)
    integrals2 = [integrals2[j, i] for i in range(d) for j in range(d - i)]

    assert np.allclose(integrals1, integrals2)


@pytest.mark.parametrize("px", range(2))
@pytest.mark.parametrize("py", range(2))
@pytest.mark.parametrize("pz", range(3))
@pytest.mark.parametrize("n", range(1, 3))
def test_integrals_tetrahedron(px, py, pz, n):
    def f(x, y, z):
        return x ** px * y ** py * z ** pz

    print(px, py, pz)
    b = symfem.elements.bernstein.bernstein_polynomials(n, 3)

    integrals1 = [float(
        (bi * f(x, y, z)).integrate([x, 0, 1 - y - z], [y, 0, 1 - z], [z, 0, 1])
    ) for bi in b]
    integrals1 = [float(i) for i in integrals1]

    integrals2 = bernstein.compute_moments_tetrahedron(n, f, px + py + pz)
    d = len(integrals2)
    integrals2 = [integrals2[k, j, i]
                  for i in range(d)
                  for j in range(d - i)
                  for k in range(d - i - j)]

    assert np.allclose(integrals1, integrals2)


@pytest.mark.parametrize("px", range(3))
@pytest.mark.parametrize("py", range(3))
@pytest.mark.parametrize("n", range(1, 3))
def test_mass_matrix_triangle(px, py, n):

    def f(x, y):
        return x ** px * y ** py

    mass1 = bernstein.compute_mass_matrix_triangle(n, f, px + py)

    b = symfem.elements.bernstein.bernstein_polynomials(n, 2)

    mass2 = np.array([[float(
        (i * j * f(x, y)).integrate([x, 0, 1-y], [y, 0, 1])
    ) for i in b] for j in b])

    print(mass1)
    print(mass2)

    assert np.allclose(mass1, mass2)


@pytest.mark.parametrize("px", range(2))
@pytest.mark.parametrize("py", range(3))
@pytest.mark.parametrize("pz", range(2))
@pytest.mark.parametrize("n", range(1, 3))
def test_mass_matrix_tetrahedron(px, py, pz, n):

    def f(x, y, z):
        return x ** px * y ** py * z ** pz

    mass1 = bernstein.compute_mass_matrix_tetrahedron(n, f, px + py + pz)

    b = symfem.elements.bernstein.bernstein_polynomials(n, 3)

    mass2 = np.array([[float(
        (bi * bj * f(x, y, z)).integrate([x, 0, 1 - y - z], [y, 0, 1 - z], [z, 0, 1])
    ) for bi in b] for bj in b])

    assert np.allclose(mass1, mass2)
