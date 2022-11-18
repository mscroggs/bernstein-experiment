import numpy as np
import symfem
import sympy
import pytest
import scipy

import bernstein
from bernstein_cffi import cffi_compile_tri

x, y, z = sympy.symbols("x y z")


def test_evaluation_triangle():
    n = 8
    rule0 = scipy.special.roots_jacobi(n, 0, 0)
    rule1 = scipy.special.roots_jacobi(n, 1, 0)
    rule0 = ((rule0[0] + 1) / 2, rule0[1] / 2)
    rule1 = ((rule1[0] + 1) / 2, rule1[1] / 4)
    pts = np.array([(x, y*(1 - x)) for x in rule1[0] for y in rule0[0]])

    b = symfem.elements.bernstein.bernstein_polynomials(n - 1, 2)
    bx = [sympy.diff(bi, x) for bi in b]
    by = [sympy.diff(bi, y) for bi in b]

    j = 0
    for a0 in range(n):
        for a1 in range(n - a0):
            c0 = np.zeros((n, n))
            c0[a1, a0] = 1.0
            print(a0, a1, j)
            z0 = bernstein.evaluate_triangle(c0, n).flatten()
            z0x = bernstein.evaluate_grad_triangle(c0, n, 'x').flatten()
            z0y = bernstein.evaluate_grad_triangle(c0, n, 'y').flatten()
            z1 = np.array([float(b[j].subs({x: p[0], y: p[1]})) for p in pts])
            z1x = np.array([float(bx[j].subs({x: p[0], y: p[1]})) for p in pts])
            z1y = np.array([float(by[j].subs({x: p[0], y: p[1]})) for p in pts])
            j += 1

            assert np.allclose(z0, z1)
            assert np.allclose(z0x, z1x)
            assert np.allclose(z0y, z1y)


def test_cffi_triangle():
    nd = 12
    nq = 15
    cffi_compile_tri(nd, nq)
    from _cffi_bernstein import ffi, lib

    def f(x, y): return x*x

    # Compute basis from f
    fdegree = 2*(nq-1) - nd
    c0 = bernstein.compute_moments_triangle(nd, f, fdegree)
    n = c0.shape[0]
    b = np.zeros(n*(n+1)//2, dtype=np.float64)
    c = 0
    for i in range(n):
        for j in range(n - i):
            b[c] = c0[i, j]
            c += 1

    # Create quadrature points and evaluate f,
    # and compute basis (cffi)
    rule0 = scipy.special.roots_jacobi(nq, 0, 0)
    rule1 = scipy.special.roots_jacobi(nq, 1, 0)
    rule0 = ((rule0[0] + 1) / 2, rule0[1] / 2)
    rule1 = ((rule1[0] + 1) / 2, rule1[1] / 4)
    pts = np.array([(x, y*(1 - x)) for x in rule1[0] for y in rule0[0]])
    f0 = np.array([f(x, y) for (x, y) in pts], dtype=np.float64)
    f2 = np.zeros_like(b)
    lib.moment_tri(ffi.cast("double *", f0.ctypes.data),
                   ffi.cast("double *", f2.ctypes.data))

    # Compare two results
    assert np.allclose(f2, b)

    z = np.zeros(nq * nq, dtype=np.float64)
    z0 = np.zeros(nq * nq, dtype=np.float64)
    z1 = np.zeros(nq * nq, dtype=np.float64)
    lib.evaluate_tri(ffi.cast("double *", b.ctypes.data),
                     ffi.cast("double *", z.ctypes.data))
    lib.evaluate_gradx_tri(ffi.cast("double *", b.ctypes.data),
                           ffi.cast("double *", z0.ctypes.data))
    lib.evaluate_grady_tri(ffi.cast("double *", b.ctypes.data),
                           ffi.cast("double *", z1.ctypes.data))

    zb = bernstein.evaluate_triangle(c0, nq).flatten()
    z0b = bernstein.evaluate_grad_triangle(c0, nq, 'x').flatten()
    z1b = bernstein.evaluate_grad_triangle(c0, nq, 'y').flatten()
    assert np.allclose(z, zb)
    assert np.allclose(z1, z0b)
    assert np.allclose(z0, z1b)


@pytest.mark.parametrize("px", range(4))
@pytest.mark.parametrize("py", range(4))
@pytest.mark.parametrize("n", range(1, 4))
def test_integrals_triangle(px, py, n):
    def f(x, y):
        return x ** px * y ** py

    b = symfem.elements.bernstein.bernstein_polynomials(n, 2)

    integrals1 = [float((bi * f(x, y)).integrate([x, 0, 1-y], [y, 0, 1]))
                  for bi in b]
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
