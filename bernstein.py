import numpy as np
import scipy.special
from scipy.special import comb


def compute_moments_tetrahedron(n, f, fdegree):
    """Compute the Bernstein moments.

    These are defined in equation (12) of https://doi.org/10.1137/11082539X
    (Ainsworth, Andriamaro, Davydov, 2011).

    Args:
      n: The polynomial degree of the Bernstein polynomials.
      f: The function to take moments with.
      fdegree: The polynomial degree of the function f.

    Returns:
      A three-dimensional array containing the Bernstein moments.
    """

    jdegree = (n + fdegree) // 2 + 1
    rule0 = scipy.special.roots_jacobi(jdegree, 2, 0)
    rule1 = scipy.special.roots_jacobi(jdegree, 1, 0)
    rule2 = scipy.special.roots_jacobi(jdegree, 0, 0)
    rule0 = ((rule0[0] + 1) / 2, rule0[1] / 8)
    rule1 = ((rule1[0] + 1) / 2, rule1[1] / 4)
    rule2 = ((rule2[0] + 1) / 2, rule2[1] / 2)

    q = len(rule1[0])
    assert len(rule2) == len(rule1)
    assert len(rule2) == len(rule0)

    f0 = np.empty((q, q, q))
    for i, x in enumerate(rule0[0]):
        for j, y in enumerate(rule1[0]):
            for k, z in enumerate(rule2[0]):
                f0[i, j, k] = f(x, y*(1 - x), z*(1 - y)*(1 - x))

    f1 = np.zeros((n+1, q, q))
    for i0, (p, w) in enumerate(zip(*rule0)):
        s = 1 - p
        r = p / s
        ww = w * s ** n
        for alpha1 in range(n + 1):
            for i1 in range(q):
                for i2 in range(q):
                    f1[alpha1, i1, i2] += ww * f0[i0, i1, i2]
            ww *= r * (n - alpha1) / (1 + alpha1)

    f2 = np.zeros((n + 1, n + 1, q))
    for i1, (p, w) in enumerate(zip(*rule1)):
        s = 1 - p
        r = p / s
        for alpha1 in range(n + 1):
            ww = w * s ** (n - alpha1)
            for alpha2 in range(n + 1 - alpha1):
                for i2 in range(q):
                    f2[alpha1, alpha2, i2] += ww * f1[alpha1, i1, i2]
                ww *= r * (n - alpha1 - alpha2) / (1 + alpha2)

    f3 = np.zeros((n + 1, n + 1, n + 1))
    for i2, (p, w) in enumerate(zip(*rule2)):
        s = 1 - p
        r = p / s
        for alpha1 in range(n + 1):
            for alpha2 in range(n + 1 - alpha1):
                ww = w * s ** (n - alpha1 - alpha2)
                for alpha3 in range(n + 1 - alpha1 - alpha2):
                    f3[alpha1, alpha2, alpha3] += ww * f2[alpha1, alpha2, i2]
                    ww *= r * (n - alpha1 - alpha2 - alpha3) / (1 + alpha3)

    return f3


def compute_moments_triangle(n, f, fdegree):
    """Compute the Bernstein moments.

    These are defined in equation (12) of https://doi.org/10.1137/11082539X
    (Ainsworth, Andriamaro, Davydov, 2011).

    Args:
      n: The polynomial degree of the Bernstein polynomials.
      f: The function to take moments with.
      fdegree: The polynomial degree of the function f.

    Returns:
      A two-dimensional array containing the Bernstein moments.
    """
    if fdegree == 0:
        return np.array(
            [[f(0, 0) / (n + 1) / (n + 2) for _ in range(n + 1)] for _ in range(n + 1)])

    rule1 = scipy.special.roots_jacobi((n + fdegree) // 2 + 1, 1, 0)
    rule2 = scipy.special.roots_jacobi((n + fdegree) // 2 + 1, 0, 0)
    rule1 = ((rule1[0] + 1) / 2, rule1[1] / 4)
    rule2 = ((rule2[0] + 1) / 2, rule2[1] / 2)

    q = len(rule1[0])
    assert len(rule2) == len(rule1)

    f0 = np.array([
        [f(p1, p2 * (1 - p1)) for p2 in rule2[0]]
        for p1 in rule1[0]])

    f1 = np.zeros((n+1, q))
    for i1, (p, w) in enumerate(zip(*rule1)):
        s = 1 - p
        r = p / s
        ww = w * s ** n
        for alpha1 in range(n + 1):
            for i2 in range(q):
                f1[alpha1, i2] += ww * f0[i1, i2]
            ww *= r * (n - alpha1) / (1 + alpha1)

    f2 = np.zeros((n + 1, n + 1))
    for i2, (p, w) in enumerate(zip(*rule2)):
        s = 1 - p
        r = p / s
        for alpha1 in range(n + 1):
            ww = w * s ** (n - alpha1)
            for alpha2 in range(n + 1 - alpha1):
                f2[alpha1, alpha2] += ww * f1[alpha1, i2]
                ww *= r * (n - alpha1 - alpha2) / (1 + alpha2)

    return f2


def compute_mass_matrix_tetrahedron(n, f=None, fdegree=0):
    """
    Compute the mass matrix with a weight function.

    These method is described in section 4.1 of https://doi.org/10.1137/11082539X
    (Ainsworth, Andriamaro, Davydov, 2011).

    Args:
      n: The polynomial degree of the Bernstein polynomials.
      f: The function to take moments with. This is the function c from the paper.
      fdegree: The polynomial degree of the function f.

    Returns:
      A mass matrix.
    """
    moments = compute_moments_tetrahedron(2 * n, f, fdegree)

    nd = (n + 1)*(n + 2)*(n + 3) // 6
    mat = np.zeros((nd, nd))

    # Pack index
    idx = {}
    c = 0
    for i in range(n + 1):
        for j in range(n + 1 - i):
            for k in range(n + 1 - i - j):
                idx[(k, j, i)] = c
                c += 1

    for a in range(n + 1):
        for a2 in range(n + 1 - a):
            for a3 in range(n + 1 - a - a2):
                i = idx[(a, a2, a3)]
                for b in range(n + 1):
                    for b2 in range(n + 1 - b):
                        for b3 in range(n + 1 - b - b2):
                            j = idx[(b, b2, b3)]
                            mat[i, j] = comb(a + b, a) * comb(a2 + b2, a2) \
                                * comb(a3 + b3, a3)\
                                * comb(2 * n - a - b - a2 - b2 - a3 - b3, n - a - a2 - a3) \
                                * moments[a + b, a2 + b2, a3 + b3]

    mat /= comb(2 * n, n)
    return mat


def compute_mass_matrix_triangle(n, f=None, fdegree=0):
    """
    Compute the mass matrix with a weight function.

    These method is described in section 4.1 of https://doi.org/10.1137/11082539X
    (Ainsworth, Andriamaro, Davydov, 2011).

    Args:
      n: The polynomial degree of the Bernstein polynomials.
      f: The function to take moments with. This is the function c from the paper.
      fdegree: The polynomial degree of the function f.

    Returns:
      A mass matrix.
    """
    moments = compute_moments_triangle(2 * n, f, fdegree)

    mat = np.zeros(((n + 1) * (n + 2) // 2, (n + 1) * (n + 2) // 2))

    # Pack index
    def idx(i, j): return ((2 * n + 3) * j - j * j) // 2 + i

    for a in range(n + 1):
        for a2 in range(n + 1 - a):
            i = idx(a, a2)
            for b in range(n + 1):
                for b2 in range(n + 1 - b):
                    j = idx(b, b2)
                    mat[i, j] = comb(a + b, a) * comb(a2 + b2, a2) \
                        * comb(2 * n - a - b - a2 - b2, n - a - a2) \
                        * moments[a + b, a2 + b2]

    mat /= comb(2 * n, n)
    return mat
