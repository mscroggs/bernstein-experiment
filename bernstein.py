import numpy as np
import scipy.special
from scipy.special import comb


def evaluate_triangle(c0, q):

    assert c0.shape[0] == c0.shape[1]
    n = c0.shape[0] - 1
    rule0 = scipy.special.roots_jacobi(q, 0, 0)
    rule1 = scipy.special.roots_jacobi(q, 1, 0)
    rule0 = ((rule0[0] + 1) / 2, rule0[1] / 2)
    rule1 = ((rule1[0] + 1) / 2, rule1[1] / 4)
    rule2 = ((rule2[0] + 1) / 2, rule2[1] / 8)

    # d = 2
    # c1 = evalstep(c0, l=2, q)
    c1 = np.zeros((n+1, q))
    for i2, p in enumerate(rule0[0]):
        s = 1 - p
        r = p / s
        for alpha1 in range(n + 1):
            w = s**(n - alpha1)
            for alpha2 in range(n + 1 - alpha1):
                c1[alpha1, i2] += w * c0[alpha1, alpha2]
                w *= r * (n - alpha1 - alpha2) / (1 + alpha2)

    # c2 = evalstep(c1, l=1, q)
    c2 = np.zeros((q, q))
    for i1, p in enumerate(rule1[0]):
        s = 1 - p
        r = p / s
        w = s**n
        for alpha1 in range(n + 1):
            for i2 in range(q):
                c2[i1, i2] += w * c1[alpha1, i2]
            w *= r * (n - alpha1) / (1 + alpha1)

    return c2


def evaluate_tetrahedron(c0, q):

    assert c0.shape[0] == c0.shape[1]
    assert c0.shape[0] == c0.shape[2]
    n = c0.shape[0] - 1
    rule0 = scipy.special.roots_jacobi(q, 0, 0)
    rule1 = scipy.special.roots_jacobi(q, 1, 0)
    rule2 = scipy.special.roots_jacobi(q, 2, 0)
    rule0 = ((rule0[0] + 1) / 2, rule0[1] / 2)
    rule1 = ((rule1[0] + 1) / 2, rule1[1] / 4)
    rule2 = ((rule2[0] + 1) / 2, rule2[1] / 8)

    # d = 3
    # c1 = evalstep(c0, l=3, q)
    c1 = np.zeros((n + 1, n + 1, q))
    for i3, p in enumerate(rule0[0]):
        s = 1 - p
        r = p / s
        for alpha1 in range(n + 1):
            w = s**(n - alpha1)
            for alpha2 in range(n + 1 - alpha1):
                for alpha3 in range(n + 1 - alpha1 - alpha2):
                    c1[alpha1, alpha2, i3] += w * c0[alpha1, alpha2, alpha3]
                    w *= r * (n - alpha1 - alpha2 - alpha3) / (1 + alpha3)

    # c2 = evalstep(c1, l=2, q)
    c2 = np.zeros((n + 1, q, q))
    for i2, p in enumerate(rule1[0]):
        s = 1 - p
        r = p / s
        for alpha1 in range(n + 1):
            w = s**(n - alpha1)
            for alpha2 in range(n + 1 - alpha1):
                c2[alpha1, i1, i2] += w * c0[alpha1, alpha2, i2]
                w *= r * (n - alpha1 - alpha2) / (1 + alpha2)

    # c3 = evalstep(c2, l=1, q)
    c3 = np.zeros((q, q, q))
    for i1, p in enumerate(rule2[0]):
        s = 1 - p
        r = p / s
        w = s**n
        for alpha1 in range(n + 1):
            for i2 in range(q):
                c2[i1, i2, i3] += w * c1[alpha1, i2, i3]
            w *= r * (n - alpha1) / (1 + alpha1)

    return c2


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

    q = (fdegree + n)//2 + 1
    rule0 = scipy.special.roots_jacobi(q, 0, 0)
    rule1 = scipy.special.roots_jacobi(q, 1, 0)
    rule0 = ((rule0[0] + 1) / 2, rule0[1] / 2)
    rule1 = ((rule1[0] + 1) / 2, rule1[1] / 4)
    assert len(rule0[0] == q)

    f0 = np.array([
        [f(p1, p2 * (1 - p1)) for p2 in rule0[0]]
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
    for i2, (p, w) in enumerate(zip(*rule0)):
        s = 1 - p
        r = p / s
        for alpha1 in range(n + 1):
            ww = w * s ** (n - alpha1)
            for alpha2 in range(n + 1 - alpha1):
                f2[alpha1, alpha2] += ww * f1[alpha1, i2]
                ww *= r * (n - alpha1 - alpha2) / (1 + alpha2)

    return f2


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


c0 = np.ones((6, 6))
print(c0)
print(evaluate_triangle(c0, 6))
