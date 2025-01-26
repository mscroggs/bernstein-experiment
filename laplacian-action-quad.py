
import numpy as np
import bernstein
from symfem.elements.bernstein import bernstein_polynomials
from sympy import symbols, diff

np.set_printoptions(precision=2, suppress=True)

n = 2
q = 3

nd = (n + 1) * (n + 1)
w = []
for u in range(n + 1):
    for v in range(n + 1):
        c0 = np.zeros((n + 1, n + 1))
        c0[u, v] = 1.0

        cd0 = c0[:, :-1] - c0[:, 1:]
        gradx = n * bernstein.evaluate_quad(cd0, q)
        cd0 = c0[:-1, :] - c0[1:, :]
        grady = n * bernstein.evaluate_quad(cd0, q)
        print(gradx, "\n", grady)
        gx = n * bernstein.compute_moments_quad(gradx, n, n - 1)
        gy = n * bernstein.compute_moments_quad(grady, n - 1, n)
        lp = np.zeros_like(c0)
        print(lp.shape, gx.shape, gy.shape)
        lp[1:, :] -= gy
        lp[:-1, :] += gy
        lp[:, 1:] -= gx
        lp[:, :-1] += gx
        print('lp = ', lp)

        for i in range(n + 1):
            for j in range(n + 1):
                w += [lp[i, j]]
w = np.array(w).reshape(nd, nd)

print(w)
print()

x, y = symbols('x y')
bx = bernstein_polynomials(n, 1)
by = [b.subs(x, y) for b in bernstein_polynomials(n, 1)]
b = [b * c for b in by for c in bx]

bx = [diff(bi, 'x') for bi in b]
by = [diff(bi, 'y') for bi in b]

lap2 = np.array([[float(
    (bi * bj).integrate([x, 0, 1], [y, 0, 1])
) for bi in bx] for bj in bx])

lap2 += np.array([[float(
    (bi * bj).integrate([x, 0, 1], [y, 0, 1])
) for bi in by] for bj in by])

print(lap2)
assert np.allclose(w, lap2)
