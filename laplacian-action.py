
import numpy as np
import bernstein
from symfem.elements.bernstein import bernstein_polynomials
from sympy import symbols, diff

np.set_printoptions(precision=2, suppress=True)

n = 2
nd = (n + 1)*(n+2)//2
q = 5
w = []
for u in range(n+1):
    for v in range(n + 1 - u):
        c0 = np.zeros((n+1, n+1))
        c0[u, v] = 1.0

        gradx = bernstein.evaluate_grad_triangle(c0, q, direction='x')
        grady = bernstein.evaluate_grad_triangle(c0, q, direction='y')

        gx = n * bernstein.compute_moments_triangle(n - 1, gradx, None)
        gy = n * bernstein.compute_moments_triangle(n - 1, grady, None)
        lp = np.zeros_like(c0)
        lp[:-1, 1:] += gy
        lp[1:, :-1] += gx
        lp[:-1, :-1] -= gx + gy
        print(lp)

        for i in range(n +1):
            for j in range(n + 1 - i):
                w += [lp[i, j]]
print(np.array(w).reshape(nd, nd))
print()

x, y = symbols('x y')
b0 = bernstein_polynomials(2, 2)

perm = [0,3,5,1,4,2]
b = [b0[i] for i in perm]
print(b)


bx = [diff(bi, 'x') for bi in b]
by = [diff(bi, 'y') for bi in b]

lap2 = np.array([[float(
    (bi * bj).integrate([x, 0, 1-y], [y, 0, 1])
) for bi in bx] for bj in bx])

lap2 += np.array([[float(
    (bi * bj).integrate([x, 0, 1-y], [y, 0, 1])
    ) for bi in by] for bj in by])


print(lap2)