
import numpy as np
import bernstein
from symfem.elements.bernstein import bernstein_polynomials
from sympy import symbols, diff

np.set_printoptions(precision=2, suppress=True)

n = 2
nd = (n + 1)*(n + 2)*(n + 3)//6
q = 4
res = []
for u in range(n+1):
    for v in range(n + 1 - u):
        for w in range(n + 1 - u - v):
            c0 = np.zeros((n + 1, n + 1, n + 1))
            c0[u, v, w] = 1.0

            gradx = bernstein.evaluate_grad_tetrahedron(c0, q, direction='x')
            grady = bernstein.evaluate_grad_tetrahedron(c0, q, direction='y')
            gradz = bernstein.evaluate_grad_tetrahedron(c0, q, direction='z')

            gx = n * bernstein.compute_moments_tetrahedron(n - 1, gradx, None)
            gy = n * bernstein.compute_moments_tetrahedron(n - 1, grady, None)
            gz = n * bernstein.compute_moments_tetrahedron(n - 1, gradz, None)
            lp = np.zeros_like(c0)
            lp[1:, :-1, :-1] += gx
            lp[:-1, 1:, :-1] += gy
            lp[:-1, :-1, 1:] += gz
            lp[:-1, :-1, :-1] -= (gx + gy + gz)
            print(lp)

            for i in range(n + 1):
                for j in range(n + 1 - i):
                    for k in range(n + 1 - i - j):
                        res += [lp[i, j, k]]
res = np.array(res).reshape(nd, nd)

print(res)
print()

x, y, z = symbols('x y z')
b0 = bernstein_polynomials(n, 3)

# Permute order of basis functions (different in symfem)
c = 0
p = np.zeros((n+1, n+1, n+1), dtype=int)
for i in range(n+1):
    for j in range(n+1-i):
        for k in range(n+1-i-j):
            p[i, j, k] = c
            c += 1
print(p)
perm = []
for i in range(n+1):
    for j in range(n+1-i):
        for k in range(n+1-i-j):
            perm += [p[k, j, i]]

print(perm)
b = [b0[i] for i in perm]
print(b)


bx = [diff(bi, 'x') for bi in b]
by = [diff(bi, 'y') for bi in b]
bz = [diff(bi, 'z') for bi in b]

lap2 = np.array([[float(
    (bi * bj).integrate([x, 0, 1 - y - z], [y, 0, 1 - z], [z, 0, 1])
) for bi in bx] for bj in bx])

lap2 += np.array([[float(
    (bi * bj).integrate([x, 0, 1 - y - z], [y, 0, 1 - z], [z, 0, 1])
    ) for bi in by] for bj in by])

lap2 += np.array([[float(
    (bi * bj).integrate([x, 0, 1 - y - z], [y, 0, 1 - z], [z, 0, 1])
    ) for bi in bz] for bj in bz])

assert np.allclose(lap2, res)
