
import numpy as np
import bernstein
from symfem.elements.bernstein import bernstein_polynomials
from sympy import symbols, diff

np.set_printoptions(precision=5, suppress=True)

n = 4
nd = (n + 1)*(n + 2)//2
q = 5
w = []
for u in range(n+1):
    for v in range(n + 1 - u):
        c0 = np.zeros((n+1, n+1))
        c0[u, v] = 1.0

        val = bernstein.evaluate_triangle(c0, q)
        print(val)
        lp =  bernstein.compute_moments_triangle(n, val, None)
        print(lp)

        quit()

        for i in range(n + 1):
            for j in range(n + 1 - i):
                w += [lp[i, j]]
w = np.array(w).reshape(nd, nd)

print(w)
print()

x, y = symbols('x y')
b0 = bernstein_polynomials(n, 2)

# Permute order of basis functions (different in symfem)
c = 0
p = np.zeros((n+1, n+1), dtype=int)
for i in range(n+1):
    for j in range(n+1-i):
        p[i, j] = c
        c += 1
print(p)
perm = []
for i in range(n+1):
    for j in range(n+1-i):
        perm += [p[j, i]]

print(perm)
b = [b0[i] for i in perm]
print(b)

mass = np.array([[float(
    (bi * bj).integrate([x, 0, 1-y], [y, 0, 1])
) for bi in b] for bj in b])

print(mass)
assert np.allclose(w, mass)
