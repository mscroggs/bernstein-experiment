
import numpy as np
import bernstein
from symfem.elements.bernstein import bernstein_polynomials
from sympy import symbols, diff

np.set_printoptions(precision=5, suppress=True)

n = 2
nd = (n + 1) * (n + 1)
q = 3
w = []
for u in range(n + 1):
    for v in range(n + 1):
        c0 = np.zeros((n+1, n+1))
        c0[u, v] = 1.0

        val = bernstein.evaluate_quad(c0, q)
        print(val)
        lp =  bernstein.compute_moments_quad(val, n, n)
        print(lp)


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

print(b)

mass = np.array([[float(
    (bi * bj).integrate([x, 0, 1], [y, 0, 1])
) for bj in b] for bi in b])

print(mass)
print(w)
assert np.allclose(w, mass)
