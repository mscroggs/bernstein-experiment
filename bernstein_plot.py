import bernstein
import numpy as np
import scipy
import matplotlib.pyplot as plt


def f(x, y): return x*y


nq = 5
c0 = bernstein.compute_moments_triangle(nq, f, 3)

rule0 = scipy.special.roots_jacobi(nq, 0, 0)
rule1 = scipy.special.roots_jacobi(nq, 1, 0)
rule0 = ((rule0[0] + 1) / 2, rule0[1] / 2)
rule1 = ((rule1[0] + 1) / 2, rule1[1] / 4)
pts = np.array([(x, y*(1 - x)) for x in rule1[0] for y in rule0[0]])

z = bernstein.evaluate_grad_triangle(c0, nq, 'y').flatten()
print(z)

plt.tricontourf(pts[:, 0], pts[:, 1], z)
plt.plot(pts[:, 0], pts[:, 1], marker='o', color='k', linewidth=0)
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.gca().set_aspect('equal')

plt.figure()
z = bernstein.evaluate_triangle(c0, nq).flatten()

plt.tricontourf(pts[:, 0], pts[:, 1], z)
plt.plot(pts[:, 0], pts[:, 1], marker='o', color='k', linewidth=0)

plt.xlim(0, 1)
plt.ylim(0, 1)
plt.gca().set_aspect('equal')
plt.show()
