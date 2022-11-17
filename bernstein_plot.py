import bernstein
import numpy as np
import scipy
import matplotlib.pyplot as plt
from bernstein_cffi import cffi_compile_tri

nq = 15
nd = 12
cffi_compile_tri(nd, nq)
from _cffi_bernstein import ffi, lib

def f(x, y): return x*x

fdegree = 2*(nq-1) - nd
c0 = bernstein.compute_moments_triangle(nd, f, fdegree)
n = c0.shape[0]
print('n=', n)
print('nd = ', nd)
b = np.zeros(n*(n+1)//2, dtype=np.float64)
c = 0
for i in range(n):
    for j in range(n - i):
        b[c] = c0[i, j]
        c+=1

z = np.zeros(nq * nq, dtype=np.float64)
z0 = np.zeros(nq * nq, dtype=np.float64)
z1 = np.zeros(nq * nq, dtype=np.float64)
lib.evaluate_tri(ffi.cast("double *", b.ctypes.data),
                 ffi.cast("double *", z.ctypes.data))
z *= (nd+1)*(nd+2)
lib.evaluate_gradx_tri(ffi.cast("double *", b.ctypes.data),
                       ffi.cast("double *", z0.ctypes.data))
z0 *= (nd+3)*(nd+2)*(nd+1)/nd
lib.evaluate_grady_tri(ffi.cast("double *", b.ctypes.data),
                       ffi.cast("double *", z1.ctypes.data))
z1 *= (nd+3)*(nd+2)*(nd+1)/nd
print(z.max(), z.min(), 1/z.min())
print(z0.max(), z0.min(), 1/z0.min())
print(z1.max(), z1.min(), 1/z1.min())

rule0 = scipy.special.roots_jacobi(nq, 0, 0)
rule1 = scipy.special.roots_jacobi(nq, 1, 0)
rule0 = ((rule0[0] + 1) / 2, rule0[1] / 2)
rule1 = ((rule1[0] + 1) / 2, rule1[1] / 4)
pts = np.array([(x, y*(1 - x)) for x in rule1[0] for y in rule0[0]])

plt.tricontourf(pts[:, 0], pts[:, 1], z0)
plt.plot(pts[:, 0], pts[:, 1], marker='o', color='k', linewidth=0)
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.gca().set_aspect('equal')

plt.figure()
plt.tricontourf(pts[:, 0], pts[:, 1], z1)
plt.plot(pts[:, 0], pts[:, 1], marker='o', color='k', linewidth=0)
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.gca().set_aspect('equal')

plt.figure()
plt.tricontourf(pts[:, 0], pts[:, 1], z)
plt.plot(pts[:, 0], pts[:, 1], marker='o', color='k', linewidth=0)
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.gca().set_aspect('equal')

plt.show()
