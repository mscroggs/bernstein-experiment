
import numpy as np
import bernstein_cffi

np.set_printoptions(precision=2, suppress=True, linewidth=210)

nq = 16
nt = 3
code = bernstein_cffi.cffi_compile_all(nt, nq)

from _cffi_bernstein import ffi, lib # noqa

eval_grad = [None,
             lib.evaluate_grad_tri_1,
             lib.evaluate_grad_tri_2,
             lib.evaluate_grad_tri_3,
             lib.evaluate_grad_tri_4,
             lib.evaluate_grad_tri_5]

moment = [None,
          lib.moment_tri_1,
          lib.moment_tri_2,
          lib.moment_tri_3,
          lib.moment_tri_4,
          lib.moment_tri_5]

stiff = [None,
         lib.stiff_action_tri_1,
         lib.stiff_action_tri_2,
         lib.stiff_action_tri_3,
         lib.stiff_action_tri_4,
         lib.stiff_action_tri_5,
         ]


n = 3
nd = (n + 1) * (n + 2) // 2

# Figure out the add/subtract positions of B(n-1) in B(n)
xmap = np.arange(n+1, nd, dtype=int)
xymap = []
d = 0
for i in range(n):
    xymap += [np.arange(d, d + n - i, dtype=int)]
    d += n + 1 - i
xymap = np.concatenate(xymap)

ymap = xymap + 1

print(xmap, ymap, xymap)

res = np.zeros((nd, nd))

w = []
for k in range(nd):
    c0 = np.zeros(nd)
    c0[k] = 1.0

    grad = np.zeros((2, nq, nq), dtype=np.float64)
    eval_grad[n](ffi.cast("double *", c0.ctypes.data),
                 ffi.cast("double *", grad.ctypes.data))
    gradx = grad[0, :]
    grady = grad[1, :]

    gx = np.zeros(n*(n+1)//2, dtype=np.float64)
    moment[n-1](ffi.cast("double *", gradx.ctypes.data),
                ffi.cast("double *", gx.ctypes.data))
    gy = np.zeros(n*(n+1)//2, dtype=np.float64)
    moment[n-1](ffi.cast("double *", grady.ctypes.data),
                ffi.cast("double *", gy.ctypes.data))
    gx *= n
    gy *= n
    res[k, xmap] = gx
    res[k, ymap] += gy
    res[k, xymap] -= (gx + gy)

    resk = np.zeros(nd, dtype=np.float64)
    stiff[n](ffi.cast("double *", c0.ctypes.data),
             ffi.cast("double *", resk.ctypes.data))
#    print(resk)

# print('res = \n', res)

ndtet = (nt+3)*(nt+2)*(nt+1)//6
print('tet = ', ndtet)
coords = np.array([0.0, 0.0, 0.0,
                   1.0, 0.0, 0.0,
                   0.0, 1.0, 0.0,
                   0.0, 0.0, 1.0], dtype=np.float64)
for i in range(ndtet):
    c0 = np.zeros(ndtet, dtype=np.float64)
    c0[i] = 1.0
    res = np.zeros(ndtet, dtype=np.float64)
    lib.stiff_action_tet(ffi.cast("double *", c0.ctypes.data),
                         ffi.cast("double *", res.ctypes.data),
                         ffi.cast("double *", coords.ctypes.data))
    print(res)
