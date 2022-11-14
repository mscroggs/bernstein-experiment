
import scipy.special
from scipy.special import comb
import numpy as np
from cffi import FFI


def compute_mass_matrix_triangle(n, fdegree):

    rule1 = scipy.special.roots_jacobi((2*n + fdegree) // 2 + 1, 1, 0)
    rule2 = scipy.special.roots_jacobi((2*n + fdegree) // 2 + 1, 0, 0)
    rule1 = ((rule1[0] + 1) / 2, rule1[1] / 4)
    rule2 = ((rule2[0] + 1) / 2, rule2[1] / 2)
    q = len(rule1[0])
    nd = (n + 1) * (n + 2) // 2

    cmat = [str(int(comb(p+q, p))) for p in range(n+1) for q in range(n+1)]

    ccode = f"""

#define nd {nd}
#define nq {q}

void tabulate_bernstein_mass_tri(double *f0, double *A)
{{
  // Input: f0 at quadrature points (size {q} x {q})
  // Output: A - mass matrix (size {(n+1)*(n+2)//2} x {(n+1)*(n+2)//2})
  double rule1p[{q}] = {{{', '.join([str(p) for p in rule1[0]])}}};
  double rule1w[{q}] = {{{', '.join([str(p) for p in rule1[1]])}}};
  double rule2p[{q}] = {{{', '.join([str(p) for p in rule2[0]])}}};
  double rule2w[{q}] = {{{', '.join([str(p) for p in rule2[1]])}}};

  double f1[{(2*n+1)}][{q}] = {{}};

  for (int i1 = 0; i1 < {q}; ++i1)
  {{
    double s = 1.0 - rule1p[i1];
    double r = rule1p[i1] / s;
    double ww = rule1w[i1];
    for (int alpha1 = 0; alpha1 < {2*n}; ++alpha1)
      ww *= s;
    for (int alpha1 = 0; alpha1 < {2*n+1}; ++alpha1)
    {{
      for (int i2 = 0; i2 < {q}; ++i2)
        f1[alpha1][i2] += ww * f0[i1*{q} + i2];
      ww *= r *({2*n}-alpha1)/(1.0 + alpha1);
    }}
  }}

  double f2[{(2*n+1)}][{(2*n+1)}] = {{0}};

  for (int i2 = 0; i2 < {q}; ++i2)
  {{
    double s = 1.0 - rule2p[i2];
    double r = rule2p[i2]/s;
    double s0 = 1.0;
    for (int alpha1 = 0; alpha1 < {2*n+1}; ++alpha1)
    {{
      double ww = rule2w[i2] * s0;
      s0 *= s;
      for (int alpha2 = 0; alpha2 < alpha1+1; ++alpha2)
      {{
        f2[{2*n}-alpha1][alpha2] += ww * f1[{2*n}-alpha1][i2];
        ww *= r * (alpha1 - alpha2) / (1.0 + alpha2);
      }}
    }}
  }}

  // double A[{(n + 1) * (n + 2) // 2}][{(n + 1) * (n + 2) // 2}] = {{0}};
  double cmat[{n+1}][{n+1}] = {{{', '.join(cmat)}}};

  for (int a = 0; a < {n+1}; ++a)
  {{
    int i = a;
    int ia = {n+1};
    for (int a2 = 0; a2 < ({n + 1} - a); ++a2)
    {{
      // int i = ({(2 * n + 3)} * a2 - a2 * a2) / 2 + a;
      for (int b = 0; b < {n + 1}; ++b)
      {{
        int j = b;
        int jb = {n+1};
        for (int b2 = 0; b2 < {n + 1} - b; ++b2)
        {{
        // int j = ({(2 * n + 3)} * b2 - b2 * b2) / 2 + b;
        A[i*{(n+1)*(n+2)//2} + j] = cmat[a][b] * cmat[a2][b2]
           * cmat[{n} - a - a2][{n} - b - b2] * f2[a + b][a2 + b2] / cmat[{n}][{n}];
        j += jb;
        jb--;
        }}
      }}
      i += ia;
      ia--;
    }}
  }}
}}

"""

    return (nd, q, ccode)

def cffi_mass_tri(n, fdegree):
    nd, q, code = compute_mass_matrix_triangle(n, fdegree)
    ffi = FFI()
    ffi.set_source("_cffi_bernstein", code)
    ffi.cdef("void tabulate_bernstein_mass_tri(double *f0, double *A);");
    ffi.compile(verbose=True)
    from _cffi_bernstein import ffi, lib
    f0 = np.ones((q, q), dtype=np.float64)
    A = np.empty((nd, nd), dtype=np.float64)
    lib.tabulate_bernstein_mass_tri(ffi.cast("double *", f0.ctypes.data),
                                    ffi.cast("double *", A.ctypes.data))
    return(A)

print(cffi_mass_tri(5, 4))
