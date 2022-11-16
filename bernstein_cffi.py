import scipy.special
import numpy as np
from cffi import FFI


def codegen_tri(n, q):

    rule0 = scipy.special.roots_jacobi(q, 0, 0)
    rule1 = scipy.special.roots_jacobi(q, 1, 0)
    rule0 = ((rule0[0] + 1) / 2, rule0[1] / 2)
    rule1 = ((rule1[0] + 1) / 2, rule1[1] / 4)

    ccode = f"""
void evaluate_tri(double *c0, double *c2)
{{
  // Input: c0 ({(n+1)*(n+2)//2}) - dofs
  // Output: c2 ({q*q}) - values at quadrature points
  double rule1p[{q}] = {{{', '.join([str(p) for p in rule1[0]])}}};
  double rule0p[{q}] = {{{', '.join([str(p) for p in rule0[0]])}}};

  double c1[{(n+1)}][{q}] = {{}};

  for (int i2 = 0; i2 < {q}; ++i2)
  {{
    double s = 1.0 - rule0p[i2];
    double r = rule0p[i2] / s;
    double ww = 1.0;
    int c = 0;
    for (int alpha1m = 0; alpha1m < {n + 1}; ++alpha1m)
    {{
      double w = ww;
      double c1v = 0.0;
      for (int alpha2 = 0; alpha2 < alpha1m + 1; ++alpha2)
      {{
        c1v += w * c0[c++];
        w *= r * (alpha1m - alpha2)/(1.0 + alpha2);
      }}
      c1[{n} - alpha1m][i2] = c1v;
      ww *= s;
    }}
  }}

  // double c2[{q}][{q}] = {{0}};

  for (int i1 = 0; i1 < {q}; ++i1)
  {{
    double s = 1.0 - rule1p[i1];
    double r = rule1p[i1]/s;
    double w = 1.0;
    for (int i = 0; i < {n}; ++i)
      w *= s;
    for (int alpha1 = 0; alpha1 < {n + 1}; ++alpha1)
    {{
      for (int i2 = 0; i2 < {q}; ++i2)
        c2[{q}*i1 + i2] += w * c1[alpha1][i2];
      w *= r * ({n} - alpha1) / (1.0 + alpha1);
    }}
  }}
}}

void moment_tri(double *f0, double *f2)
{{
    // Input f0 at quadrature points ({q*q})
    // Output f2 basis function values ({(n+1)*(n+2)//2})
    double rule1w[{q}] = {{{', '.join([str(p) for p in rule1[1]])}}};
    double rule0w[{q}] = {{{', '.join([str(p) for p in rule0[1]])}}};
    double rule1p[{q}] = {{{', '.join([str(p) for p in rule1[0]])}}};
    double rule0p[{q}] = {{{', '.join([str(p) for p in rule0[0]])}}};

    double f1[{n+1}][{q}] = {{}};
    for (int i1 = 0; i1 < {q}; ++i1)
    {{
        double s = 1 - rule1p[i1];
        double r = rule1p[i1] / s;
        double ww = rule1w[i1];
        for (int j = 0; j < {n}; ++j)
          ww *= s;
        for (int alpha1 = 0; alpha1 < {n+1}; ++alpha1)
        {{
            for (int i2 = 0; i2 < {q}; ++i2)
                f1[alpha1][i2] += ww * f0[i1*{q} + i2];
            ww *= r * ({n} - alpha1) / (1 + alpha1);
        }}
    }}

    // double f2[{(n+2)*(n+1)//2}] = {{}};
    for (int i2 = 0; i2 < {q}; ++i2)
    {{
      double s = 1 - rule0p[i2];
      double r = rule0p[i2] / s;
      double w = rule0w[i2];
      double w0 = 1.0;
      int c = 0;
      for (int alpha1m = 0; alpha1m < {n + 1}; ++alpha1m)
      {{
        double ww = w * w0;
        w0 *= s;
        for (int alpha2 = 0; alpha2 < alpha1m + 1; ++alpha2)
        {{
          f2[c++] += ww * f1[{n} - alpha1m][i2];
          ww *= r * (alpha1m - alpha2) / (1 + alpha2);
        }}
      }}
    }}
}}

"""

    return ccode


def codegen_tet(n, q):

    rule0 = scipy.special.roots_jacobi(q, 0, 0)
    rule1 = scipy.special.roots_jacobi(q, 1, 0)
    rule2 = scipy.special.roots_jacobi(q, 2, 0)
    rule0 = ((rule0[0] + 1) / 2, rule0[1] / 2)
    rule1 = ((rule1[0] + 1) / 2, rule1[1] / 4)
    rule2 = ((rule2[0] + 1) / 2, rule2[1] / 8)

    ccode = f"""
void evaluate_tet(double *c0, double *c3)
{{
  // Input: c0 ({(n+1)*(n+2)*(n+3)//6}) - dofs
  // Output: c3 ({q*q*q}) - values at quadrature points
  double rule2p[{q}] = {{{', '.join([str(p) for p in rule2[0]])}}};
  double rule1p[{q}] = {{{', '.join([str(p) for p in rule1[0]])}}};
  double rule0p[{q}] = {{{', '.join([str(p) for p in rule0[0]])}}};

  double c1[{n+1}][{n+1}][{q}] = {{}};

  for (int i2 = 0; i2 < {q}; ++i2)
  {{
    double s = 1.0 - rule0p[i2];
    double r = rule0p[i2] / s;
    int c = 0;
    for (int alpha1 = 0; alpha1 < {n + 1}; ++alpha1)
    {{
      for (int alpha2 = 0; alpha2 < {n + 1} - alpha1; ++alpha2)
      {{
        double w = 1.0;
        for (int j = 0; j < {n} - alpha1 - alpha2; ++j)
          w *= s;
        double c1v = 0.0;
        for (int alpha3 = 0; alpha3 < {n + 1} - alpha1 - alpha2; ++alpha3)
        {{
          c1v += w * c0[c++];
          w *= r * ({n} - alpha1 - alpha2 - alpha3)/(1.0 + alpha3);
        }}
        c1[alpha1][alpha2][i2] = c1v;
      }}
    }}
  }}

  double c2[{n+1}][{q}][{q}] = {{0}};

  for (int i1 = 0; i1 < {q}; ++i1)
  {{
    double s = 1.0 - rule1p[i1];
    double r = rule1p[i1]/s;
    for (int alpha1 = 0; alpha1 < {n + 1}; ++alpha1)
    {{
      double w = 1.0;
      for (int j = 0; j < {n} - alpha1; ++j)
        w *= s;
      for (int alpha2 = 0; alpha2 < {n+1} - alpha1; ++alpha2)
      {{
        for (int i2 = 0; i2 < {q}; ++i2)
          c2[alpha1][i1][i2] += w * c1[alpha1][alpha2][i2];
        w *= r * ({n} - alpha1 - alpha2) / (1.0 + alpha2);
      }}
    }}
  }}

  // double c3[{q}][{q}][{q}] = {{0}};
  for (int i0 = 0; i0 < {q}; ++i0)
  {{
    double s = 1.0 - rule2p[i0];
    double r = rule2p[i0]/s;
    double w = 1.0;
    for (int j = 0; j < {n}; ++j)
      w *= s;
    for (int alpha1 = 0; alpha1 < {n + 1}; ++alpha1)
    {{
      for (int i1 = 0; i1 < {q}; ++i1)
        for (int i2 = 0; i2 < {q}; ++i2)
          c3[i0*{q*q} + i1*{q} + i2] += w * c2[alpha1][i1][i2];
       w *= r * ({n} - alpha1) / (1.0 + alpha1);
    }}
  }}

}}

void moment_tet(double *f0, double *f3)
{{
    // Input f0 at quadrature points ({q*q*q})
    // Output f2 basis function values ({(n+1)*(n+2)*(n+3)//6})
    double rule2w[{q}] = {{{', '.join([str(p) for p in rule2[1]])}}};
    double rule1w[{q}] = {{{', '.join([str(p) for p in rule1[1]])}}};
    double rule0w[{q}] = {{{', '.join([str(p) for p in rule0[1]])}}};
    double rule2p[{q}] = {{{', '.join([str(p) for p in rule2[0]])}}};
    double rule1p[{q}] = {{{', '.join([str(p) for p in rule1[0]])}}};
    double rule0p[{q}] = {{{', '.join([str(p) for p in rule0[0]])}}};

    double f1[{n+1}][{q}][{q}] = {{}};
    for (int i0 = 0; i0 < {q}; ++i0)
    {{
        double s = 1 - rule2p[i0];
        double r = rule2p[i0] / s;
        double ww = rule2w[i0];
        for (int j = 0; j < {n}; ++j)
          ww *= s;
        for (int alpha1 = 0; alpha1 < {n + 1}; ++alpha1)
        {{
          for (int i1 = 0; i1 < {q}; ++i1)
            for (int i2 = 0; i2 < {q}; ++i2)
              f1[alpha1][i1][i2] += ww * f0[i0*{q*q} + i1*{q} + i2];
          ww *= r * ({n} - alpha1) / (1.0 + alpha1);
        }}
    }}

    double f2[{n+1}][{n+1}][{q}] = {{}};
    for (int i1 = 0; i1 < {q}; ++i1)
    {{
      double s = 1 - rule1p[i1];
      double r = rule1p[i1] / s;
      double w = rule1w[i1];
      for (int alpha1 = 0; alpha1 < {n + 1}; ++alpha1)
      {{
        double ww = w;
        for (int j = 0; j < {n} - alpha1; ++j)
          ww *= s;
        for (int alpha2 = 0; alpha2 < {n+1} - alpha1; ++alpha2)
        {{
          for (int i2 = 0; i2 < {q}; ++i2)
            f2[alpha1][alpha2][i2] += ww * f1[alpha1][i1][i2];
          ww *= r * ({n} - alpha1 - alpha2) / (1.0 + alpha2);
        }}
      }}
    }}

    // double f3[{(n+3)*(n+2)*(n+1)//6}]
    for (int i2 = 0; i2 < {q}; ++i2)
    {{
      double s = 1.0 - rule0p[i2];
      double r = rule0p[i2] / s;
      double w = rule0w[i2];
      int c = 0;
      for (int alpha1 = 0; alpha1 < {n+1}; ++alpha1)
        for (int alpha2 = 0; alpha2 < {n+1} - alpha1; ++alpha2)
        {{
          double ww = w;
          for (int j = 0; j < {n} - alpha1 - alpha2; ++j)
            ww *= s;
          for (int alpha3 = 0; alpha3 < {n+1} - alpha1 - alpha2; ++alpha3)
          {{
             f3[c++] += ww * f2[alpha1][alpha2][i2];
             ww *= r * ({n} - alpha1 - alpha2 -alpha3)/(1.0 + alpha3);
          }}
        }}
    }}

}}

"""

    return ccode


def cffi_eval_tri(n, q):
    code = codegen_tri(n, q)
    print(code)
    ffi = FFI()
    ffi.set_source("_cffi_bernstein", code)
    ffi.cdef("void evaluate_tri(double *c0, double *c2);")
    ffi.cdef("void moment_tri(double *f0, double *f2);")
    ffi.compile(verbose=False)
    from _cffi_bernstein import ffi, lib

    # Input basis function
    c0 = np.ones((n + 1)*(n + 2)//2, dtype=np.float64)
    print('B[c0] = ', c0)

    # Interpolate at quadrature points
    c2 = np.zeros(q * q, dtype=np.float64)
    lib.evaluate_tri(ffi.cast("double *", c0.ctypes.data),
                     ffi.cast("double *", c2.ctypes.data))
    print('q = ', c2)

    # Compute moment back into basis
    f2 = np.zeros_like(c0)
    lib.moment_tri(ffi.cast("double *", c2.ctypes.data),
                   ffi.cast("double *", f2.ctypes.data))
    print('B[f2] = ', f2, sum(f2))

    
def cffi_eval_tet(n, q):
    code = codegen_tet(n, q)
    print(code)
    ffi = FFI()
    ffi.set_source("_cffi_bernstein", code)
    ffi.cdef("void evaluate_tet(double *c0, double *c3);")
    ffi.cdef("void moment_tet(double *f0, double *f3);")
    ffi.compile(verbose=False)
    from _cffi_bernstein import ffi, lib

    # Input basis function
    c0 = np.ones((n + 1)*(n + 2)*(n + 3)//6, dtype=np.float64)
    print('B[c0] = ', c0)

    # Interpolate at quadrature points
    c3 = np.zeros(q * q * q, dtype=np.float64)
    lib.evaluate_tet(ffi.cast("double *", c0.ctypes.data),
                     ffi.cast("double *", c3.ctypes.data))
    print('q = ', c3)

    # Compute moment back into basis
    f3 = np.zeros_like(c0)
    lib.moment_tet(ffi.cast("double *", c3.ctypes.data),
                   ffi.cast("double *", f3.ctypes.data))
    print('B[f3] = ', f3, sum(f3))

    

cffi_eval_tet(5, 5)
