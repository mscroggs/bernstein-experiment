import scipy.special
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

  double c1[{(n + 1)}][{q}] = {{}};

  for (int i2 = 0; i2 < {q}; ++i2)
  {{
    double s = 1.0 - rule0p[i2];
    double r = rule0p[i2] / s;
    int c = 0;
    for (int alpha1 = 0; alpha1 < {n + 1}; ++alpha1)
    {{
      double w = 1.0;
      for (int j = 0; j < {n} - alpha1; ++j)
        w *= s;
      double c1v = 0.0;
      for (int alpha2 = 0; alpha2 < {n + 1} - alpha1; ++alpha2)
      {{
        c1v += w * c0[c++];
        w *= r * ({n} - alpha1 - alpha2)/(1.0 + alpha2);
      }}
      c1[alpha1][i2] = c1v;
    }}
  }}

  // double c2[{q}][{q}] = {{0}};

  for (int i1 = 0; i1 < {q}; ++i1)
  {{
    double s = 1.0 - rule1p[i1];
    double r = rule1p[i1] / s;
    double w = 1.0;
    for (int i = 0; i < {n}; ++i)
      w *= s;
    for (int alpha1 = 0; alpha1 < {n + 1}; ++alpha1)
    {{
      for (int i2 = 0; i2 < {q}; ++i2)
        c2[{q} * i1 + i2] += w * c1[alpha1][i2];
      w *= r * ({n} - alpha1) / (1.0 + alpha1);
    }}
  }}
}}

void evaluate_grad_tri(double *c0, double *c2)
{{
  // Input: c0 ({(n + 1) * (n + 2) // 2}) - dofs
  // Output: c2 (2, {q * q}) - gradient values at quadrature points
  double rule1p[{q}] = {{{', '.join([str(p) for p in rule1[0]])}}};
  double rule0p[{q}] = {{{', '.join([str(p) for p in rule0[0]])}}};

  double c1[2][{n}][{q}];

  for (int i2 = 0; i2 < {q}; ++i2)
  {{
    double s = 1.0 - rule0p[i2];
    double r = rule0p[i2] / s;
    int c = 0;
    int dc = {n + 1};
    for (int alpha1 = 0; alpha1 < {n}; ++alpha1)
    {{
      double w = 1.0;
      for (int j = 0; j < {n - 1} - alpha1; ++j)
        w *= s;
      double c1vx = 0.0;
      double c1vy = 0.0;
      for (int alpha2 = 0; alpha2 < {n} - alpha1; ++alpha2)
      {{
        c1vx += w * (c0[c + 1] - c0[c]);
        c1vy += w * (c0[c + dc] - c0[c]);
        ++c;
        w *= r * ({n - 1} - alpha1 - alpha2)/(1.0 + alpha2);
      }}
      ++c;
      --dc;
      c1[0][alpha1][i2] = {n} * c1vx;
      c1[1][alpha1][i2] = {n} * c1vy;
    }}
  }}

  // double c2[2][{q}][{q}] = {{0}};

  for (int dim = 0; dim < 2; ++dim)
    for (int i1 = 0; i1 < {q}; ++i1)
    {{
      double s = 1.0 - rule1p[i1];
      double r = rule1p[i1] / s;
      double w = 1.0;
      for (int i = 0; i < {n - 1}; ++i)
        w *= s;
      for (int alpha1 = 0; alpha1 < {n}; ++alpha1)
      {{
        for (int i2 = 0; i2 < {q}; ++i2)
          c2[{q*q} * dim + {q} * i1 + i2] += w * c1[dim][alpha1][i2];
        w *= r * ({n - 1} - alpha1) / (1.0 + alpha1);
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
        double s = 1.0 - rule1p[i1];
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

    // double f2[{(n + 2) * (n + 1) // 2}] = {{}};
    for (int i2 = 0; i2 < {q}; ++i2)
    {{
      double s = 1.0 - rule0p[i2];
      double r = rule0p[i2] / s;
      double w = rule0w[i2];
      int c = 0;
      for (int alpha1 = 0; alpha1 < {n + 1}; ++alpha1)
      {{
        double ww = w;
        for (int j = 0; j < {n} - alpha1; ++j)
          ww *= s;
        for (int alpha2 = 0; alpha2 < {n + 1} - alpha1; ++alpha2)
        {{
          f2[c++] += ww * f1[alpha1][i2];
          ww *= r * ({n} - alpha1 - alpha2) / (1 + alpha2);
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

void evaluate_grad_tet(double *c0, double *c3)
{{
  // Input: c0 ({(n)*(n+1)*(n+2)//6}) - dofs
  // Output: c3 (3, {q*q*q}) - gradient values at quadrature points
  double rule2p[{q}] = {{{', '.join([str(p) for p in rule2[0]])}}};
  double rule1p[{q}] = {{{', '.join([str(p) for p in rule1[0]])}}};
  double rule0p[{q}] = {{{', '.join([str(p) for p in rule0[0]])}}};

  double c1[3][{n}][{n}][{q}];

  for (int i2 = 0; i2 < {q}; ++i2)
  {{
    double s = 1.0 - rule0p[i2];
    double r = rule0p[i2] / s;
    int c = 0;
    int dz = {(n+2)*(n+1)//2};
    for (int alpha1 = 0; alpha1 < {n}; ++alpha1)
    {{
      int dy = {n + 1} - alpha1;
      for (int alpha2 = 0; alpha2 < {n} - alpha1; ++alpha2)
      {{
        double w = 1.0;
        for (int j = 0; j < {n - 1} - alpha1 - alpha2; ++j)
          w *= s;
        double c1vx = 0.0;
        double c1vy = 0.0;
        double c1vz = 0.0;
        for (int alpha3 = 0; alpha3 < {n} - alpha1 - alpha2; ++alpha3)
        {{
          c1vx += w * (c0[c + 1] - c0[c]);
          c1vy += w * (c0[c + dy] - c0[c]);
          c1vz += w * (c0[c + dz] - c0[c]);
          ++c;
          w *= r * ({n - 1} - alpha1 - alpha2 - alpha3)/(1.0 + alpha3);
        }}
        c1[0][alpha1][alpha2][i2] = {n} * c1vx;
        c1[1][alpha1][alpha2][i2] = {n} * c1vy;
        c1[2][alpha1][alpha2][i2] = {n} * c1vz;
        ++c;
        --dy;
        --dz;
      }}
      ++c;
      --dz;
    }}
  }}

  double c2[3][{n}][{q}][{q}] = {{0}};

  for (int dim = 0; dim < 3; ++dim)
  {{
    for (int i1 = 0; i1 < {q}; ++i1)
    {{
      double s = 1.0 - rule1p[i1];
      double r = rule1p[i1]/s;
      for (int alpha1 = 0; alpha1 < {n}; ++alpha1)
      {{
        double w = 1.0;
        for (int j = 0; j < {n - 1} - alpha1; ++j)
          w *= s;
        for (int alpha2 = 0; alpha2 < {n} - alpha1; ++alpha2)
        {{
          for (int i2 = 0; i2 < {q}; ++i2)
            c2[dim][alpha1][i1][i2] += w * c1[dim][alpha1][alpha2][i2];
          w *= r * ({n - 1} - alpha1 - alpha2) / (1.0 + alpha2);
        }}
      }}
    }}
  }}

  // double c3[{q}][{q}][{q}] = {{0}};

  for (int dim = 0; dim < 3; ++dim)
  {{
    for (int i0 = 0; i0 < {q}; ++i0)
    {{
      double s = 1.0 - rule2p[i0];
      double r = rule2p[i0] / s;
      double w = 1.0;
      for (int j = 0; j < {n - 1}; ++j)
        w *= s;
      for (int alpha1 = 0; alpha1 < {n}; ++alpha1)
      {{
        for (int i1 = 0; i1 < {q}; ++i1)
          for (int i2 = 0; i2 < {q}; ++i2)
            c3[dim * {q * q * q} + i0 * {q * q} + i1 * {q} + i2] += w * c2[dim][alpha1][i1][i2];
         w *= r * ({n - 1} - alpha1) / (1.0 + alpha1);
      }}
    }}
  }}

}}


void moment_tet(double *f0, double *f3)
{{
    // Input f0 at quadrature points ({q * q * q})
    // Output f2 basis function values ({(n + 1) * (n + 2) * (n + 3) // 6})
    double rule2w[{q}] = {{{', '.join([str(p) for p in rule2[1]])}}};
    double rule1w[{q}] = {{{', '.join([str(p) for p in rule1[1]])}}};
    double rule0w[{q}] = {{{', '.join([str(p) for p in rule0[1]])}}};
    double rule2p[{q}] = {{{', '.join([str(p) for p in rule2[0]])}}};
    double rule1p[{q}] = {{{', '.join([str(p) for p in rule1[0]])}}};
    double rule0p[{q}] = {{{', '.join([str(p) for p in rule0[0]])}}};

    double f1[{n + 1}][{q}][{q}] = {{}};
    for (int i0 = 0; i0 < {q}; ++i0)
    {{
        double s = 1.0 - rule2p[i0];
        double r = rule2p[i0] / s;
        double ww = rule2w[i0];
        for (int j = 0; j < {n}; ++j)
          ww *= s;
        for (int alpha1 = 0; alpha1 < {n + 1}; ++alpha1)
        {{
          for (int i1 = 0; i1 < {q}; ++i1)
            for (int i2 = 0; i2 < {q}; ++i2)
              f1[alpha1][i1][i2] += ww * f0[i0 * {q * q} + i1 * {q} + i2];
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

    // double f3[{(n + 3) * (n + 2) * (n + 1) // 6}]
    for (int i2 = 0; i2 < {q}; ++i2)
    {{
      double s = 1.0 - rule0p[i2];
      double r = rule0p[i2] / s;
      double w = rule0w[i2];
      int c = 0;
      for (int alpha1 = 0; alpha1 < {n + 1}; ++alpha1)
        for (int alpha2 = 0; alpha2 < {n + 1} - alpha1; ++alpha2)
        {{
          double ww = w;
          for (int j = 0; j < {n} - alpha1 - alpha2; ++j)
            ww *= s;
          for (int alpha3 = 0; alpha3 < {n + 1} - alpha1 - alpha2; ++alpha3)
          {{
             f3[c++] += ww * f2[alpha1][alpha2][i2];
             ww *= r * ({n} - alpha1 - alpha2 - alpha3)/(1.0 + alpha3);
          }}
        }}
    }}

}}

"""

    return ccode


def cffi_compile_all(n, q):
    code = codegen_tri(n, q)
    code += codegen_tet(n, q)

    ffi = FFI()
    ffi.set_source("_cffi_bernstein", code)
    ffi.cdef("void evaluate_tri(double *c0, double *c2);")
    ffi.cdef("void evaluate_grad_tri(double *c0, double *c2);")
    ffi.cdef("void moment_tri(double *f0, double *f2);")

    ffi.cdef("void evaluate_tet(double *c0, double *c3);")
    ffi.cdef("void evaluate_grad_tet(double *c0, double *c3);")
    ffi.cdef("void moment_tet(double *f0, double *f3);")
    ffi.compile(verbose=False)
