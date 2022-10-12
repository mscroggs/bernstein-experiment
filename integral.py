import numpy as np
import scipy.special
import sympy
import typing

lambdas = [sympy.Symbol("lambda_1"), sympy.Symbol("lambda_2"), sympy.Symbol("lambda_3"),
           sympy.Symbol("lambda_4")]
ts = [sympy.Symbol("t_1"), sympy.Symbol("t_2"), sympy.Symbol("t_3")]
x, y, z = sympy.Symbol("x"), sympy.Symbol("y"), sympy.Symbol("z")


def f(x, y):
    return x**2
fdegree = 2


def single_choose(n: int, k: int) -> sympy.core.expr.Expr:
    """Calculate choose function of a set of powers."""
    out = sympy.Integer(1)
    for i in range(k + 1, n + 1):
        out *= i
    for i in range(1, n - k + 1):
        out /= i
    return out


def choose(n: int, powers: typing.List[int]) -> sympy.core.expr.Expr:
    """Calculate choose function of a set of powers."""
    out = sympy.Integer(1)
    for p in powers:
        out *= single_choose(n, p)
        n -= p
    return out


def bernstein_polynomials(
    n: int, d: int
) -> typing.List[sympy.core.expr.Expr]:
    """Return a list of Bernstein polynomials.

    Args:
        n: The polynomial order
        d: The topological dimension
        vars: The variables to use
    """
    poly = []
    if d == 1:
        vars = lambdas[:2]
        powers = [[n - i, i] for i in range(n + 1)]
    elif d == 2:
        vars = lambdas[:3]
        powers = [[n - i - j, j, i]
                  for i in range(n + 1)
                  for j in range(n + 1 - i)]
    elif d == 3:
        vars = lambdas[:4]
        powers = [[n - i - j - k, k, j, i]
                  for i in range(n + 1)
                  for j in range(n + 1 - i)
                  for k in range(n + 1 - i - j)]

    for p in powers:
        f = choose(n, p)
        for a, b in zip(vars, p):
            f *= a ** b
        poly.append(f)

    return poly

n = 2
rule1 = scipy.special.roots_jacobi((n + fdegree) // 2 + 1, 1, 0)
rule2 = scipy.special.roots_jacobi((n + fdegree) // 2 + 1, 0, 0)

q = len(rule1[0])
assert len(rule2) == len(rule1)



poly = bernstein_polynomials(n, 2)

def integral(p, f=1):
    p = p.subs(lambdas[0], 1-x-y).subs(lambdas[1], x).subs(lambdas[2], y)

    return (p*f).integrate([x, 0, 1-y], [y, 0, 1])

integrals = [integral(i, f(x, y)) for i in poly]

rule1 = ((rule1[0] + 1) / 2, rule1[1] / 4)
rule2 = ((rule2[0] + 1) / 2, rule2[1] / 2)

triangle_rule = (
    [(s, r*(1-s)) for s in rule1[0] for r in rule2[0]],
    [a*b for a in rule1[1] for b in rule2[1]]
)

f0 = np.array([
    [f(p1, p2 * (1 - p1)) for p2 in rule2[0]]
    for p1 in rule1[0]])

f1 = np.zeros((n+1, q))
for i1, (p, w) in enumerate(zip(*rule1)):
    s = 1 - p
    r = p / s
    ww = w * s ** n
    for alpha1 in range(n + 1):
        for i2 in range(q):
            f1[alpha1, i2] += ww * f0[i1, i2]
        ww *= r * (n - alpha1) / (1 + alpha1)

f2 = np.zeros((n+1, n+1))
for i2, (p, w) in enumerate(zip(*rule2)):
    s = 1 - p
    r = p / s
    for alpha1 in range(n + 1):
        ww = w * s ** (n - alpha1)
        for alpha2 in range(n + 1 - alpha1):
            f2[alpha1, alpha2] += ww * f1[alpha1, i2]
            ww *= r * (n - alpha1 - alpha2) / (1 + alpha2)


print(f2)


