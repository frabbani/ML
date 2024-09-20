from scipy.optimize import fsolve

import fun

f = fun.f
g = fun.g0
df_dx = fun.df_dx
dg_dx = fun.dg_dx
H = fun.H
C = fun.C


def L(x, lam):
    return f(x) + lam * g(x)


def dL_dx(x, lam):
    return (L(x + H, lam) - L(x - H, lam)) * 0.5 / H
    # return df_dx(x) + lam * dg_dx(x)


# Define a function to find critical points
def find_critical():
    def equations(vars):
        x, lam = vars
        return [L(x, lam), g(x) - C]

    # Initial guess
    return fsolve(equations, [5, 0])


c, l = find_critical()
print(f"x={c:.4f}, lambda={l:.04f}")

fun.plot(c)
