

import fun

f = fun.f
g = fun.g
g0 = fun.g0
df_dx = fun.df_dx
dg_dx = fun.dg_dx

def dL_dx(x, penalty):
    return df_dx(x) + penalty * g(x) * dg_dx(x)


GRAD_TOL = 0.1  # gradient tolerance
PERTURBATION = 0.3  # jitter amount
LEARNING_RATE = 0.001

def solve_penalty():
    x = 5  # initial guess
    learning_rate = 0.01
    x_hist = [x]
    for i in range(1, 5000):
        x -= learning_rate * dL_dx(x, 3.0)
        x_hist.append(x)
        if abs(g(x)) < 1e-3:
            print(f"converged after {i} steps! (g(x) = {g0(x)})")
            break
    return [x, x_hist]


[x, x_hist] = solve_penalty()
print(f"x={x:.5f}, g(x)={g(x):.5f}, g0(x)={g0(x):.5f}")

fun.plot(x, x_hist)
