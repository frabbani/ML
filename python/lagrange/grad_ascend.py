import fun

f = fun.f
g = fun.g
df_dx = fun.df_dx
dg_dx = fun.dg_dx


def L(x, lam):
    return f(x) + lam * g(x)


def dL_dx(x, lam):
    # in terms of maximizing you add, minimize you subtract
    return df_dx(x) + lam * dg_dx(x)
    #    return (L(x + H, lam) - L(x - H, lam)) / (2 * H)


def solve():
    x = 5
    lam = 0
    x_hist = [x]
    learning_rate = 0.1
    for k in range(1, 1000):
        if abs(g(x)) < 1e-3:
            print(f"found after {k} steps!")
            break
        x += learning_rate * dL_dx(x, lam)
        x_hist.append(x)
        lam -= learning_rate * g(x)
    return [x, lam, x_hist]


c, l, x_hist = solve()
print(f"x={c:.4f}, lambda={l:.04f}")

fun.plot(c, x_hist)

