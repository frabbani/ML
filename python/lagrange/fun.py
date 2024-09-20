import numpy as np
import matplotlib.pyplot as plt

H = 0.01
C = 1.5


def f(x):
    o = x - 2
    return 1 - 0.5 * o ** 2


def g0(x):
    o = x - 1
    return o ** 2 + 0.3


def g(x):
    return g0(x) - C


def df_dx(x):
    return (f(x + H) - f(x - H)) * 0.5 / H


def dg_dx(x):
    return (g(x + H) - g(x - H)) * 0.5 / H


def plot(x_crit, x_hist=None):
    # Create x values for plotting
    x_values = np.linspace(-15, 15, 100)
    f_values = f(x_values)
    g_values = g0(x_values)

    # Plot f(x) and g(x)
    plt.figure(figsize=(10, 6))
    plt.plot(x_values, f_values, label='f(x)', color='blue')
    plt.plot(x_values, g_values, label='g(x)', color='orange')

    plt.xlim(-10, 10)
    plt.ylim(-10, 10)

    # Highlight critical points on the plot

    plt.plot(x_crit, f(x_crit), 'ro')  # Red points for critical points
    plt.plot(x_crit, g0(x_crit), 'go')  # Red points for critical points
    plt.annotate(f'({x_crit:.2f}, {f(x_crit):.2f})', (x_crit, f(x_crit)), textcoords="offset points", xytext=(0, 10),
                 ha='center')
    if x_hist is not None:
        plt.plot(x_hist, [f(xi) for xi in x_hist], label="path of x", color="green", alpha=0.9)
    plt.plot(x_values, [1.5 for xi in x_values], label="constraint", color="blue", alpha=0.25)

    # Set plot limits and labels
    plt.axhline(0, color='black', linewidth=0.5, ls='--')
    plt.axvline(0, color='black', linewidth=0.5, ls='--')
    plt.title('Plot of f(x) and g(x) with Critical Points')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid()
    plt.show()
