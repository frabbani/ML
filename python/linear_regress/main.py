import random
import math
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression



class Line:  # y = mx + b
    def __init__(self, m, b):
        self.m = m
        self.b = b
        return

    def norm(self):
        return [-self.m, 1]

    def y(self, x_):
        return self.m * x_ + self.b

    def random_point(self, x_, dist_):
        dist_ = dist_ * random.uniform(-1, +1)
        n = self.norm()
        return [x_ + n[0] * dist_, self.y(x_) + n[1] * dist_]

    def gen_points(self, count, offs_, dist_):
        points_ = []
        for i in range(1, count):
            points_.append(self.random_point(i * offs_, dist_))
        return points_

    def __str__(self):
        if self.b < 0:
            return f"y = {self.m:.3f}x - {-self.b:.3f}"
        if self.b > 0:
            return f"y = {self.m:.3f}x + {self.b:.3f}"
        return f"y = {self.m:.3f}x"


def regress(points_):
    sum_x = 0.0
    sum_y = 0.0
    sum_xy = 0.0
    sum_xsq = 0.0
    for p in points_:
        sum_x += p[0]
        sum_y += p[1]
        sum_xy += p[0] * p[1]
        sum_xsq += p[0] * p[0]
    n = float(len(points_))
    m = (n * sum_xy - sum_x * sum_y) / (n * sum_xsq - sum_x * sum_x)
    b = (sum_y - m * sum_x) / n
    return Line(m, b)


def scikit_regress(points_):
    xs = [[p[0]] for p in points_]
    ys = [[p[1]] for p in points_]
    model = LinearRegression()
    model.fit(xs, ys)
    return Line(float(model.coef_[0].item()), float(model.intercept_[0]))


line = Line(0.233, -15.7)
points = line.gen_points(50, 5, 15)

line_reg = regress(points)
line_reg_sk = scikit_regress(points)

plt.figure(figsize=(10, 6))
x_values = [xy[0] for xy in points]
y_values = [xy[1] for xy in points]
plt.scatter(x_values, y_values, label='samples', color='red')

x_max = 50 * 5
x_plot = [0]
for x in range(0, 275):
    x_plot.append(x)
plt.xlim(0, 250)

plt.plot(x_plot, [line.y(xi) for xi in x_plot], linestyle='--', label=f"source.: " + str(line), color='#BDAA00')
plt.plot(x_plot, [line_reg.y(xi) for xi in x_plot], label=f"custom: " + str(line_reg), color='green')
#plt.plot(x_plot, [line_reg_sk.y(xi) for xi in x_plot], label=f"scikit....: " + str(line_reg_sk), color='blue')

plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid()
plt.show()
print("goodbye!")

