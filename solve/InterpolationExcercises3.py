import sys
import os
sys.path.append(os.getcwd())

import sympy as sym
from numericalmethods import Interpolation

x = sym.Symbol('x')

print("===== Excercise 1 =====")

x_values = [-2, -1, 0, 1, 2]
y_values = [2, -1, -2, -1, 2]

s1 = Interpolation.LinearInterpolatingSpline(x_values, y_values, x).spline()
print(f"s1: {s1}")

print("===== Excercise 2 =====")

x_values = [-1, 2, 3]
y_values = [-4, 5, 24]

s2 = Interpolation.SquareInterpolatingSpline(x_values, y_values, x).spline()
print(f"s2: {s2}")

print("===== Excercise 3 =====")

print(Interpolation.InterpolatingSpline.is_spline(
    [4*x**2 + 2, 3*x + 10, x-1], [0, 1, 4, 7], x
))

print(Interpolation.InterpolatingSpline.is_spline(
    [-2/5*x**3 + 6/5*x**2 - 19/5*x + 5, -6/5*x**2 + x + 9/5, 2/5*x**3 - 24/5*x**2 + 59/5*x - 9], [1, 2, 3, 4], x
))


print("===== Excercise 4 =====")

f0 = -2/5*x**3 + 6/5*x**2 - 19/5*x + 5
f3 = 2/5*x**3 - 24/5*x**2 + 59/5*x - 9

print(f"y0: {f0.subs(x, 1)} | y3: {f3.subs(x, 4)}")

print("===== Excercise 5 =====")

a, b = sym.symbols('a b')

fa = a*x**3 + 6/5*x**2 - 19/5*x + 5
fb = -6/5*x**2 + b*x + 9/5

b_value = sym.solve(fb.subs(x, 2) + 1, b)[0]
a_value = sym.solve(fa.subs(x, 1) - 2, a)[0]

print(f"a: {a_value}, b: {b_value}")
