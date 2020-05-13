import sys
import os
sys.path.append(os.getcwd())

import sympy as sym
from numericalmethods import Interpolation

x = sym.Symbol('x')

print("===== Excercise 1 =====")

x_values = [*sym.symbols('x0 x1 x2')]
y_values = [*sym.symbols('f(x0) f(x1) f(x2)')]

differences = Interpolation.NewtonDifferences(x_values, y_values)

print(f"f[x0, x1, x2] = {differences.coefficient(2, 0)}")

print("===== Excercise 2 =====")

x_values = [*sym.symbols('x4, x5 x6 x7')]
y_values = [*sym.symbols('y4, y5 y6 y7')]

L2 = Interpolation.NewtonInterpolatingPolynomial(x_values, y_values, x).polynomial(2)
c5 = Interpolation.LagrangeInterpolatingPolynomial(x_values, y_values, x).coefficient(3, 1)
print(f"L2 = {L2}")
print(f"c5 = {c5}")

print("===== Excercise 3 =====")
print("SKIP - THEORY QUESTION")

print("===== Excercise 4 =====")
x_values = [-1, 0, 1]
y_values = [-2, -2, 0]
print(sym.expand(Interpolation.LagrangeInterpolatingPolynomial(x_values, y_values, x).polynomial(2)))

print("===== Excercise 5 =====")
x_values = [-3, -2, -1, 0, 1, 2]
y_values = [10, 21, 4, 1, 6, 85]
L5 = x**5+3*x**4+x**2+1
print(f"Fits? {Interpolation.InterpolatingPolynomial(x_values, y_values, x).fit(L5)}")

print("===== Excercise 6 =====")
x_values = [2, 4, 6, 7]
y_values = [28, 18, 14, 10]
at_x = 5
print(f"c3({at_x}): {Interpolation.LagrangeInterpolatingPolynomial(x_values, y_values, x).coefficient(3, 3).subs(x, at_x)}")

print("===== Excercise 7 =====")
L4 = x**3 - 2*x**2 + x - 1
print(f"y3: {L4.subs(x, -1)}")

print("===== Excercise 8 =====")
print("SKIP - THEORY QUESTION")

print("===== Excercise 9 =====")
x_values = [1, 3, 4, 5]
y_values = [0.230, 0.228, 0.226, 0.222]
at_x = 2
nip = Interpolation.NewtonInterpolatingPolynomial(x_values, y_values, x)
print(f"L1({at_x}): {nip.polynomial(2).subs(x, at_x)}")
print(f"|r1({at_x})|: {nip.practical_error_polynomial_difference(1).subs(x, at_x)}")

print("===== Excercise 10 =====")
result = Interpolation.InterpolatingSpline.is_spline([4*x**2 + 2, 3*x + 10, x - 1], [0, 1, 4, 7], x)
print(f"Is spline: {result[0]}. {f'Of degree: {result[1]}' if result[0] else ''}")

print("===== Excercise 11 =====")
result = Interpolation.InterpolatingSpline.is_spline(
    [(3.0/7.0)*x**3 + (18.0/7.0)*x**2 + (12.0/7.0)*x + (4.0/7.0), (-1.0/7.0)*x**3 + (6.0/7.0)*x**2, (1.0/7.0)*x**3 + (6.0/7.0)*x**2, (-3.0/7.0)*x**3 + (18.0/7.0)*x**2-(12.0/7.0)*x+(4.0/7.0)],
    [-2.0, -1.0, 0.0, 1.0, 2.0],
    x,
    degree=3,
)
print(result)
print(f"Is spline of degree 3: {result[0]}.")

print("===== Excercise 12 =====")

a, b = sym.symbols('a b')

fa = a*x**3 + 6/5*x**2 + 19/5*x + 5
fb = -6/5*x**2 + b*x + 9/5

b_value = sym.solve(fb.subs(x, 2) + 1, b)[0]
a_value = sym.solve(fa.subs(x, 2).subs(b, b_value) + 1, a)[0]

y0 = fa.subs(a, a_value).subs(x, 1)
print(f"a: {a_value} | y0: {y0}")


print("===== Excercise 13 =====")

x_values = [2, 6, 7]
y_values = [0.188, 0.169, 0.165]

at_x = 3
actual = 0.177+x*0

lip = Interpolation.LagrangeInterpolatingPolynomial(x_values, y_values, x)
print(f"L1({at_x}): {lip.polynomial(1).subs(x, at_x)}")
print(f"|L1({at_x}) - y*|: {sym.Abs(lip.polynomial(1).subs(x, at_x) - actual)}")