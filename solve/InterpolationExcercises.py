import sys
import os
sys.path.append(os.getcwd())

import sympy as sym
from numericalmethods import Interpolation

x = sym.Symbol('x')

x_values = [0, 1, 2, 4]
y_values = [0, 1, 4, 16]
at_x = 0.5

lip = Interpolation.LagrangeInterpolatingPolynomial(x_values, y_values, x)

lpoly = lip.polynomial(1)
spoly = lip.polynomial(2)
print(f"Linear Interpolating Polynomial (LIP): {lpoly}")
print(f"Square Interpolating Polynomial (SIP): {spoly}")
print(f"Value of LIP({at_x}): {lpoly.subs(x, at_x)}")
print(f"Value of SIP({at_x}): {spoly.subs(x, at_x)}")
print(f"Error of LIP({at_x}): {lip.practical_error_polynomial_difference(1).subs(x, at_x)}")
print(f"Error of SIP({at_x}): {lip.practical_error_polynomial_difference(2).subs(x, at_x)}")

print("===== Excercise 2 =====")

x_values = [1, 4, 8]
y_values = [0.106, 0.100, 0.097]

lip = Interpolation.NewtonInterpolatingPolynomial(x_values, y_values, x)
at_x = 2

lpoly = lip.polynomial(1)
print(f"Value of polynomial({at_x}): {lpoly.subs(x, at_x)}")
print(f"Error of polynomial({at_x}): {lip.practical_error_polynomial_difference(1).subs(x, at_x)}")

print("===== Excercise 3 =====")

x_values = [0.2, 0.5, 1.0]
y1_values = [125, 8, 1]
y2_values = [1.22, 1.65, 2.72]
y_values = [y1 - y2 for y1, y2 in zip(y1_values, y2_values)]

lip = Interpolation.LagrangeInterpolatingPolynomial(x_values, y_values, x, inverse=True)
at_y = 0

lpoly = lip.polynomial(1)
print(f"Value of polynomial({at_y}): {lpoly.subs(x, at_y)}")

print("===== Excercise 4 =====")

x_values = [-2, -1, 0, 1, 2, 3]
y_values = [-62, -10, -10, -8,  2, 98]

poly =  x**5 - 2*x**4 + 3*x**2 - 10

print(f"Fits? {Interpolation.InterpolatingPolynomial(x_values, y_values, x).fit(poly)}")

print("===== Excercise 5 =====")

x_values = [12, 13, 15, 22, 24]
y_values = [22, 24, 37, 25, 23]

print(f"Fit at {Interpolation.InterpolatingPolynomial(x_values, y_values, x).fit_points(14, 2)}")

print("===== Excercise 6 =====")

x_values = [2050, 2100, 2150, 2200, 2250]
y_values = [107, 110, 108, 70, 53]

lip = Interpolation.LagrangeInterpolatingPolynomial(x_values, y_values, x)

lpoly = lip.polynomial(1)
cpoly = lip.polynomial(3)

at_x = 2130

print(f"Value of linear IP({at_x}): {lpoly.subs(x, at_x)}")
print(f"Value of cubic IP({at_x}): {cpoly.subs(x, at_x)}")

at_y = 109

inverse_lip = Interpolation.NewtonInterpolatingPolynomial(x_values, y_values, x, inverse=True)
print(f"Value of inverse linear IP({at_y}): {inverse_lip.polynomial(1, offset=3).subs(x, at_y)}")

print("===== Excercise 7 =====")

fx = 1/x**5
at_x = 1.5

x_values = range(1, 6)
y_values = [fx.subs(x, x_value) for x_value in x_values]

ip = Interpolation.InterpolatingPolynomial(x_values, y_values, x)

for degree in range(1, 5):
    print(f"L{degree} error at {at_x} : {ip.theoretical_error(fx, degree, 1, 5).subs(x, at_x)}")

print("===== Excercise 8 =====")

x_values = [15, 18, 22]
y_values = [24, 37, 25]

lip = Interpolation.LagrangeInterpolatingPolynomial(x_values, y_values, x)
degree = 2
index = 2
at_x = 16
print(f"c{index}({at_x}) = {lip.coefficient(degree, index).subs(x, at_x)}")
