import sys
import os
sys.path.append(os.getcwd())

import sympy as sym
from numericalmethods import Interpolation

x = sym.Symbol('x')

print("===== Excercise 1 =====")

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

lip = Interpolation.NewtonInterpolatingPolynomial(x_values, y_values, x)
newtonDifferences = lip.coefficients(3)

lpoly = lip.polynomial(1)
cpoly = lip.polynomial(3)

at_x = 2130

print(f"Value of linear IP({at_x}): {lpoly.subs(x, at_x)}")
print(f"Value of cubic IP({at_x}): {cpoly.subs(x, at_x)}")

inverse_lip = Interpolation.NewtonInterpolatingPolynomial(x_values, y_values, x, inverse=True)
print(cpoly)
print(inverse_lip.polynomial(1))
