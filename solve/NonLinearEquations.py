import sys
import os
sys.path.append(os.getcwd())

import sympy as sym
from numericalmethods import NonLinearEquations
from numericalmethods import Utility

x = sym.Symbol('x')

print("===== Excercise 1 =====")

fx = x**3 - 27
interval = (2.8, 3.1)
print(NonLinearEquations.MidpointNLE(fx, x, interval).solve(Utility.Precision(10**-2)))


print("===== Excercise 3 =====")

fx = sym.sin(x)/x
interval = (0.5, 1)
print(NonLinearEquations.FixedPointNLE(fx, x, interval, 0.5).solve(Utility.Precision(0.01)))

print("===== Excercise 4 =====")

fx = x**2 - 4
print(NonLinearEquations.NewtonNLE(fx, x, 1.5).solve(Utility.Iteration(3)))

print("===== Excercise 5 =====")

fx = 2*x**3 - 6*x**2 + 4
print(NonLinearEquations.SecantNLE(fx, x, -0.8, -0.7).solve(Utility.Iteration(2)))

print("===== Excercise 7 =====")

fx = x**3 - sym.exp(2/x) - 3
interval = (1.5, 2.0)
print(NonLinearEquations.MidpointNLE(fx, x, interval).solve(Utility.Iteration(2)))
print(NonLinearEquations.MidpointNLE(fx, x, interval).solve(Utility.Precision(10**-3)))

print("===== Excercise 8 =====")

fx = 1/x**3 - sym.exp(2*x)
print(NonLinearEquations.NewtonNLE(fx, x, 0.6).solve(Utility.Precision(0.001)))

print("===== Excercise 9 =====")

Rx = x - x**2 + 5
Tx = 2*x + sym.log(x + 1)
fx = Rx - Tx
print(NonLinearEquations.SecantNLE(fx, x, 1.5,  1.57).solve(Utility.Iteration(2)))

print("===== Excercise 10 =====")

fx = sym.atan(x) + sym.pi

print(NonLinearEquations.FixedPointNLE(fx, x, (sym.pi, 2*sym.pi), 4.49).solve(Utility.Precision(0.01)))


print("===== Excercise 11 =====")

fx = sym.sin(x)

print(NonLinearEquations.SecantNLE(fx, x, 3.14, 3.141).solve(Utility.Precision(10**-6)))


print("===== Excercise 12 =====")

fx = 9*x**3 + 11*x - 4

print(NonLinearEquations.MidpointNLE(fx, x, (0, 0.5)).solve(Utility.Iteration(2)))
print(NonLinearEquations.MidpointNLE(fx, x, (0, 0.5)).solve(Utility.Iteration(8)))


print("===== Excercise 14 =====")

fx = 1/2*(x + 5/x)

print(NonLinearEquations.FixedPointNLE(fx, x, (2, 3), 2).solve(Utility.Precision(10**-4)))

print("===== Excercise 15 =====")

fx = 40*sym.sqrt(x**3) - 875*x + 35000

result = NonLinearEquations.NewtonNLE(fx, x, 60).solve(Utility.Iteration(3))
print(result)

print("===== Excercise 15 =====")

fx = sym.log(x) - 1
print(NonLinearEquations.SecantNLE(fx, x, 2.71, 2.72, constant=True).solve(Utility.Iteration(1)))


