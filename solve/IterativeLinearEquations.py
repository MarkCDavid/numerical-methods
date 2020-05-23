import sys
import os
sys.path.append(os.getcwd())

import sympy as sym
import numpy as np
from numericalmethods import IterativeLinearEquations
from numericalmethods import Utility

x = sym.Symbol('x')

print("===== Excercise 1 =====")

X = np.array([1, -2, 0, -1, 0.5])

print(Utility.Norms.vector_N1(X))
print(Utility.Norms.vector_N2(X))
print(Utility.Norms.vector_N3(X))

print("===== Excercise 3 =====")

A = np.array([
    [1, 2, -3, 4],
    [2, -1, 4, -3],
    [1, 2, 3, -4],
    [-4, 3, 2, -1]
])

print(Utility.Norms.matrix_N(A))
print(Utility.Norms.matrix_N1(A))
print(Utility.Norms.matrix_N2(A))
print(Utility.Norms.matrix_N3(A))


print("===== Excercise 6 =====")

A = np.array([
    [ 1.0, -2.0,  7.0],
    [ 0.9,  0.1, -0.3],
    [-0.1,  0.6,  0.2]
])
f = np.array([-2.0, 1.2, 0.6])
jacobi = IterativeLinearEquations.JacobiILE(A, f)
print(jacobi.converges())
print(jacobi.converges_strict())


print("===== Excercise 7 =====")

A = np.array([
    [2, 1, 0],
    [1, 2, 1],
    [0, 1, 2]
])
f = np.array([0, 0, 1])
x0 = np.array([0, 0, 0])
jacobi = IterativeLinearEquations.JacobiILE(A, f)
print(jacobi.converges_strict())
print(np.ceil(jacobi.iterations(10**-5)))

print("===== Excercise 8 =====")

print(jacobi.solve(x0, Utility.Iteration(2))[1])


print("===== Excercise 9 =====")
zeidel = IterativeLinearEquations.ZeidelILE(A, f)
print(zeidel.converges_strict())
print(np.ceil(zeidel.iterations(10**-5)))

print("===== Excercise 11 =====")

A = np.array([
    [1, 0, 3],
    [0, 1, 3],
    [3, 3, 1]
])

fp = IterativeLinearEquations.FixedPointILE(A, [0, 0, 0])
print(fp.tau)
print(fp.converges())
print(fp.converges_strict())

print("===== Excercise 12 =====")

A = np.array([
    [2, 1, 0],
    [1, 2, 1],
    [0, 1, 2]
])
f = np.array([0, 0, 1])
x0 = np.array([0, 0, 0])
fp = IterativeLinearEquations.FixedPointILE(A, f)
print(fp.solve(x0, Utility.Iteration(2)))