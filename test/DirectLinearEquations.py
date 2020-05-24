import unittest
import sympy as sym
import numpy as np
from numericalmethods import DirectLinearEquations, Utility
from test.cassert import CustomAssertions

class DirectLinearEquationsTests(unittest.TestCase, CustomAssertions):

    def test_gauss_method(self):
        A = [[1, -2, 3, 1],
            [3, -6, 9, 4],
            [2, -3, 8, 5],
            [-4, 8, -13, -8]]
        f = [2, 8, 14, -17]
        self.assertListAlmostEqual(DirectLinearEquations.GaussMethod(A, f).solve(), [1.0, 2.0, 1.0, 2.0])

    def test_relocation_method(self):
        tridiagonal_matrix = Utility.TridiagonalMatrix(
            [1, 1, 1], 
            [1, 3, -3, 2],
            [1, -0.5, -1],
            [0.5, 1, 2, 2]
        )

        relocation_dle = DirectLinearEquations.RelocationMethod(tridiagonal_matrix)
        
        self.assertListAlmostEqual(relocation_dle.solve(), [0.5556, -0.0556, -1.2222, 1.6111], places=4)
        self.assertListAlmostEqual([a[1] for a in relocation_dle.alpha()], [-1.0, 0.25, -0.363636], places=4)
        self.assertListAlmostEqual([b[1] for b in relocation_dle.beta()], [0.5, 0.25, -0.636364, 1.6111], places=4)

    def test_cholesky_method(self):
        A = [[4, 2.4, 0.8],
            [2.4, 3, -0.15],
            [0.8, -0.15, 4]]

        f = [5, 7, 0]

        self.assertListAlmostEqual(DirectLinearEquations.CholeskyMethod(A, f).solve(), [-0.364, 2.633, 0.171], places=3)
