import unittest
import sympy as sym
import numpy as np
from numericalmethods import IterativeLinearEquations, Utility
from test.cassert import CustomAssertions


class NormsTest(unittest.TestCase):

    def test_vector_norms(self):
        X = np.array([1.0, -2.0, 0.0, -1.0, 0.5])

        self.assertAlmostEqual(Utility.Norms.vector_N1(X), 2, places=3)
        self.assertAlmostEqual(Utility.Norms.vector_N2(X), 4.5, places=3)
        self.assertAlmostEqual(Utility.Norms.vector_N3(X), 2.5, places=3)
    
    def test_matrix_norms(self):
        A = np.array([
            [1, 2, -3, 4],
            [2, -1, 4, -3],
            [1, 2, 3, -4],
            [-4, 3, 2, -1],
        ])

        self.assertAlmostEqual(Utility.Norms.matrix_N(A), 2*np.sqrt(30), places=3)
        self.assertAlmostEqual(Utility.Norms.matrix_N1(A), 10, places=3)
        self.assertAlmostEqual(Utility.Norms.matrix_N2(A), 12, places=3)
        self.assertAlmostEqual(Utility.Norms.matrix_N3(A), 8.838, places=3)

class IterativeLinearEquationsTest(unittest.TestCase, CustomAssertions):

    def test_jacobi(self):
        A = np.array([
            [ 1.0, -2.0,  7.0],
            [ 0.9,  0.1, -0.3],
            [-0.1,  0.6,  0.2]
        ])
        f = np.array([-2.0, 1.2, 0.6])
        jacobi = IterativeLinearEquations.JacobiILE(A, f)
        self.assertEqual(jacobi.converges(), False)
        self.assertEqual(jacobi.converges_strict(), False)

        A = np.array([
            [2, 1, 0],
            [1, 2, 1],
            [0, 1, 2]
        ])
        f = np.array([0, 0, 1])
        x0 = np.array([0, 0, 0])
        jacobi = IterativeLinearEquations.JacobiILE(A, f)
        self.assertEqual(jacobi.converges_strict(), True)
        self.assertEqual(np.ceil(jacobi.iterations(10**-5)), 34)
        self.assertListAlmostEqual(jacobi.solve(x0, Utility.Iteration(2))[1], np.array([0, -0.25, 0.5]))

    def test_zeidel(self):
        A = np.array([
            [2, 1, 0],
            [1, 2, 1],
            [0, 1, 2]
        ])
        f = np.array([0, 0, 1])
        x0 = np.array([0, 0, 0])
        zeidel = IterativeLinearEquations.ZeidelILE(A, f)
        self.assertEqual(zeidel.converges_strict(), True)
        self.assertEqual(np.ceil(zeidel.iterations(10**-5)), 17)
        self.assertListAlmostEqual(zeidel.solve(x0, Utility.Precision(0))[1], np.array([ 0.25, -0.5 ,  0.75]))

    def test_fixed_point(self):
        A = np.array([
            [1, 0, 3],
            [0, 1, 3],
            [3, 3, 1]
        ])
        fixed_point = IterativeLinearEquations.FixedPointILE(A, [0, 0, 0])
        self.assertAlmostEqual(fixed_point.tau, 1)
        self.assertEqual(fixed_point.converges(), False)
        self.assertEqual(fixed_point.converges_strict(), False)

        A = np.array([
            [2, 1, 0],
            [1, 2, 1],
            [0, 1, 2]
        ])
        f = np.array([0, 0, 1])
        x0 = np.array([0, 0, 0])
        self.assertListAlmostEqual(IterativeLinearEquations.FixedPointILE(A, f).solve(x0, Utility.Iteration(2))[1],  np.array([0 , -0.25, 0.5]))