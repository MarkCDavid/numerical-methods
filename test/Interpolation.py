import unittest
import sympy as sym
import numpy as np
from numericalmethods import Interpolation
from test.cassert import CustomAssertions

class InterpolatingPolynomialTest(unittest.TestCase):

    def test_basis_polynomial(self):
        x = sym.Symbol('x')
        x_values = [1, 2, 3, 4]
        y_values = [1, 2, 3, 4]
        polynomial = Interpolation.InterpolatingPolynomial(x_values, y_values, x)

        self.assertEqual(polynomial.basis_polynomial(1), (x - 1))
        self.assertEqual(polynomial.basis_polynomial(2), (x - 1)*(x - 2))
        self.assertEqual(polynomial.basis_polynomial(3), (x - 1)*(x - 2)*(x - 3))
        self.assertEqual(polynomial.basis_polynomial(4), (x - 1)*(x - 2)*(x - 3)*(x - 4))

        self.assertEqual(polynomial.basis_polynomial(2, offset=1), (x - 2)*(x - 3))
        self.assertEqual(polynomial.basis_polynomial(2, offset=2), (x - 3)*(x - 4))
        self.assertEqual(polynomial.basis_polynomial(3, offset=1), (x - 2)*(x - 3)*(x - 4))

        self.assertEqual(polynomial.basis_polynomial(2, skip=0), (x - 2))
        self.assertEqual(polynomial.basis_polynomial(2, skip=1), (x - 1))
        self.assertEqual(polynomial.basis_polynomial(3, skip=0), (x - 2)*(x - 3))
        self.assertEqual(polynomial.basis_polynomial(3, skip=1), (x - 1)*(x - 3))
        self.assertEqual(polynomial.basis_polynomial(3, skip=2), (x - 1)*(x - 2))

    def test_error(self):
        x = sym.Symbol('x')
        fx = sym.exp(x**2)
        x_values = np.arange(-1, 1.1, 0.2)
        y_values = [fx.subs(x, xi) for xi in x_values]
        N_values = [1, 2, 3, 8, 9, 10]
        r_values = [0.05947, 0.01464, 0.00493, 0.00008, 0.00005, 0.00001]
        at = -0.9

        lip = Interpolation.LagrangeInterpolatingPolynomial(x_values, y_values, x)
        for N, r in zip(N_values, r_values):
            self.assertAlmostEqual(lip.error(fx, N).subs(x, at), r, places=5)

    def test_theoretical_error(self):
        x = sym.Symbol('x')
        fx = sym.exp(x**2)
        x_values = np.arange(-1, 1.1, 0.2)
        y_values = [fx.subs(x, xi) for xi in x_values]
        N_values = [1, 2, 3, 8, 9, 10]
        r_values = [0.05947, 0.01464, 0.00493, 0.00008, 0.00005, 0.00001]
        at = -0.9

        lip = Interpolation.LagrangeInterpolatingPolynomial(x_values, y_values, x)
        for N, r in zip(N_values, r_values):
            self.assertAlmostEqual(lip.error(fx, N).subs(x, at), r, places=5)

    def test_fit(self):
        x = sym.Symbol('x')
        x_values = [-2, -1, 0, 1, 2, 3]
        y_values = [-62, -10, -10, -8, 2, 98]

        fx_fit = x**5 - 2*x**4 + 3*x**2 - 10
        fx_unfit = x**5 - 2*x**4 + 3*x**2 - 20
        polynomial = Interpolation.InterpolatingPolynomial(x_values, y_values, x)

        self.assertTrue(polynomial.fit(fx_fit))
        self.assertFalse(polynomial.fit(fx_unfit))

    def test_fit_points(self):
        x = sym.Symbol('x')
        x_values = [12, 13, 15, 22, 24]
        y_values = [22, 24, 37, 25, 23]
        polynomial = Interpolation.InterpolatingPolynomial(x_values, y_values, x)

        at_x = 6
        degree = 2
        self.assertEqual(polynomial.fit_points(at_x, degree), (12, 13, 15))

class NewtonInterpolatingPolynomialTest(unittest.TestCase, CustomAssertions):

    def test_coefficient(self):
        x = sym.Symbol('x')
        x_values = [0, 1, 3, 4]
        y_values = [3, 2, 1, 0]
        polynomial = Interpolation.NetwonInterpolatingPolynomial(x_values, y_values, x)

        self.assertAlmostEqual(
            polynomial.coefficients(3), 
            [[3, 2, 1, 0],
            [-1.0, -0.5, -1.0],
            [0.16666666666666666, -0.16666666666666666],
            [-0.08333333333333333]]
        )

    def test_polynomial(self):
        x = sym.Symbol('x')
        self.assertSymPyEqual(Interpolation.NetwonInterpolatingPolynomial([1, 2], [-1, 0], x).polynomial(1), x - 2)
        self.assertSymPyEqual(Interpolation.NetwonInterpolatingPolynomial([-2, 0, 1], [0, -2, 0], x).polynomial(2), x**2 + x - 2)
        self.assertSymPyEqual(Interpolation.NetwonInterpolatingPolynomial([-3, -1, 0, 2], [0, 12, 12, 30], x).polynomial(3), x**3 + 2*x**2 + x + 12)


class LagrangeInterpolatingPolynomialTest(unittest.TestCase, CustomAssertions):

    def test_coefficient(self):
        x = sym.Symbol('x')

        self.assertAlmostEqual(Interpolation.LagrangeInterpolatingPolynomial([2, 4, 6, 7], [28, 18, 14, 10], x).coefficient(3, 3).subs(x, 5), -0.2)
        self.assertAlmostEqual(Interpolation.LagrangeInterpolatingPolynomial([15, 18, 22], [24, 37, 25], x).coefficient(2, 2).subs(x, 16), -1.0/14.0)

    def test_polynomial(self):
        x = sym.Symbol('x')
        self.assertSymPyEqual(Interpolation.LagrangeInterpolatingPolynomial([1, 2], [-1, 0], x).polynomial(1), x - 2)
        self.assertSymPyEqual(Interpolation.LagrangeInterpolatingPolynomial([-2, 0, 1], [0, -2, 0], x).polynomial(2), x**2 + x - 2)
        self.assertSymPyEqual(Interpolation.LagrangeInterpolatingPolynomial([-3, -1, 0, 2], [0, 12, 12, 30], x).polynomial(3), x**3 + 2*x**2 + x + 12)

if __name__ == '__main__':
    unittest.main()
