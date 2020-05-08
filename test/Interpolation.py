import unittest
import sympy as sym
from numericalmethods import Interpolation
from test.cassert import CustomAssertions

class InterpolatingPolynomialTest(unittest.TestCase):

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

    def test_basis_polynomial(self):
        x = sym.Symbol('x')
        x_values = [1, 2, 3, 4]
        y_values = [1, 2, 3, 4]
        polynomial = Interpolation.NetwonInterpolatingPolynomial(x_values, y_values, x)

        self.assertEqual(polynomial.basis_polynomial(1), (x - 1))
        self.assertEqual(polynomial.basis_polynomial(2), (x - 1)*(x - 2))
        self.assertEqual(polynomial.basis_polynomial(3), (x - 1)*(x - 2)*(x - 3))
        self.assertEqual(polynomial.basis_polynomial(4), (x - 1)*(x - 2)*(x - 3)*(x - 4))

    def test_polynomial(self):
        x = sym.Symbol('x')
        self.assertSymPyEqual(Interpolation.NetwonInterpolatingPolynomial([1, 2], [-1, 0], x).polynomial(1), x - 2)
        self.assertSymPyEqual(Interpolation.NetwonInterpolatingPolynomial([-2, 0, 1], [0, -2, 0], x).polynomial(2), x**2 + x - 2)
        self.assertSymPyEqual(Interpolation.NetwonInterpolatingPolynomial([-3, -1, 0, 2], [0, 12, 12, 30], x).polynomial(3), x**3 + 2*x**2 + x + 12)

if __name__ == '__main__':
    unittest.main()