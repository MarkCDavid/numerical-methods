import unittest
import sympy as sym
from numericalmethods import Interpolation



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
   

if __name__ == '__main__':
    unittest.main()