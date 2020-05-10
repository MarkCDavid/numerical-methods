import unittest
import sympy as sym
import numpy as np
from numericalmethods import Interpolation, Utility
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


class InterpolatingSplineTest(unittest.TestCase):

    def test_is_spline(self):
        x = sym.Symbol('x')

        functions = [
            x**2, 
            1/9*x**2 + 16/9*x - 8/9, 
            17/75*x**2 + 64/75*x + 24/25
        ]
        points = [0, 1, 4, 9]
        self.assertEqual(
            Interpolation.InterpolatingSpline.is_spline(functions, points, x),
            (True, 2)
        )

        functions = [
            x**3 - x**2 + 2*x - 1, 
            -x**2 + 2*x - 1, 
            x**3 - 4*x**2 + 5*x - 2,
            x**3 + x**2 - 25*x + 43
        ]
        points = [-2, 0, 1, 3, 4]
        self.assertEqual(
            Interpolation.InterpolatingSpline.is_spline(functions, points, x),
            (False, 2)
        )

        functions = [
            4*x**2 + 2,
            3*x + 10,
            x - 1
        ]
        points = [0, 1, 4, 7]
        self.assertEqual(
            Interpolation.InterpolatingSpline.is_spline(functions, points, x),
            (False, 0)
        )

        functions = [
            -2.25*x**2 + 9*x + 99,
            3.25*x**2 - 57*x + 297,
            -3.25*x**2 + 73*x - 353,
            3.5*x**2 - 116*x + 970,
            -3.25*x**2 + 127*x - 1217,
            3.875*x**2 - 186.5*x + 2231.5,
        ]
        points = [2, 6, 10, 14, 18, 22, 26]
        self.assertEqual(
            Interpolation.InterpolatingSpline.is_spline(functions, points, x, places=4),
            (True, 2)
        )

class SpecificInterpolatingSplineTest(unittest.TestCase, CustomAssertions):

    def test_spline(self):
        x = sym.Symbol('x')
        x_values = [2, 6, 10, 14, 18, 22, 26]
        y_values = [108, 72, 52, 32, 16, 4, 2]


        expected = [
            (126 - 9*x, Utility.interval(2, 6, x)),
            (102 - 5*x, Utility.interval(6, 10, x)),
            (102 - 5*x, Utility.interval(10, 14, x)),
            (88 - 4*x, Utility.interval(14, 18, x)),
            (70 - 3*x, Utility.interval(18, 22, x)),
            (15 - 0.5*x, Utility.interval(22, 26, x))
        ]
        result = Interpolation.LinearInterpolatingSpline(x_values, y_values, x).spline()
        self.assertSymPyListAlmostEqual(
            [x[0] for x in expected],
            [x[0] for x in result]
        )
        self.assertListEqual(
            [x[1] for x in expected],
            [x[1] for x in result]
        )

        expected = [
            (-2.25*x**2 + 9*x + 99, Utility.interval(2, 6, x)),
            (3.25*x**2 - 57*x + 297, Utility.interval(6, 10, x)),
            (-3.25*x**2 + 73*x - 353, Utility.interval(10, 14, x)),
            (3.5*x**2 - 116*x + 970, Utility.interval(14, 18, x)),
            (-3.25*x**2 + 127*x - 1217, Utility.interval(18, 22, x)),
            (3.875*x**2 - 186.5*x + 2231.5, Utility.interval(22, 26, x)),
        ]
        result = Interpolation.SquareInterpolatingSpline(x_values, y_values, x).spline()
        self.assertSymPyListAlmostEqual(
            [x[0] for x in expected],
            [x[0] for x in result]
        )
        self.assertListEqual(
            [x[1] for x in expected],
            [x[1] for x in result]
        )

        expected = [
            (0.0681*x**3 - 0.4084*x**2 - 9.2723*x + 127.6337, Utility.interval(2, 6, x)),
            (-0.0903*x**3 + 2.4430*x**2 - 26.3809*x + 161.8510, Utility.interval(6, 10, x)),
            (0.0433*x**3 - 1.5666*x**2 + 13.7152*x + 28.1971, Utility.interval(10, 14, x)),
            (-0.0204*x**3 + 1.1089*x**2 - 23.7415*x + 202.9952, Utility.interval(14, 18, x)),
            (0.0383*x**3 - 2.0584*x**2 + 33.2700*x - 139.0740, Utility.interval(18, 22, x)),
            (-0.0389*x**3 + 3.0344*x**2 - 78.7713*x + 682.5625, Utility.interval(22, 26, x)),
        ]
        result = Interpolation.CubicInterpolatingSpline(x_values, y_values, x).spline()
        self.assertSymPyListAlmostEqual(
            [x[0] for x in expected],
            [x[0] for x in result]
        )
        self.assertListEqual(
            [x[1] for x in expected],
            [x[1] for x in result]
        )

if __name__ == '__main__':
    unittest.main()
