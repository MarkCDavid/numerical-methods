import unittest
import sympy as sym
import numpy as np
from numericalmethods import NonLinearEquations
from numericalmethods import Utility
from test.cassert import CustomAssertions


class NonLinearEquationsTest(unittest.TestCase):

    def test_root_existance(self):
        x = sym.Symbol('x')
        self.assertEqual(NonLinearEquations.NonLinearEquations(2*x + 3, x).root_parity((-2, -1)), Utility.Parity.ODD)
        self.assertEqual(NonLinearEquations.NonLinearEquations(x**2 + 4 * x - 5, x).one_root((-6, -4)), True)
        self.assertEqual(NonLinearEquations.NonLinearEquations(x**2 + 4 * x - 5, x).root_parity((-6, 6)), Utility.Parity.EVEN)
        self.assertEqual(NonLinearEquations.NonLinearEquations(x, x).one_root((1, 2)), False)
        self.assertEqual(NonLinearEquations.NonLinearEquations(sym.exp(x) - x**2 + 4, x).one_root((-3, -2)), True)
        self.assertEqual(NonLinearEquations.NonLinearEquations(sym.exp(3*x) - 2 - 2 * sym.sin(x), x).root_parity((-3, 1)), Utility.Parity.ODD)
        self.assertEqual(NonLinearEquations.NonLinearEquations(sym.exp(3*x) - 2 - 2 * sym.sin(x), x).one_root((-3, 1)), False)

    def test_middle_point(self):
        x = sym.Symbol('x') 
        self.assertAlmostEqual(NonLinearEquations.MidpointNLE(x * sym.sin(x) - 1, x, (0.0, sym.pi/2.0)).solve(Utility.Precision(10**-2))[0], 1.111, places=3)
        self.assertAlmostEqual(NonLinearEquations.MidpointNLE(x * sym.sin(x) - 1, x, (0.0, sym.pi/2.0)).converges_in(10**-2), 6.295, places=3)

    def test_fixed_point(self):
        x = sym.Symbol('x')
        self.assertAlmostEqual(NonLinearEquations.FixedPointNLE(sym.atan(x) + sym.pi, x, (sym.pi, 3*sym.pi/2), 4).solve(Utility.Precision(10**-3))[0], 4.4934, places=4)

    def test_newton_method(self):
        x = sym.Symbol('x') 
        self.assertAlmostEqual(NonLinearEquations.NewtonNLE(x * sym.sin(x) - 1, x, 0.5).solve(Utility.Precision(10**-6))[0], 1.114157, places=6)
        self.assertAlmostEqual(NonLinearEquations.NewtonNLE(x * sym.sin(x) - 1, x, 0.5, constant=True).solve(Utility.Precision(10**-6))[0], 1.114157, places=6)

    def test_secant_method(self):
        x = sym.Symbol('x')
        self.assertAlmostEqual(NonLinearEquations.SecantNLE(x**4 + 2*x**3 - x - 1, x, -3, -2).solve(Utility.Precision(10**-4))[0], -1.8667, places=3)
        self.assertAlmostEqual(NonLinearEquations.SecantNLE(x**4 + 2*x**3 - x - 1, x, -3, -2, constant=True).solve(Utility.Precision(10**-4))[0], -1.8667, places=3)
        self.assertAlmostEqual(NonLinearEquations.SecantNLE(x**4 + 2*x**3 - x - 1, x, 0.6, 0.7).solve(Utility.Precision(10**-4))[0], 0.8667, places=3)
        self.assertAlmostEqual(NonLinearEquations.SecantNLE(x**4 + 2*x**3 - x - 1, x, 0.6, 0.7, constant=True).solve(Utility.Precision(10**-4))[0], 0.8667, places=3)
        


