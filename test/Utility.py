import unittest
import sympy as sym
import numpy as np
from numericalmethods import Utility
from test.cassert import CustomAssertions

class UtilityTest(unittest.TestCase):

    def test_value_sampling(self):
        x = sym.Symbol('x')
        self.assertAlmostEqual(Utility.value_sampling(x, x, 0, 5, step_size=1), [0, 1, 2, 3, 4, 5])
        self.assertAlmostEqual(Utility.value_sampling(x**2, x, 0, 5, step_size=1), [0, 1, 4, 9, 16, 25])

    def test_maximum_absolute_value(self):
        x = sym.Symbol('x')
        self.assertAlmostEqual(Utility.maximum_absolute_value(x, x, 0, 5, 0), 5)
        self.assertAlmostEqual(Utility.maximum_absolute_value(x**2, x, 0, 5, 0), 25)
        self.assertAlmostEqual(Utility.maximum_absolute_value(1/(1+x**2), x, -1, 1, 1, step_size=0.01), 0.6495, places=4)
        self.assertAlmostEqual(Utility.maximum_absolute_value(1/(1+x**2), x, -1, 1, 2, step_size=0.01), 2.0, places=4)
        self.assertAlmostEqual(Utility.maximum_absolute_value(1/(1-x**3), x, -2, 0, 1, step_size=0.01), 0.8399, places=4)
        self.assertAlmostEqual(Utility.maximum_absolute_value(1/(1-x**3), x, -2, 0, 2, step_size=0.01), 1.7374, places=4)


if __name__ == '__main__':
    unittest.main()
