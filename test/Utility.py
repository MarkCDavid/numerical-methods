import unittest
import sympy as sym
import numpy as np
import operator
from numericalmethods import Utility
from test.cassert import CustomAssertions

class UtilityTest(unittest.TestCase):

    def test_value_sampling(self):
        x = sym.Symbol('x')
        interval = (0, 5)
        self.assertAlmostEqual(Utility.value_sampling(x, x, interval, step_size=1), [0, 1, 2, 3, 4, 5])
        self.assertAlmostEqual(Utility.value_sampling(x**2, x, interval, step_size=1), [0, 1, 4, 9, 16, 25])

    def test_maximum_absolute_value(self):
        x = sym.Symbol('x')
        interval = (0, 5)
        self.assertAlmostEqual(Utility.maximum_absolute_value(x, x, interval, 0), 5)
        self.assertAlmostEqual(Utility.maximum_absolute_value(x**2, x, interval, 0), 25)
        interval = (-1, 1)
        self.assertAlmostEqual(Utility.maximum_absolute_value(1/(1+x**2), x, interval, 1, step_size=0.01), 0.6495, places=4)
        self.assertAlmostEqual(Utility.maximum_absolute_value(1/(1+x**2), x, interval, 2, step_size=0.01), 2.0, places=4)
        interval = (-2, 0)
        self.assertAlmostEqual(Utility.maximum_absolute_value(1/(1-x**3), x,interval, 1, step_size=0.01), 0.8399, places=4)
        self.assertAlmostEqual(Utility.maximum_absolute_value(1/(1-x**3), x, interval, 2, step_size=0.01), 1.7374, places=4)

    def test_triangle_array(self):
        self.assertListEqual(Utility.triangle_array(3), [[None, None, None], [None, None], [None]])
        self.assertListEqual(Utility.triangle_array(4, default_value=5), [[5, 5, 5, 5],[5, 5, 5], [5, 5], [5]])

    def test_gap(self):
        self.assertEqual(Utility.gap([0, 2, 5], 1), 3)
        self.assertEqual(Utility.gap([3, 6, 7, 2, 5 ,1], 4), -4)

    def invert_data(self):
        self.assertEqual(Utility.invert_data([0, 1, 2], [5, 4, 3]), ([3, 4, 5], [2, 1, 0]))
        self.assertEqual(Utility.invert_data([0, 1, 2], [5, 4, 3], key=operator.itemgetter(0)), ([0, 1, 2], [5, 4, 3]))

    def test_memoized(self):

        class FailSecondCall:

            def __init__(self, function):
                self.function = function
                self.calls = {}
            
            def __call__(self, *args):
                if args not in self.calls:
                    self.calls[args] = 0
                self.calls[args] += 1
                if self.calls[args] > 1:
                    raise AssertionError("Multiple calls")
                return self.function(*args)
                

        square = lambda x: x**2
        memoizedFunction = Utility.Memoized(FailSecondCall(square))
        memoizedFunction(5)
        memoizedFunction(5)
        memoizedFunction(6)
        memoizedFunction(6)

        nonMemoizedFunction = FailSecondCall(square)
        nonMemoizedFunction(5)
        with self.assertRaises(AssertionError):
            nonMemoizedFunction(5)
        
        nonMemoizedFunction(6)
        with self.assertRaises(AssertionError):
            nonMemoizedFunction(6)

        outer = lambda x: x[0]**2 - x[1](x[0])
        inner = lambda x: x/2
        memoizedFunction = Utility.Memoized(FailSecondCall(outer))
        memoizedFunction((5, inner))
        memoizedFunction((5, inner))
        memoizedFunction((6, inner))
        memoizedFunction((6, inner))

        nonMemoizedFunction = FailSecondCall(outer)
        nonMemoizedFunction((5, inner))
        with self.assertRaises(AssertionError):
            nonMemoizedFunction((5, inner))
        
        nonMemoizedFunction((6, inner))
        with self.assertRaises(AssertionError):
            nonMemoizedFunction((6, inner))


if __name__ == '__main__':
    unittest.main()
