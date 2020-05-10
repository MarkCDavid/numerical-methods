"""Utility module, designed to assist solving excercises in university course for Numerical Methods.

Contains methods that do not belong to a specific category.
"""

import sympy as sym
import numpy as np
import json

def value_sampling(fx, symbol, interval_start, interval_end, step_size=0.05):
    """Sample the values of a function within the specified interval."""
    return [fx.subs(symbol, x) for x in np.arange(interval_start, interval_end + step_size/2.0, step_size)]

def maximum_absolute_value(fx, symbol, interval_start, interval_end, derivative_degree=0, step_size=0.05):
    """Calculate the maximum absolute value of a function within the specified interval.
    
    Provides a possibility to provide a derivative degree.
    """
    return max(value_sampling(sym.Abs(sym.diff(fx, symbol, derivative_degree)), symbol, interval_start, interval_end, step_size))

def triangle_array(size, default_value=None):
    """Create a triangular array."""
    return [[default_value for _ in range(size - i)] for i in range(size)]

def interval(start, end, symbol):
    """Create an interval."""
    return (symbol > start) & (symbol <= end)

def gap(values, index):
    """Calculate gap between two values."""
    return values[index + 1] - values[index]


class Memoized:
    """Wrapper class for function calls to be memoized.
    
    First call to the function performs the calculation and the return value is memoized. Sequential calls return memoized value.
    """

    @staticmethod
    def key(*args, **kwargs):
        return args, json.dumps(kwargs)

    def __init__(self, function, initial_values=None):
        """Create memoization wrapper."""
        self.values = {} if initial_values is None else initial_values
        self.function = function

    def __call__(self, *args, **kwargs):
        """Perform calculation/memoization."""
        key = Memoized.key(*args, *kwargs)
        if key not in self.values:
            self.values[key] = self.function(*args, **kwargs)
        return self.values[key]

