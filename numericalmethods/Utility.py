"""Utility module, designed to assist solving excercises in university course for Numerical Methods.

Contains methods that do not belong to a specific category.
"""

import sympy as sym
import numpy as np

def value_sampling(fx, symbol, interval_start, interval_end, step_size=0.05):
    """Sample the values of a function within the specified interval."""
    return [fx.subs(symbol, x) for x in np.arange(interval_start, interval_end + step_size/2.0, step_size)]

def maximum_absolute_value(fx, symbol, interval_start, interval_end, derivative_degree=0, step_size=0.05):
    """Calculate the maximum absolute value of a function within the specified interval.
    
    Provides a possibility to provide a derivative degree.
    """
    return max(value_sampling(sym.Abs(sym.diff(fx, symbol, derivative_degree)), symbol, interval_start, interval_end, step_size))
