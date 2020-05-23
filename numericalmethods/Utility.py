"""Utility module, designed to assist solving excercises in university course for Numerical Methods.

Contains methods that do not belong to a specific category.
"""

from functools import reduce
import sympy as sym
import numpy as np
import operator
import json
import enum

def value_sampling(fx, symbol, interval, step_size=0.05):
    """Sample the values of a function within the specified interval."""
    return [fx.subs(symbol, x) for x in arange(interval, step_size=step_size)]

def arange(interval, *, step_size=None, step_evaluator=None):
    """Fetch a range from start to end, inclusive."""
    step_size = step_size if step_evaluator is None else step_evaluator.step_size(interval)
    return np.arange(interval[0], interval[1] + step_size/2.0, step_size)

def subinterval(points, size=2):
    """Zip up interval points into subintervals."""
    return list(zip(*[points[i:] for i in range(size)]))

def maximum_absolute_value(fx, symbol, interval, derivative_degree=0, step_size=0.05):
    """Calculate the maximum absolute value of a function within the specified interval.
    
    Provides a possibility to provide a derivative degree.
    """
    return max(value_sampling(sym.Abs(sym.diff(fx, symbol, derivative_degree)), symbol, interval, step_size))

def signum(value):
    """Get the sign of the value."""
    return 1 if value >= 0 else -1

def same_sign(function, symbol, interval, step_size=0.05):
    """Check if the function is of the same sign over the provided interval."""
    sign = signum(function.subs(symbol, interval[0]))
    return all([sign == signum(x) for x in value_sampling(function, symbol, interval, step_size=step_size)])

def triangle_array(size, default_value=None):
    """Create a triangular array."""
    return [[default_value for _ in range(size - i)] for i in range(size)]

def interval(start, end, symbol):
    """Create a logical interval."""
    return (symbol > start) & (symbol <= end)

def gap(values, index):
    """Calculate gap between two values."""
    return values[index + 1] - values[index]

def invert_data(header, data, key=operator.itemgetter(1)):
    """Inverts header with data and sorts by the provided key."""
    return zip(*sorted(zip(header, data), key=key))

def product(data):
    """Perform multiplication on each element in the data list."""
    return reduce(operator.mul, data, 1)

class Parity(enum.Enum):
    """Parity enumeration."""
    EVEN = 0,
    ODD = 1

class Memoized:
    """Wrapper class for function calls to be memoized.
    
    First call to the function performs the calculation and the return value is memoized. Sequential calls return memoized value.
    """

    @staticmethod
    def key(*args, **kwargs):
        """Generate a consistent key from args and kwargs."""
        return args, json.dumps(kwargs)

    def __init__(self, function, initial_values=None):
        """Create memoization wrapper."""
        self.values = {} if initial_values is None else initial_values
        self.function = function

    def __call__(self, *args, **kwargs):
        """Perform calculation/memoization."""
        key = Memoized.key(*args, **kwargs)
        if key not in self.values:
            self.values[key] = self.function(*args, **kwargs)
        return self.values[key]

class Condition:
    """Base class for condition."""
    def __init__(self, condition):
        self.condition = condition

class Precision(Condition):
    """Condition for precision."""
    
    def check(self, current):
        """Check if provided value is more precise than condition."""
        return current <= self.condition

class Iteration(Condition):
    """Condition for iteration."""
    
    def __init__(self, condition):
        """Initialize iteration counter to 0."""
        super().__init__(condition)
        self.iteration = 0
    
    def check(self, current):
        """Iterate iteration counter and check if we have done specified amount of iterations."""
        if self.iteration >= self.condition:
            return True
        self.iteration += 1

