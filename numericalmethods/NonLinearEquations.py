"""Non Linear Equation module, designed to assist solving excercises in university course for Numerical Methods."""

import sympy as sym
import numpy as np
from numericalmethods import Utility


class NonLinearEquations:
    """Base class for Non Linear Equations.

    Provides root count parity determening function.
    Unable to solve equations. For that use MidpointNLE, FixedPointNLE, NewtonNLE or SecantNLE classes.
    """

    def __init__(self, function, symbol):
        """Construct base NLE class instance."""
        self.function = function
        self.symbol = symbol
        self.iteration = 0
        
    def solve(self, condition):
        """Solve the non-linear equation using condition to stop iterating."""
        succ = self.successor()
        iteration = 0
        while not condition.check(self.precision(succ)):
            iteration += 1
            self.iterate(succ)   
            succ = self.successor()
        return (sym.N(succ), iteration, self.precision(succ), self.true_root(succ))

    def root_parity(self, interval):
        """Calculate the parity of the count of roots."""
        sign = self.function.subs(self.symbol, interval[0]) * self.function.subs(self.symbol, interval[1])
        return Utility.Parity.ODD if sign < 0 else Utility.Parity.EVEN

    def true_root(self, successor):
        """Check if the given root is a true root."""
        return self.precision(successor) < 10**-12

    def one_root(self, interval):
        """Check if the function has only on root in the provided interval.
        
        NOTE: Function has a possibility to evaluate to False even if there is only one root in the specified interval.
        This is due to the fact that the theorem this method is implementing only provides certainty one way - if validation passes
        the user can be certain that the function has only one root in the provided interval.
        
        Please do an additional check to see how many solutions you have in the provided interval.
        """
        return self.root_parity(interval) == Utility.Parity.ODD and Utility.same_sign(sym.diff(self.function, self.symbol), self.symbol, interval)


class MidpointNLE(NonLinearEquations):
    """NLE solver using Midpoint method."""

    def __init__(self, function, symbol, interval):
        """Construct Midpoint NLE class instance."""
        super().__init__(function, symbol)
        self.initial_interval = interval
        self.interval = interval
    
    def successor(self):
        """Get the next iteration value."""
        return (self.interval[0] + self.interval[1])/2.0

    def precision(self, successor):
        """Get the precision for iteration value."""
        return (self.interval[1] - self.interval[0])/2.0

    def iterate(self, successor):
        """Proceed to the next iteration."""
        value_interval_start = self.function.subs(self.symbol, self.interval[0])
        value_interval_middle = self.function.subs(self.symbol, successor)
        self.interval =  (self.interval[0], successor) if value_interval_start * value_interval_middle < 0 else (successor, self.interval[1])

    def converges_in(self, precision):
        """Calculate how many steps it would take for the solution to converge to true root."""
        return sym.log((self.initial_interval[0] + self.initial_interval[1])/precision, 2) - 1

class FixedPointNLE(NonLinearEquations):
    """NLE solver using Fixed Point method."""

    def __init__(self, function, symbol, interval, x0):
        """Construct Fixed Point NLE class instance."""
        super().__init__(function, symbol)
        self.x0 = x0
        self.q = Utility.maximum_absolute_value(self.function, self.symbol, interval, 1)
    
    def successor(self):
        """Get the next iteration value."""
        return self.function.subs(self.symbol, self.x0)

    def precision(self, successor):
        """Get the precision for iteration value."""
        return (self.q * sym.Abs(successor - self.x0))/(1 - self.q)

    def iterate(self, successor):
        """Proceed to the next iteration."""
        self.x0 = successor

class NewtonNLE(NonLinearEquations):
    """NLE solver using Newton method."""

    def __init__(self, function, symbol, x0, constant=False):
        """Construct Newton NLE class instance."""
        super().__init__(function, symbol)
        self.derivative = sym.diff(self.function)
        self.x0 = x0
        self.x0_c = x0 if constant else None

    def successor(self):
        """Get the next iteration value."""
        function_value = self.function.subs(self.symbol, self.x0)
        derivative_value = self.derivative.subs(self.symbol, self.x0 if self.x0_c is None else self.x0_c)
        return self.x0- sym.N(function_value/derivative_value)

    def precision(self, successor):
        """Get the precision for iteration value."""
        return sym.Abs(successor - self.x0)

    def iterate(self, successor):
        """Proceed to the next iteration."""
        self.x0 = successor

class SecantNLE(NonLinearEquations):
    """NLE solver using Secant method."""

    def __init__(self, function, symbol, x0, x1, constant=False):
        """Construct Secant NLE class instance."""
        super().__init__(function, symbol)
        self.derivative = self._derivative(self.function)
        self.iteration = 1
        self.x0 = x0
        self.x0_c = x0 if constant else None
        self.x1 = x1

    def successor(self):
        """Get the next iteration value."""
        function_value = self.function.subs(self.symbol, self.x1)
        derivative_value = self.derivative(self.x0 if self.x0_c is None else self.x0_c, self.x1)
        return self.x1 - sym.N(function_value/derivative_value)

    def precision(self, successor):
        """Get the precision for iteration value."""
        return sym.Abs(successor - self.x0)

    def iterate(self, successor):
        """Proceed to the next iteration."""
        self.x0 = self.x1
        self.x1 = successor

    def _derivative(self, function):
        """Generate derivative by definition. Returns a function with (x0, x1) as parameters."""
        return lambda x0, x1: (function.subs(self.symbol, x1) - function.subs(self.symbol, x0))/(x1 - x0)