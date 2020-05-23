import sympy as sym
import numpy as np
from numericalmethods import Utility


class NonLinearEquations:

    def __init__(self, function, symbol):
        self.function = function
        self.symbol = symbol
        self.iteration = 0
        
    def solve(self, condition):
        succ = self.successor()
        while not condition.check(self.precision(succ)):
            self.iteration += 1
            self.iterate(succ)   
            succ = self.successor()
        return (sym.N(succ), self.iteration, self.precision(succ), self.true_root(succ))

    def root_parity(self, interval):
        sign = self.function.subs(self.symbol, interval[0]) * self.function.subs(self.symbol, interval[1])
        return Utility.Parity.ODD if sign < 0 else Utility.Parity.EVEN

    def true_root(self, successor):
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

    def __init__(self, function, symbol, interval):
        super().__init__(function, symbol)
        self.initial_interval = interval
        self.interval = interval
    
    def successor(self):
        return (self.interval[0] + self.interval[1])/2.0

    def precision(self, successor):
        return (self.interval[1] - self.interval[0])/2.0

    def iterate(self, successor):
        value_interval_start = self.function.subs(self.symbol, self.interval[0])
        value_interval_middle = self.function.subs(self.symbol, successor)
        self.interval =  (self.interval[0], successor) if value_interval_start * value_interval_middle < 0 else (successor, self.interval[1])

    def converges_in(self, precision):
        return sym.log((self.initial_interval[0] + self.initial_interval[1])/precision, 2) - 1

class FixedPointNLE(NonLinearEquations):

    def __init__(self, function, symbol, interval, x0):
        super().__init__(function, symbol)
        self.x0 = x0
        self.q = Utility.maximum_absolute_value(self.function, self.symbol, interval, 1)
    
    def successor(self):
        return self.function.subs(self.symbol, self.x0)

    def precision(self, successor):
        return (self.q * sym.Abs(successor - self.x0))/(1 - self.q)

    def iterate(self, successor):
        self.x0 = successor

class NewtonNLE(NonLinearEquations):

    def __init__(self, function, symbol, x0, constant=False):
        super().__init__(function, symbol)
        self.derivative = sym.diff(self.function)
        self.x0 = x0
        self.x0_c = x0 if constant else None

    def successor(self):
        function_value = self.function.subs(self.symbol, self.x0)
        derivative_value = self.derivative.subs(self.symbol, self.x0 if self.x0_c is None else self.x0_c)
        return self.x0- sym.N(function_value/derivative_value)

    def precision(self, successor):
        return sym.Abs(successor - self.x0)

    def iterate(self, successor):
        self.x0 = successor

class SecantNLE(NonLinearEquations):

    def __init__(self, function, symbol, x0, x1, constant=False):
        super().__init__(function, symbol)
        self.derivative = self._derivative(self.function)
        self.iteration = 1
        self.x0 = x0
        self.x0_c = x0 if constant else None
        self.x1 = x1

    def successor(self):
        function_value = self.function.subs(self.symbol, self.x1)
        derivative_value = self.derivative(self.x0 if self.x0_c is None else self.x0_c, self.x1)
        return self.x1 - sym.N(function_value/derivative_value)

    def precision(self, successor):
        return sym.Abs(successor - self.x0)

    def iterate(self, successor):
        self.x0 = self.x1
        self.x1 = successor

    def _derivative(self, function):
        return lambda x0, x1: (function.subs(self.symbol, x1) - function.subs(self.symbol, x0))/(x1 - x0)