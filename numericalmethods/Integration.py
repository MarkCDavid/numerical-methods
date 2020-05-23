"""Integration module, designed to assist solving excercises in university course for Numerical Methods."""

import sympy as sym
import numpy as np
from numericalmethods import Utility, Interpolation


class IntegrationData:
    """Base class for Integration Data."""

    def evaluate(self, x):
        """Evaluate function at the point x."""
        return self.function.subs(self.symbol, x)

class FunctionIntegrationData(IntegrationData):
    """Integration Data based on a function."""

    def __init__(self, function, symbol):
        """Construct the integration data."""
        self.function = function
        self.symbol = symbol

class ValuePairIntegrationData(IntegrationData):
    """Integration Data based on a (x, y) value pairs.
    
    Generates the function using an interpolating polynomial.
    """

    def __init__(self, x_values, y_values, symbol):
        """Construct the integration data."""
        self.function = Interpolation.LagrangeInterpolatingPolynomial(x_values, y_values, symbol).polynomial(len(x_values) - 1)
        self.symbol = symbol

class StepSize:
    """Step Size evaluator, using step size directly."""
    
    def __init__(self, h):
        """Construct the step size evaluator."""
        self.h = h

    def step_size(self, interval):
        """Evaluate step size."""
        return self.h

    def scale(self, by):
        """Scale the step size by specified amount."""
        return StepSize(self.h / by)


class NodeCount:
    """Step Size evaluator, using number of node points."""

    def __init__(self, N):
        """Construct the step size evaluator."""
        self.N = N

    def step_size(self, interval):
        """Evaluate step size."""
        return (interval[1] - interval[0]) / self.N

    def scale(self, by):
        """Scale the node count by specified amount."""
        return NodeCount(self.N * by)


class Integration:
    """Base integration class providing abstract methods for integral calculation and error evaluation."""

    def __init__(self, integration_data):
        """Construct the integration class."""
        self.integration_data = integration_data
        self.symbol = integration_data.symbol

    def integrate(self, interval, step_evaluator=None, *, condition=None):
        """Integrate the data."""
        step_evaluator = step_evaluator if condition is None else NodeCount(2)
        while True:
            if condition is None:
                return self._integrate(interval, step_evaluator)
            error = self.runge_error(interval, step_evaluator)
            if condition.check(abs(error)):
                return self._integrate(interval, step_evaluator) + error
            step_evaluator = step_evaluator.scale(2.0)

    def absolute_error(self, interval, step_evaluator):
        """Absolute error of the data."""
        true_value = sym.N(sym.integrate(self.integration_data.function, (self.symbol, interval[0], interval[1])))
        approxiamte_value = self._integrate(interval, step_evaluator)
        return sym.Abs(true_value - approxiamte_value)

    def theoretical_error(self, interval, step_evaluator):
        """Theoretical error of the data."""
        step_size = step_evaluator.step_size(interval)
        interval_size = interval[1] - interval[0]
        M = Utility.maximum_absolute_value(self.integration_data.function, self.symbol, interval, self.error_degree)
        return sym.Abs((M * (step_size ** self.error_degree) * interval_size) / self.error_divisor)

    def runge_error(self, interval, step_evaluator):
        """Runge error of the data.
        
        Return value is not absolute.
        """
        s1n = self._integrate(interval, step_evaluator.scale(0.5))
        s2n = self._integrate(interval, step_evaluator)
        return (s2n - s1n) / (2 ** self.error_degree - 1)

    def steps(self, interval, precision):
        """Calculate (using theoretical error) how large and how many steps you would need to integrate the data with the specified precision."""
        interval_size = interval[1] - interval[0]
        M = Utility.maximum_absolute_value(self.integration_data.function, self.symbol, interval, self.error_degree)
        h = sym.sqrt((precision * self.error_divisor) /(M * interval_size))
        return (h, interval_size/h)

    def _integrate(self, interval, step_evaluator):
        """Integrate the data for the actual value."""
        step_size = step_evaluator.step_size(interval)
        sigma_sum = sum([self.integrate_interval(subinterval) for subinterval in Utility.subinterval(Utility.arange(interval, step_evaluator=step_evaluator))])
        return self.multiplier * step_size * sigma_sum

class MiddleSquareIntegration(Integration):
    """Integration using Middle Square method."""

    def __init__(self, integration_data):
        """Construct the integration class and provide constants to use in abstract base class methods."""
        super().__init__(integration_data)
        self.multiplier = 1
        self.error_degree = 2
        self.error_divisor = 24

    def integrate_interval(self, interval):
        """Interval integration step for Middle Square integration.
        
        f(x_i/2)
        """
        midpoint = (interval[0] + interval[1])/2.0
        return self.integration_data.evaluate(midpoint)

class TrapezoidIntegration(Integration):
    """Integration using Trapezoid method."""

    def __init__(self, integration_data):
        """Construct the integration class and provide constants to use in abstract base class methods."""
        super().__init__(integration_data)
        self.multiplier = 0.5
        self.error_degree = 2
        self.error_divisor = 12

    def integrate_interval(self, interval):
        """Interval integration step for Trapezoid integration.
        
        f(x_i) + f(x_i+1)
        """
        return self.integration_data.evaluate(interval[0]) + self.integration_data.evaluate(interval[1])


class SimpsonIntegration(Integration):
    """Integration using Simpson method."""

    def __init__(self, integration_data):
        """Construct the integration class and provide constants to use in abstract base class methods."""
        super().__init__(integration_data)
        self.multiplier = 1.0/6.0
        self.error_degree = 4
        self.error_divisor = 2880

    def integrate_interval(self, interval):
        """Interval integration step for Simpson integration.
        
        f(x_i) + 4*f(x_i/2) + f(x_i+1)
        """
        midpoint = (interval[0] + interval[1])/2.0
        return self.integration_data.evaluate(interval[0]) + 4.0 * self.integration_data.evaluate(midpoint) + self.integration_data.evaluate(interval[1])
