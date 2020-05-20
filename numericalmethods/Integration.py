import sympy as sym
import numpy as np
from numericalmethods import Utility, Interpolation

# INTERVAL -> CLASS
# RUNGE WHY NEGATIVE?

class IntegrationData:

    def evaluate(self, x):
        return self.function.subs(self.symbol, x)

class FunctionIntegrationData(IntegrationData):

    def __init__(self, function, symbol):
        self.function = function
        self.symbol = symbol

class ValuePairIntegrationData(IntegrationData):

    def __init__(self, x_values, y_values, symbol):
        self.function = Interpolation.LagrangeInterpolatingPolynomial(x_values, y_values, symbol).polynomial(len(x_values) - 1)
        self.symbol = symbol

class StepSize:
    def __init__(self, h):
        self.h = h

    def step_size(self, interval):
        return self.h

    def scale(self, by):
        return StepSize(self.h / by)


class NodeCount:
    def __init__(self, N):
        self.N = N

    def step_size(self, interval):
        return (interval[1] - interval[0]) / self.N

    def scale(self, by):
        return NodeCount(self.N * by)


class Integration:

    def __init__(self, integration_data):
        self.integration_data = integration_data
        self.symbol = integration_data.symbol

    def integrate(self, interval, step_evaluator=None, *, condition=None):
        step_evaluator = step_evaluator if condition is None else NodeCount(2)
        while True:
            if condition is None:
                return self._integrate(interval, step_evaluator)
            error = self.runge_error(interval, step_evaluator)
            if condition.check(abs(error)):
                return self._integrate(interval, step_evaluator) + error
            step_evaluator = step_evaluator.scale(2.0)

    def absolute_error(self, interval, step_evaluator):
        true_value = sym.N(sym.integrate(self.integration_data.function, (self.symbol, interval[0], interval[1])))
        approxiamte_value = self._integrate(interval, step_evaluator)
        return sym.Abs(true_value - approxiamte_value)

    def theoretical_error(self, interval, step_evaluator):
        step_size = step_evaluator.step_size(interval)
        interval_size = interval[1] - interval[0]
        M = Utility.maximum_absolute_value(self.integration_data.function, self.symbol, interval[0], interval[1], self.error_degree)
        return sym.Abs((M * (step_size ** self.error_degree) * interval_size) / self.error_divisor)

    def runge_error(self, interval, step_evaluator):
        s1n = self._integrate(interval, step_evaluator.scale(0.5))
        s2n = self._integrate(interval, step_evaluator)
        return (s2n - s1n) / (2 ** self.error_degree - 1)

    def steps(self, interval, precision):
        interval_size = interval[1] - interval[0]
        M = Utility.maximum_absolute_value(self.integration_data.function, self.symbol, interval[0], interval[1], self.error_degree)
        h = sym.sqrt((precision * self.error_divisor) /(M * interval_size))
        return (h, interval_size/h)

    def _integrate(self, interval, step_evaluator):
        step_size = step_evaluator.step_size(interval)
        sigma_sum = sum([self.integrate_interval(subinterval) for subinterval in Utility.subinterval(Utility.arange(interval[0], interval[1], step=step_size))])
        return self.multiplier * step_size * sigma_sum

class MiddleSquareIntegration(Integration):

    def __init__(self, integration_data):
        super().__init__(integration_data)
        self.multiplier = 1
        self.error_degree = 2
        self.error_divisor = 24

    def integrate_interval(self, interval):
        midpoint = (interval[0] + interval[1])/2.0
        return self.integration_data.evaluate(midpoint)

class TrapezoidIntegration(Integration):

    def __init__(self, integration_data):
        super().__init__(integration_data)
        self.multiplier = 0.5
        self.error_degree = 2
        self.error_divisor = 12

    def integrate_interval(self, interval):
        return self.integration_data.evaluate(interval[0]) + self.integration_data.evaluate(interval[1])


class SimpsonIntegration(Integration):

    def __init__(self, integration_data):
        super().__init__(integration_data)
        self.multiplier = 1.0/6.0
        self.error_degree = 4
        self.error_divisor = 2880

    def integrate_interval(self, interval):
        midpoint = (interval[0] + interval[1])/2.0
        return self.integration_data.evaluate(interval[0]) + 4.0 * self.integration_data.evaluate(midpoint) + self.integration_data.evaluate(interval[1])
