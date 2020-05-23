import unittest
import sympy as sym
import numpy as np
from numericalmethods import Integration, Utility
from test.cassert import CustomAssertions

class IntegrationTest(unittest.TestCase, CustomAssertions):

    def test_square_method(self):
        x = sym.Symbol('x')
        fx = 1/x
        integration_data = Integration.FunctionIntegrationData(fx, x)
        integration = Integration.MiddleSquareIntegration(integration_data)
        interval = (1, 3)

        self.assertAlmostEqual(integration.integrate(interval, Integration.NodeCount(4)), 1.089755, places=4)
        self.assertAlmostEqual(integration.absolute_error(interval, Integration.NodeCount(4)), 0.008857, places=4)
        self.assertAlmostEqual(integration.theoretical_error(interval, Integration.NodeCount(4)), 0.041667, places=4)
        self.assertAlmostEqual(abs(integration.runge_error(interval, Integration.NodeCount(4))), 0.007696, places=4)

    def test_square_valuepair_method(self):
        x = sym.Symbol('x')
        fx = 1/x
        interval = (1, 3)
        x_values = Utility.arange(interval, step_evaluator=Integration.NodeCount(8))
        y_values = [fx.subs(x, xi) for xi in x_values]
        integration_data = Integration.ValuePairIntegrationData(x_values, y_values, x)

        integration = Integration.MiddleSquareIntegration(integration_data)

        self.assertAlmostEqual(integration.integrate(interval, Integration.NodeCount(4)), 1.089755, places=4)
        self.assertAlmostEqual(integration.absolute_error(interval, Integration.NodeCount(4)), 0.008857, places=4)
        self.assertAlmostEqual(integration.theoretical_error(interval, Integration.NodeCount(4)), 0.040667, places=4)
        self.assertAlmostEqual(abs(integration.runge_error(interval, Integration.NodeCount(4))), 0.007696, places=4)

    def test_trapezoid_method(self):
        x = sym.Symbol('x')
        fx = 1/x
        interval = (1, 3)
        integration_data = Integration.FunctionIntegrationData(fx, x)

        integration = Integration.TrapezoidIntegration(integration_data)

        self.assertAlmostEqual(integration.integrate(interval, Integration.NodeCount(4)), 1.11667, places=4)
        self.assertAlmostEqual(integration.absolute_error(interval, Integration.NodeCount(4)), 0.01806, places=4)
        self.assertAlmostEqual(integration.theoretical_error(interval, Integration.NodeCount(4)), 0.08333, places=4)
        self.assertAlmostEqual(abs(integration.runge_error(interval, Integration.NodeCount(4))), 0.016666, places=4)

    def test_trapezoid_valuepair_method(self):
        x = sym.Symbol('x')
        fx = 1/x
        interval = (1, 3)
        x_values = Utility.arange(interval, step_evaluator=Integration.NodeCount(8))
        y_values = [fx.subs(x, xi) for xi in x_values]
        integration_data = Integration.ValuePairIntegrationData(x_values, y_values, x)

        integration = Integration.TrapezoidIntegration(integration_data)

        self.assertAlmostEqual(integration.integrate(interval, Integration.NodeCount(4)), 1.11667, places=4)
        self.assertAlmostEqual(integration.absolute_error(interval, Integration.NodeCount(4)), 0.01806, places=4)
        self.assertAlmostEqual(integration.theoretical_error(interval, Integration.NodeCount(4)), 0.08133, places=4)
        self.assertAlmostEqual(abs(integration.runge_error(interval, Integration.NodeCount(4))), 0.016666, places=4)


    def test_simpson_method(self):
        x = sym.Symbol('x')
        fx = 1/x
        interval = (1, 3)
        integration_data = Integration.FunctionIntegrationData(fx, x)
        integration = Integration.SimpsonIntegration(integration_data)

        self.assertAlmostEqual(integration.integrate(interval, Integration.NodeCount(4)), 1.098725, places=4)
        self.assertAlmostEqual(integration.absolute_error(interval, Integration.NodeCount(4)), 0.000113, places=4)
        self.assertAlmostEqual(integration.theoretical_error(interval, Integration.NodeCount(4)), 0.001042, places=4)
        self.assertAlmostEqual(abs(integration.runge_error(interval, Integration.NodeCount(4))), 0.000085, places=4)

    
    def test_simpson_valuepair_method(self):
        x = sym.Symbol('x')
        fx = 1/x
        interval = (1, 3)
        x_values = Utility.arange(interval, step_evaluator=Integration.NodeCount(8))
        y_values = [fx.subs(x, xi) for xi in x_values]
        integration_data = Integration.ValuePairIntegrationData(x_values, y_values, x)

        integration = Integration.SimpsonIntegration(integration_data)

        self.assertAlmostEqual(integration.integrate(interval, Integration.NodeCount(4)), 1.098725, places=4)
        self.assertAlmostEqual(integration.absolute_error(interval, Integration.NodeCount(4)), 0.000113, places=4)
        self.assertAlmostEqual(integration.theoretical_error(interval, Integration.NodeCount(4)), 0.000693, places=4)
        self.assertAlmostEqual(abs(integration.runge_error(interval, Integration.NodeCount(4))), 0.000085, places=4)


    def test_step_value(self):
        interval = (1, 3)

        self.assertListAlmostEqual([
            Integration.StepSize(1/2).step_size(interval),
            Integration.StepSize(1/4).step_size(interval),
            Integration.StepSize(1/8).step_size(interval),
            Integration.StepSize(1/16).step_size(interval),
            Integration.StepSize(1/32).step_size(interval),
        ], [1/2, 1/4, 1/8, 1/16, 1/32])

        self.assertListAlmostEqual([
            Integration.NodeCount(1).step_size(interval),
            Integration.NodeCount(2).step_size(interval),
            Integration.NodeCount(4).step_size(interval),
            Integration.NodeCount(8).step_size(interval),
            Integration.NodeCount(16).step_size(interval),
        ], [2, 1, 1/2, 1/4, 1/8])
