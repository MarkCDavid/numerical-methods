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
        integration = Integration.Integration(integration_data)
        
        interval_start, interval_end = 1, 3

        self.assertAlmostEqual(integration.integrate(Integration.SquareMethod(), interval_start, interval_end, Integration.NodeCount(4)), 1.089755, places=4)
        self.assertAlmostEqual(integration.absolute_error(Integration.SquareMethod(), interval_start, interval_end, Integration.NodeCount(4)), 0.008857, places=4)
        self.assertAlmostEqual(integration.theoretical_error(Integration.SquareMethod(), interval_start, interval_end, Integration.NodeCount(4)), 0.041667, places=4)

    def test_square_valuepair_method(self):
        x = sym.Symbol('x')
        fx = 1/x
        x_values = Utility.arange(1, 3, step=0.25)
        y_values = [fx.subs(x, xi) for xi in x_values]
        integration_data = Integration.ValuePairIntegrationData(x_values, y_values, x)
        integration = Integration.Integration(integration_data)
        
        interval_start, interval_end = 1, 3

        self.assertAlmostEqual(integration.integrate(Integration.SquareMethod()), 1.089755, places=4)

    def test_trapezoid_method(self):
        x = sym.Symbol('x')
        fx = 1/x
        integration_data = Integration.FunctionIntegrationData(fx, x)
        integration = Integration.Integration(integration_data)
        
        interval_start, interval_end = 1, 3

        self.assertAlmostEqual(integration.integrate(Integration.TrapezoidMethod(), interval_start, interval_end, Integration.NodeCount(4)), 1.11667, places=4)
        self.assertAlmostEqual(integration.absolute_error(Integration.TrapezoidMethod(), interval_start, interval_end, Integration.NodeCount(4)), 0.01806, places=4)
        self.assertAlmostEqual(integration.theoretical_error(Integration.TrapezoidMethod(), interval_start, interval_end, Integration.NodeCount(4)), 0.08333, places=4)

    def test_trapezoid_valuepair_method(self):
        x = sym.Symbol('x')
        fx = 1/x
        x_values = Utility.arange(1, 3, step=0.5)
        y_values = [fx.subs(x, xi) for xi in x_values]
        integration_data = Integration.ValuePairIntegrationData(x_values, y_values, x)
        integration = Integration.Integration(integration_data)
        
        interval_start, interval_end = 1, 3

        self.assertAlmostEqual(integration.integrate(Integration.TrapezoidMethod()), 1.11667, places=4)

    def test_simpson_method(self):
        x = sym.Symbol('x')
        fx = 1/x
        integration_data = Integration.FunctionIntegrationData(fx, x)
        integration = Integration.Integration(integration_data)
        
        interval_start, interval_end = 1, 3

        self.assertAlmostEqual(integration.integrate(Integration.SimpsonMethod(), interval_start, interval_end, Integration.NodeCount(4)), 1.098725, places=4)
        self.assertAlmostEqual(integration.absolute_error(Integration.SimpsonMethod(), interval_start, interval_end, Integration.NodeCount(4)), 0.000113, places=4)
        self.assertAlmostEqual(integration.theoretical_error(Integration.SimpsonMethod(), interval_start, interval_end, Integration.NodeCount(4)), 0.001042, places=4)

    
    def test_simpson_valuepair_method(self):
        x = sym.Symbol('x')
        fx = 1/x
        x_values = Utility.arange(1, 3, step=0.5)
        y_values = [fx.subs(x, xi) for xi in x_values]
        integration_data = Integration.ValuePairIntegrationData(x_values, y_values, x)
        integration = Integration.Integration(integration_data)
        
        interval_start, interval_end = 1, 3

        self.assertAlmostEqual(integration.integrate(Integration.SimpsonMethod()), 1.1, places=4)



    def test_rugne_error(self):
        x = sym.Symbol('x')
        fx = 1/x
        integration_data = Integration.FunctionIntegrationData(fx, x)
        integration = Integration.Integration(integration_data)
        interval_start, interval_end = 1, 3

        self.assertAlmostEqual(integration.runge_error(Integration.SquareMethod(), interval_start, interval_end, Integration.NodeCount(2)), 0.007696, places=4)
        self.assertAlmostEqual(integration.runge_error(Integration.TrapezoidMethod(), interval_start, interval_end, Integration.NodeCount(2)), 0.016666, places=4)
        self.assertAlmostEqual(integration.runge_error(Integration.SimpsonMethod(), interval_start, interval_end, Integration.NodeCount(2)), 0.000085, places=4)

    def test_step_value(self):
        interval_start, interval_end = 1, 3

        self.assertListAlmostEqual([
            Integration.StepSize(1/2).step_size(interval_start, interval_end),
            Integration.StepSize(1/4).step_size(interval_start, interval_end),
            Integration.StepSize(1/8).step_size(interval_start, interval_end),
            Integration.StepSize(1/16).step_size(interval_start, interval_end),
            Integration.StepSize(1/32).step_size(interval_start, interval_end),
        ], [1/2, 1/4, 1/8, 1/16, 1/32])

        self.assertListAlmostEqual([
            Integration.NodeCount(1).step_size(interval_start, interval_end),
            Integration.NodeCount(2).step_size(interval_start, interval_end),
            Integration.NodeCount(4).step_size(interval_start, interval_end),
            Integration.NodeCount(8).step_size(interval_start, interval_end),
            Integration.NodeCount(16).step_size(interval_start, interval_end),
        ], [2, 1, 1/2, 1/4, 1/8])
