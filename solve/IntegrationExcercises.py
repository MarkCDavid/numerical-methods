import sys
import os
sys.path.append(os.getcwd())

import sympy as sym
from numericalmethods import Interpolation
from numericalmethods import Integration
from numericalmethods import Utility

x = sym.Symbol('x')

print("===== Excercise 1 =====")

integration_data = Integration.FunctionIntegrationData(x**3 - 1, x)
interval = (0, 2)
step_evaluator = Integration.NodeCount(4)


def pp_result(integration):
    S = integration.integrate(interval, step_evaluator)
    e = integration.absolute_error(interval, step_evaluator)
    R = integration.theoretical_error(interval, step_evaluator)
    print(f'S = {S} | e = {e} | R = {R}')

pp_result(Integration.MiddleSquareIntegration(integration_data))
pp_result(Integration.TrapezoidIntegration(integration_data))
pp_result(Integration.SimpsonIntegration(integration_data))

print("===== Excercise 2 =====")

precision = 10**-5
integration_data = Integration.FunctionIntegrationData(x**4 * sym.exp(2 * x), x)
interval = (1, 3)

print("a) h = %f | N = %f" % Integration.MiddleSquareIntegration(integration_data).steps(interval, precision))
print("a) h = %f | N = %f" % Integration.TrapezoidIntegration(integration_data).steps(interval, precision))

integration_data = Integration.FunctionIntegrationData(x**3 * sym.cos(3 * x), x)
interval = (0, sym.pi)

print("b) h = %f | N = %f" % Integration.MiddleSquareIntegration(integration_data).steps(interval, precision))
print("b) h = %f | N = %f" % Integration.TrapezoidIntegration(integration_data).steps(interval, precision))

print("===== Excercise 3 =====")

precision = 10**-3
integration_data = Integration.FunctionIntegrationData((x**(3 / 2)) / ((2 - x)**(1 / 2)), x)
interval = (0, 1)
step_evaluator = Integration.NodeCount(2)

print(abs(Integration.SimpsonIntegration(integration_data).runge_error(interval, step_evaluator)) < precision)

print("===== Excercise 4 =====")

precision = 10**-3
integration_data = Integration.FunctionIntegrationData(sym.sqrt(1 + sym.sin(2 * x)**2), x)
interval = (0, sym.pi/2.0)
print(sym.N(Integration.SimpsonIntegration(integration_data).integrate(interval, condition=Utility.Precision(precision))))

print("===== Excercise 5 =====")

integration_data = Integration.FunctionIntegrationData(2 * x, x)
interval = (2, 5)
fr1 = Integration.MiddleSquareIntegration(integration_data).integrate(interval, Integration.StepSize(0.05)) * 3
integration_data = Integration.FunctionIntegrationData(5 * x**2 + 3, x)
interval = (5, 9)
fr2 = Integration.MiddleSquareIntegration(integration_data).integrate(interval, Integration.StepSize(0.05)) * 4
print(f"Total time: {fr1 + fr2}")

print("===== Excercise 7 =====")

x_values = [0, 10, 20, 30, 40, 50, 60, 70, 80]
y_values = [10, 3, 10, 45, 10, 15, 30, 4, 12]

integration_data = Integration.ValuePairIntegrationData(x_values, y_values, x)
interval = (0, 80)
print(Integration.SimpsonIntegration(integration_data).integrate(interval, Integration.NodeCount(2)))
print(Integration.SimpsonIntegration(integration_data).integrate(interval, Integration.NodeCount(4)))
print(abs(Integration.SimpsonIntegration(integration_data).runge_error(interval, Integration.NodeCount(4))))


print("===== Excercise 8 =====")

x_values = [0, 2, 4, 6, 8, 10, 12, 14, 16]
y_values = [0, 13, 20, 23, 25, 28, 29, 29, 30]

integration_data = Integration.ValuePairIntegrationData(x_values, y_values, x)
interval = (0, 16)
print(Integration.MiddleSquareIntegration(integration_data).integrate(interval, Integration.NodeCount(4)))
print(Integration.TrapezoidIntegration(integration_data).integrate(interval, Integration.NodeCount(4)))
print(Integration.SimpsonIntegration(integration_data).integrate(interval, Integration.NodeCount(4)))
