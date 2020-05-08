import sympy as sym
import numpy as np

def value_sampling(fx, symbol, interval_start, interval_end, step_size=0.05):
    return [fx.subs(symbol, x) for x in np.arange(interval_start, interval_end + step_size/2.0, step_size)]

def maximum_absolute_value(fx, symbol, interval_start, interval_end, derivative_degree, step_size=0.05):
    return max(value_sampling(sym.Abs(sym.diff(fx, symbol, derivative_degree)), symbol, interval_start, interval_end, step_size))
