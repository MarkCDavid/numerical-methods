import sympy as sym
import operator


class Optimisation:
    
    def __init__(self, function, symbol):
        self.function = function
        self.symbol = symbol

class GoldenRatioOptimisation(Optimisation):
    
    def __init__(self, function, symbol):
        super().__init__(function, symbol)
        self.q = sym.N((3 - sym.sqrt(5))/2)

    def optimize(self, a, b, condition, key=min):
        while True:
            x1, x2 = self.next(a, b)
            xm = self.optimal([a, b, x1, x2], key=key)
            a, b = (a, x2) if abs(xm - a) < abs(xm - b) else (x1, b)

            if condition.check(b - a):
                return xm

    def next(self, a, b):
        difference = sym.N((b - a) * self.q)
        return a + difference, b - difference

    def optimal(self, x_values, key):
        return key([(xi, self.function.subs(self.symbol, xi)) for xi in x_values], key=operator.itemgetter(1))[0]


class NetwonMethodOptimisation(Optimisation):
    
    def __init__(self, function, symbol):
        super().__init__(function, symbol)
        self.function_d1 = sym.diff(function, symbol, 1)
        self.function_d2 = sym.diff(function, symbol, 2)

    def optimize(self, x0, condition):
        while True:
            x1 = self.next(x0)
            if condition.check(abs(x0 - x1)):
                return x1
            x0 = x1

    def next(self, x):
        return x - self.function_d1.subs(self.symbol, x)/self.function_d2.subs(self.symbol, x)

    def is_minimum(self, x):
        return self.function_d2.subs(self.symbol, x) > 0

    def is_maximum(self, x):
        return self.function_d2.subs(self.symbol, x) < 0
