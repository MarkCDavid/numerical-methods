import sympy as sym
from itertools import product
from operator import itemgetter

class InterpolatingPolynomial:

    def __init__(self, x_values, y_values, symbol):
        self.x_values = x_values
        self.y_values = y_values
        self.symbol = symbol

    def fit(self, function):
        return all([
            function.subs(self.symbol, xi) == yi 
            for xi, yi 
            in zip(self.x_values, self.y_values)
        ])

    def fit_points(self, at_point, degree):
        def score(group):
            return sum([abs(at_point - x) for x in group])

        return min([
            (group, score(group))
            for group 
            in zip(*[self.x_values[i:] for i in range(degree + 1)])
         ], key=itemgetter(1))[0]
        
