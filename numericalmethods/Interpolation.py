from functools import reduce 
import sympy as sym
import operator
from numericalmethods import Utility

class InterpolatingPolynomial:

    def __init__(self, x_values, y_values, symbol):
        self.x_values = x_values
        self.y_values = y_values
        self.size = len(x_values)
        self.symbol = symbol

    def basis_polynomial(self, degree, skip=None, offset=0):
        return reduce(
            operator.mul, 
            [self.symbol - x for i, x in enumerate(self.x_values[offset:degree+offset]) if i != skip],
            1
        )

    def error(self, fx, degree):
        return sym.Abs(fx - self.polynomial(degree))

    def theoretical_error(self, fx, degree, interval_start, interval_end):
        return (Utility.maximum_absolute_value(fx, self.symbol, interval_start, interval_end, degree + 1) * sym.Abs(self.basis_polynomial(degree + 1)))/sym.factorial(degree + 1)

    def practical_error_polynomial_difference(self, degree):
        return sym.Abs(self.polynomial(degree + 1) - self.polynomial(degree))

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
         ], key=operator.itemgetter(1))[0]
        
    def polynomial(self, degree):
        print("This is not any particular interpolating polynomial.")
        print("Please use NetwonInterpolatingPolynomial or LagrangeInterpolatingPolynomial")

class NetwonInterpolatingPolynomial(InterpolatingPolynomial):

    def __init__(self, x_values, y_values, symbol):
        super().__init__(x_values, y_values, symbol)
        self._coefficient = [[None for _ in range(self.size - i)] if i != 0 else self.y_values for i in range(self.size) ]
        self.coefficient(self.size - 1, 0)
    
    def polynomial(self, degree):
        if degree == 0:
            return self.y_values[degree]
        else:
            return sym.simplify(self.polynomial(degree - 1) + self.basis_polynomial(degree) * self.coefficient(degree))

    def coefficient(self, degree, index=0):
        if self._coefficient[degree][index] is None:
            self._coefficient[degree][index] = (self.coefficient(degree - 1, index) - self.coefficient(degree - 1, index + 1))/(self.x_values[index] - self.x_values[index + degree])
        return self._coefficient[degree][index]

    def practical_error_next_degree(self, degree):
        return sym.Abs(self.coefficient(degree, 0)) * sym.Abs(self.basis_polynomial(degree))

    def coefficients(self, degree):
        return self._coefficient[:degree + 1]

        
class LagrangeInterpolatingPolynomial(InterpolatingPolynomial):

    def polynomial(self, degree, offset=0):
        return sum([(self.coefficient(degree, index, offset=offset) * self.y_values[index]) for index in range(degree + 1 + offset)])

    def coefficient(self, degree, index, offset=0):
        basis = self.basis_polynomial(degree + 1, skip=index, offset=offset)
        return basis/basis.subs(self.symbol, self.x_values[index])
