"""Interpolation module, designed to assist solving excercises in university course for Numerical Methods."""

from functools import reduce
from numericalmethods import Utility
import sympy as sym
import operator


class InterpolatingPolynomial:
    """Base class for Polynomial Interpolation.
    
    Unable to generate polynomials. Use NetwonInterpolatingPolynomial or LagrangeInterpolatingPolynomial classes.
    """

    def __init__(self, x_values, y_values, symbol):
        """Create an interpolating polynomial generator for base calculations, that do not require generating the interpolating polynomial."""
        self.x_values = x_values
        self.y_values = y_values
        self.size = len(x_values)
        self.symbol = symbol

    def basis_polynomial(self, degree, skip=None, offset=0):
        """Generate a basis polynomial, of specified degree.
        
        Basis polynomial:
            (x - x0)(x - x1)...(x - xN).
        Provides a possibility to skip a multiplier of specified index:
            (x - x0)(x - x2) for degree=3, skip=1.
        Provides a possibility to calculate basis polynomial with an offset:
            (x - x1)(x - x2) for degree=2, offset=1.
        """
        return reduce(
            operator.mul, 
            [self.symbol - x for i, x in enumerate(self.x_values[offset:degree+offset]) if i != skip],
            1
        )

    def error(self, fx, degree):
        """Calculate the interpolation error. Returns a symbolic function."""
        return sym.Abs(fx - self.polynomial(degree))

    def theoretical_error(self, fx, degree, interval_start, interval_end):
        """Calculate the theoretical interpolation error. Returns a symbolic function."""
        return (Utility.maximum_absolute_value(fx, self.symbol, interval_start, interval_end, degree + 1) * sym.Abs(self.basis_polynomial(degree + 1)))/sym.factorial(degree + 1)

    def practical_error_polynomial_difference(self, degree):
        """Calculate the interpolation error using polynomial difference. Returns a symbolic function.
        
        Polynomial difference:
            |L_n+1 - L_n|
        """
        return sym.Abs(self.polynomial(degree + 1) - self.polynomial(degree))

    def fit(self, function):
        """Check if the function was derived from the initially specified values."""
        return all([
            function.subs(self.symbol, xi) == yi 
            for xi, yi 
            in zip(self.x_values, self.y_values)
        ])

    def fit_points(self, at_point, degree):
        """Find the best point choice for polynomial interpolation, for a specified point."""
        def score(group):
            return sum([abs(at_point - x) for x in group])

        return min([
            (group, score(group))
            for group 
            in zip(*[self.x_values[i:] for i in range(degree + 1)])
         ], key=operator.itemgetter(1))[0]
        
    def polynomial(self, degree):
        """Generate a polynomial. Non-functional for the base class."""
        raise NotImplementedError("Using a base class. No method for polynomial generation.")

class NetwonInterpolatingPolynomial(InterpolatingPolynomial):
    """Polynomial Interpolation using Netwon Method."""

    def __init__(self, x_values, y_values, symbol):
        """Create an interpolating polynomial generator."""
        super().__init__(x_values, y_values, symbol)
        self.__newton_differences = NewtonDifferences(x_values, y_values)
    
    def polynomial(self, degree):
        """Generate an interpolating polynomial of specified degree."""
        if degree == 0:
            return self.y_values[degree]
        else:
            return sym.simplify(self.polynomial(degree - 1) + self.basis_polynomial(degree) * self.coefficient(degree))

    def coefficient(self, degree, index=0):
        """Fetch Newton Difference coefficient of specified degree."""
        return self.__newton_differences.coefficient(degree, index)

    def practical_error_next_degree(self, degree):
        """Calculate the interpolation error using approximation. Returns a symbolic function."""
        return sym.Abs(self.coefficient(degree, 0)) * sym.Abs(self.basis_polynomial(degree))

    def coefficients(self, degree):
        """Fetch Newton Difference coefficient table of specified degree."""
        return self.__newton_differences.coefficients(degree)

        
class LagrangeInterpolatingPolynomial(InterpolatingPolynomial):
    """Polynomial Interpolation using Lagrange Method."""

    def polynomial(self, degree, offset=0):
        """Generate an interpolating polynomial of specified degree."""
        return sum([(self.coefficient(degree, index, offset=offset) * self.y_values[index + offset]) for index in range(degree + 1)])

    def coefficient(self, degree, index, offset=0):
        """Generate Lagrangian coefficient of specified degree and index."""
        basis = self.basis_polynomial(degree + 1, skip=index, offset=offset)
        return basis/basis.subs(self.symbol, self.x_values[index + offset])


class InterpolatingSpline:
    
    def __init__(self, x_values, y_values, symbol):
        self.x_values = x_values
        self.y_values = y_values
        self.size = len(x_values)
        self.symbol = symbol

    @staticmethod
    def is_spline(functions, points, symbol, degree=None, places=5):
        max_degree = max([sym.degree(x) for x in functions])
        precission = 10**(-places)
        current_degree = 0
        while True:
            max_difference = max([sym.Abs(f0.subs(symbol, xi) - f1.subs(symbol, xi)) for f0, f1, xi in zip(functions, functions[1:], points[1:])])
            if max_difference > precission:
                return (current_degree >= max_degree, current_degree) if degree is None else (current_degree == degree, current_degree)
            functions = [sym.diff(fx, symbol) for fx in functions]
            current_degree += 1


class LinearInterpolatingSpline(InterpolatingSpline):

    def spline(self):
        return [LagrangeInterpolatingPolynomial(self.x_values, self.y_values, self.symbol).polynomial(1, offset=offset) for offset in range(self.size - 1)]


class NewtonDifferences:

    def __init__(self, x_values, y_values):
        """Construct NewtonDifferences data class."""
        size = len(x_values)
        self._coefficient = Utility.triangle_array(size)
        self._coefficient[0] = y_values
        self.__recursive_coefficient(x_values, size - 1, 0)
        
    def __recursive_coefficient(self, x_values, degree, index):
        """Generate Newton Difference coefficient of specified degree."""
        if self._coefficient[degree][index] is None:
            self._coefficient[degree][index] = (self.__recursive_coefficient(x_values, degree - 1, index) - self.__recursive_coefficient(x_values, degree - 1, index + 1))/(x_values[index] - x_values[index + degree])
        return self._coefficient[degree][index]

    def coefficient(self, degree, index=0):
        return self._coefficient[degree][index]
    
    def coefficients(self, degree):
        return self._coefficient[:degree + 1]
