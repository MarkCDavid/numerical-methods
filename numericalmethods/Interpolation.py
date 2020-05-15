"""Interpolation module, designed to assist solving excercises in university course for Numerical Methods."""

from functools import reduce
from numericalmethods import Utility
import sympy as sym
import operator


class InterpolatingPolynomial:
    """Base class for Polynomial Interpolation.
    
    Unable to generate polynomials. Use NewtonInterpolatingPolynomial or LagrangeInterpolatingPolynomial classes.
    """

    def __init__(self, x_values, y_values, symbol, inverse=False):
        """Create an interpolating polynomial generator for base calculations, that do not require generating the interpolating polynomial."""

        if inverse:
            y_values, x_values = Utility.invert_data(x_values, y_values)

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
        return Utility.product([self.symbol - x for (i, x) in enumerate(self.x_values[offset:degree+offset]) if i != skip])

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

    def practical_error_next_degree(self, degree, newton_differences=None):
        """Calculate the interpolation error using approximation. Returns a symbolic function."""
        return sym.Abs(newton_differences.coefficient(degree, 0)) * sym.Abs(self.basis_polynomial(degree))

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
        
    def polynomial(self, degree, offset=0):
        """Generate a polynomial. Non-functional for the base class."""
        raise NotImplementedError("Using a base class. No method for polynomial generation.")

class NewtonInterpolatingPolynomial(InterpolatingPolynomial):
    """Polynomial Interpolation using Netwon Method."""

    def __init__(self, x_values, y_values, symbol, inverse=False):
        """Create an interpolating polynomial generator."""
        super().__init__(x_values, y_values, symbol, inverse)
        self.__newton_differences = NewtonDifferences(self.x_values, self.y_values)
    
    def polynomial(self, degree, offset=0):
        """Generate an interpolating polynomial of specified degree."""
        if degree == 0:
            return self.y_values[degree+offset]
        else:
            return self.polynomial(degree - 1, offset=offset) + self.basis_polynomial(degree, offset=offset) * self.coefficient(degree, offset=offset)

    def practical_error_next_degree(self, degree, newton_differences=None):
        """Calculate the interpolation error using approximation. Returns a symbolic function."""
        return super().practical_error_next_degree(degree, self.__newton_differences)

    def coefficient(self, degree, index=0, offset=0):
        """Fetch Newton Difference coefficient of specified degree."""
        return self.__newton_differences.coefficient(degree, index, offset=offset)

    def coefficients(self, degree, offset=0):
        """Fetch Newton Difference coefficient table of specified degree."""
        return self.__newton_differences.coefficients(degree, offset=offset)

class LagrangeInterpolatingPolynomial(InterpolatingPolynomial):
    """Polynomial Interpolation using Lagrange Method."""

    def polynomial(self, degree, offset=0):
        """Generate an interpolating polynomial of specified degree."""
        return sym.N(sum([(self.coefficient(degree, index, offset=offset) * self.y_values[index + offset]) for index in range(degree + 1)]))

    def coefficient(self, degree, index, offset=0):
        """Generate Lagrangian coefficient of specified degree and index."""
        basis = self.basis_polynomial(degree + 1, skip=index, offset=offset)
        return basis/basis.subs(self.symbol, self.x_values[index + offset])

class InterpolatingSpline:
    """Base class for Spline Interpolation."""
    
    def __init__(self, x_values, y_values, symbol):
        """Create InterpolatingSpline class instance."""
        self.x_values = x_values
        self.y_values = y_values
        self.size = len(x_values)
        self.symbol = symbol

    @staticmethod
    def is_spline(functions, points, symbol, degree=None, places=5):
        """Check if the provided piecwise function (provided as a list of symbolic equations and a list of points) is a spline and calculate its' degree.
        
        If degree is provided, check if the spline is a spline of specified degree.
        """
        max_degree = max([sym.degree(x) for x in functions])
        precission = 10**(-places)
        current_degree = 0
        while True:
            max_difference = max([sym.Abs(f0.subs(symbol, xi) - f1.subs(symbol, xi)) for f0, f1, xi in zip(functions, functions[1:], points[1:])])
            if max_difference > precission:
                return (current_degree >= max_degree, current_degree) if degree is None else (current_degree == degree, current_degree)
            functions = [sym.diff(fx, symbol) for fx in functions]
            current_degree += 1

    def spline(self, natural=0):
        """Generate the spline. Non-functional for the base class."""
        return [(self.piece(index), Utility.interval(self.x_values[index], self.x_values[index + 1], self.symbol)) for index in range(self.size - 1)]

    def piece(self, index, natural=0):
        """Calculate a spline piece function, for values at specified index."""
        raise NotImplementedError("Using a base class. No method for spline piece generation.")

class LinearInterpolatingSpline(InterpolatingSpline):
    """Spline Interpolation using Linear Polynomials."""

    def piece(self, index, natural=0):
        """Calculate a spline piece function, for values at specified index. Calculated using Linear Interpolating Polynomial."""
        return LagrangeInterpolatingPolynomial(self.x_values, self.y_values, self.symbol).polynomial(1, offset=index)

class SquareInterpolatingSpline(InterpolatingSpline):
    """Spline Interpolation using Square Polynomials."""

    def __init__(self, x_values, y_values, symbol):
        """Construct class Instance. Define values to be memoized."""
        super().__init__(x_values, y_values, symbol)
        self.n = sym.Symbol('e')
        self.netwton_differences = NewtonDifferences(x_values, y_values)
        self.A = Utility.Memoized(self.A)
        self.B = Utility.Memoized(self.B)
        self.C = Utility.Memoized(self.C)

    def piece(self, index, natural=0):
        """Calculate a spline piece function, for values at specified index. Calculated using recursive parameter definition."""
        return sym.expand((self.A(index) * (self.symbol ** 2) + self.B(index) * self.symbol + self.C(index)).subs(self.n, natural))

    def A(self, index):
        """Recursively calculate parameter A at specified index."""
        if index == 0:
            delta_y = self.y_values[index + 1] - self.y_values[index]
            delta_x = self.x_values[index + 1] - self.x_values[index]
            return (delta_y - self.n * delta_x)/(delta_x**2)
        else:
            f = self.netwton_differences.coefficient(1, index)
            return (f - 2*self.A(index - 1)*self.x_values[index] - self.B(index - 1))/(self.x_values[index + 1] - self.x_values[index])

    def B(self, index):
        """Recursively calculate parameter B at specified index."""
        if index == 0:
            return self.n - 2*self.A(0)*self.x_values[0]
        else:
            return 2*(self.A(index - 1) - self.A(index)) * self.x_values[index] + self.B(index - 1)

    def C(self, index):
        """Recursively calculate parameter C at specified index."""
        if index == 0:
            return self.y_values[0] + self.A(0)*self.x_values[0]**2 - self.x_values[0] * self.n
        else:
            return self.y_values[index] + (self.A(index) - 2 * self.A(index - 1))*self.x_values[index]**2 - self.B(index - 1)*self.x_values[index]

class CubicInterpolatingSpline(InterpolatingSpline):
    """Spline Interpolation using Cubic Polynomials."""

    def __init__(self, x_values, y_values, symbol):
        """Construct class Instance. Define values to be memoized."""
        super().__init__(x_values, y_values, symbol)
        self.n = sym.Symbol('e')
        self.m = Utility.Memoized(self.m, { Utility.Memoized.key(0): 0, Utility.Memoized.key(self.size - 1): 0})
        self.a = Utility.Memoized(self.a, {})
        self.b = Utility.Memoized(self.b, {})

    def piece(self, index, natural=0):
        """Calculate a spline piece function, for values at specified index. Calculated using recursive parameter definition."""
        basis = self.symbol - self.x_values[index]
        return sym.expand(self.h(index)*basis**3 + self.g(index)*basis**2 + self.e(index)*basis + self.y_values[index])

    def m(self, index):
        """Recursively calculate parameter m at specified index."""
        if index == self.size - 2:
            return self.b(index)
        else:
            return self.a(index) * self.m(index+1) + self.b(index)

    def a(self, index):
        """Recursively calculate parameter alpha at specified index."""
        B = self.B(index)
        C = self.C(index)
        if index == 1:
            return -C/B
        else:
            A = self.A(index)
            return -C/(A*self.a(index - 1) + B)

    def b(self, index):
        """Recursively calculate parameter beta at specified index."""
        B = self.B(index)
        D = self.D(index)
        if index == 1:
            return D/B
        else:
            A = self.A(index)
            return (D - A*self.b(index - 1))/(A*self.a(index - 1) + B)

    """Calculate parameter h at specified index."""
    def h(self, index):
        return (self.m(index + 1) - self.m(index))/(6*Utility.gap(self.x_values, index))

    """Calculate parameter g at specified index."""
    def g(self, index):
        return self.m(index)/2

    """Calculate parameter e at specified index."""
    def e(self, index):
        return Utility.gap(self.y_values, index)/Utility.gap(self.x_values, index) - (Utility.gap(self.x_values, index)*(2*self.m(index) + self.m(index + 1)))/6

    """Calculate parameter A at specified index."""
    def A(self, index):
        return Utility.gap(self.x_values, index - 1)/6

    """Calculate parameter B at specified index."""
    def B(self, index):
        return (Utility.gap(self.x_values, index - 1) + Utility.gap(self.x_values, index))/3

    """Calculate parameter C at specified index."""
    def C(self, index):
        return Utility.gap(self.x_values, index)/6

    """Calculate parameter D at specified index."""
    def D(self, index):
        return Utility.gap(self.y_values, index)/Utility.gap(self.x_values, index) - Utility.gap(self.y_values, index - 1)/Utility.gap(self.x_values, index - 1)

class NewtonDifferences:

    def __init__(self, x_values, y_values):
        """Construct NewtonDifferences data class."""
        self.x_values = x_values
        self.y_values = y_values
        self.size = len(x_values)
        self.coefficient = Utility.Memoized(self.coefficient)
        self.coefficients = Utility.Memoized(self.coefficients)

    def coefficient(self, degree, index, *, offset=0):
        """Fetch Newton Difference coefficient of speicified degree and index."""
        if degree == 0:
            return self.y_values[index + offset]
        else:
            return (self.coefficient(degree - 1, index, offset=offset) - self.coefficient(degree - 1, index + 1, offset=offset))/(self.x_values[index + offset] - self.x_values[index + offset + degree])

    def coefficients(self, degree, *, offset=0):
        """Fetch Newton Difference coefficient table of speicified degree."""
        return [
            [self.coefficient(degree_index, item_index + offset) for item_index in range(item_count)]
            for degree_index, item_count
            in zip(range(degree + 1), reversed(range(1, degree + 2)))
        ]
