import sympy as sym

class CustomAssertions:
    def assertSymPyEqual(self, expected, calculated):
        if sym.simplify(expected - calculated) != 0:
            raise AssertionError(f'Expected: {expected} | Got: {got}')