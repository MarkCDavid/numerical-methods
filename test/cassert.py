import sympy as sym

class CustomAssertions:
    def assertSymPyEqual(self, expected, calculated):
        if sym.simplify(expected - calculated) != 0:
            raise AssertionError(f'Expected: {expected} | Got: {calculated}')

    def assertSymPyListEqual(self, expected, calculated):
        for pair in zip(expected, calculated):
            if sym.simplify(pair[0] - pair[1]) != 0:
                raise AssertionError(f'Expected: {expected} | Got: {calculated}')

    def assertSymPyListAlmostEqual(self, expected, calculated, places=4):
        def round_expession(expression):
            result = expression
            for part in sym.preorder_traversal(expression):
                if isinstance(part, sym.Float):
                    result = result.subs(part, round(part, places))
            return result

        
        for pair in zip(expected, calculated):
            if sym.simplify(round_expession(pair[0]) - round_expession(pair[1])) != 0:
                raise AssertionError(f'Expected: {pair[0]} | Got: {calculated}')