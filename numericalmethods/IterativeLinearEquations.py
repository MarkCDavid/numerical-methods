"""Iterative Linear Equation module, designed to assist solving excercises in university course for Numerical Methods."""

import sympy as sym
import numpy as np
import math

from numericalmethods import Utility

class IterativeLinearEquation:
    """Base class for ILE."""

    def __init__(self, A, f):
        """Calculate base values."""
        self.A = np.array(A)
        self.f = np.array(f)
        self.S, self.B = self.S_B()
        self.q = Utility.Norms.absolute_eigenvalue(self.S)
        self.heavy_diagonal = all([abs(Ai[i]) > sum(Aij for j, Aij in enumerate(Ai) if i != j) for i, Ai in enumerate(self.A)])
        

    def solve(self, x0, condition):
        """Solve for given x0 and condition."""
        X = [np.array(x0)]
        iteration = 0
        while not condition.check(self.precision(X, iteration)):
            X.append(self.iterate(X[iteration]))
            iteration += 1
        return (iteration, X[iteration], self.precision(X, iteration))

    def iterations(self, precision):
        """Calulate number of iterations required until specified precision reached."""
        return np.log(precision)/np.log(self.q)

    def precision(self, X, iteration):
        """Calulate precision of specified iteration."""
        return np.absolute(X[iteration] - X[iteration - 1]).max() if iteration != 0 else np.inf

class JacobiILE(IterativeLinearEquation):
    """ILE solver using Jacobi Method."""

    def __init__(self, A, f):
        """Calculate additional value for max ratio."""
        super().__init__(A, f)
        self.max_ratio = max([sum(abs(Aij/Ai[i]) for j, Aij in enumerate(Ai) if i != j) for i, Ai in enumerate(self.A)])
    
    def S_B(self):
        """Compute S and B matrices."""
        S = np.zeros(self.A.shape)
        B = np.zeros(self.f.shape)
        for i, Ai in enumerate(self.A):
            for j, Aij in enumerate(Ai):
                S[i][j] = -Aij/Ai[i] if i != j else 0
            B[i] = self.f[i]/Ai[i]
        return (S, B)
    
    def iterate(self, Xi):
        """Compute next iteration value."""
        x = np.zeros(self.B.shape)
        for i, Bi in enumerate(self.B):
            x[i] = Bi
            for j, Sij in enumerate(self.S[i]):
                x[i] += Sij*Xi[j]
        return x

    def converges(self):
        """Check if the solver converges given initial parameters."""
        return self.heavy_diagonal and self.max_ratio < 1

    def converges_strict(self):
        """Check if the solver converges given initial parameters."""
        return self.q < 1

class ZeidelILE(IterativeLinearEquation):
    """ILE solver using Zeidel Method."""

    def __init__(self, A, f):
        """Calculate additional value - matrix is positive definite."""
        super().__init__(A, f)
        self.positive_definite = np.all(np.linalg.eigvals(self.A) > 0)
    
    def S_B(self):
        """Compute S and B matrices."""
        diagonal = np.diag(np.diag(self.A))
        lowerTriangle = np.tril(self.A) - diagonal
        upperTriangle = np.triu(self.A) - diagonal
        inverse_DL = np.linalg.inv(diagonal + lowerTriangle)
        return (np.matmul(-inverse_DL, upperTriangle), np.matmul(inverse_DL, self.f))
    
    def iterate(self, Xi):
        """Compute next iteration value."""
        return np.matmul(self.S, Xi) + self.B

    def converges(self):
        """Check if the solver converges given initial parameters."""
        return self.heavy_diagonal and self.positive_definite

    def converges_strict(self):
        """Check if the solver converges given initial parameters."""
        return self.q < 1 and self.positive_definite

class FixedPointILE(IterativeLinearEquation):
    """ILE solver using Fixed Point Method."""

    def __init__(self, A, f):
        """Calculate additional value - tau."""
        eigenvalues = Utility.Norms.eigenvalues(A)
        self.tau = 2 / (min(eigenvalues) + max(eigenvalues))
        super().__init__(A, f)

    def S_B(self):
        """Compute S and B matrices."""
        return (np.identity(self.A.shape[0]) - self.tau * self.A,  self.tau * self.f)

    def iterate(self, Xi):
        """Compute next iteration value."""
        return np.matmul(self.S, Xi) + self.B

    def converges(self):
        """Check if the solver converges given initial parameters."""
        return self.tau > 0 and self.tau < (2 / self.q)

    def converges_strict(self):
        """Check if the solver converges given initial parameters."""
        return self.q < 1