import sympy as sym
import numpy as np
import math

from numericalmethods import Utility

class GaussMethod:

    def __init__(self, A, f):
        self.matrix = [[*Ai, fi] for Ai, fi in zip(A, f)]
        self.row_count = len(self.matrix)
        self.column_count = len(self.matrix[0])
        self.forward()
        self.backward()

    def solve(self):
        return self.result

    def forward(self):
        for pi in range(self.row_count):
            for ri in range(pi + 1, self.row_count):
                if self.matrix[pi][pi] == 0:
                    self.matrix[pi], self.matrix[ri] = self.matrix[ri], self.matrix[pi]
                scale = self.matrix[ri][pi]/self.matrix[pi][pi]
                for ci in range(pi, self.column_count):
                    self.matrix[ri][ci] -= self.matrix[pi][ci] * scale
                
    def backward(self):
        self.result = []
        f = self.column_count - 1
        for pi in reversed(range(self.row_count)):
            result = self.matrix[pi][f]/self.matrix[pi][pi]
            for ri in reversed(range(pi)):
                self.matrix[ri][f] -= self.matrix[ri][pi] * result
                self.matrix[ri][pi] = 0
            self.result.append(result)
        self.result.reverse()


class RelocationMethod:

    def __init__(self, tridiagonal_matrix):
        self.tridiagonal_matrix = tridiagonal_matrix

        self.a = Utility.Memoized(self.a)
        self.b = Utility.Memoized(self.b)
        self.x = Utility.Memoized(self.x)

        self.result = []
        for i in reversed(range(0, len(self.tridiagonal_matrix.f))):
            self.result.append(self.x(i))
        self.result.reverse()

    def __divisor(self, index):
        return

    def a(self, index):
        if index == 0:
            return -self.tridiagonal_matrix.C[index]/self.tridiagonal_matrix.B[index]
        else:
            return -self.tridiagonal_matrix.C[index]/(self.tridiagonal_matrix.A[index]*self.a(index-1) + self.tridiagonal_matrix.B[index])

    def b(self, index):
        if index == 0:
            return self.tridiagonal_matrix.f[index]/self.tridiagonal_matrix.B[index]
        else:
            return (self.tridiagonal_matrix.f[index] - self.tridiagonal_matrix.A[index] * self.b(index-1))/(self.tridiagonal_matrix.A[index]*self.a(index-1) + self.tridiagonal_matrix.B[index])

    def x(self, index):
        if index == len(self.tridiagonal_matrix.f) - 1:
            return self.b(index)
        else:
            return self.a(index) * self.x(index + 1) + self.b(index)

    def alpha(self):
        return [(key[0][0], self.a.values[key]) for key in self.a.values]

    def beta(self):
        return [(key[0][0], self.b.values[key]) for key in self.b.values]

    def solve(self):
        return self.result


class CholeskyMethod:

    def __init__(self, A, f):
        self.f = np.array(f)
        self.A = np.array(A)
        self.L = self.decompose()
        self.Y = np.linalg.solve(self.L, self.f)
        self.X = np.linalg.solve(np.transpose(self.L), self.Y)

    def solve(self):
        return self.X

    def decompose(self):
        L = np.zeros(self.A.shape)
        for i, (Ai, Li) in enumerate(zip(self.A, L)):
            for j, Lj in enumerate(L[:i+1]):
                sum_difference = Ai[j] - sum(Li[k] * Lj[k] for k in range(j))
                L[i][j] = math.sqrt(sum_difference) if i == j else 1.0/Lj[j] * sum_difference
        return L


