
import numpy as np
from scipy.optimize import linprog

def one_way_nash(A): # row player minimizing, returns row player policy
    N = len(A)
    assert N == len(A[0])

    A_ub = np.hstack((A.T, -np.ones((N, 1))))
    b_ub = np.zeros(N)

    A_eq = np.ones((1, N+1))
    A_eq[0][N] = 0
    b_eq = np.ones((1, 1))

    c = np.zeros(N+1)
    c[N] = 1

    bounds = [(0, 1)] * N + [(None, None)]

    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds)

    return res.x

def getNash(A):
    # row player chooses row, col player chooses column
    # row player minimizing, col player maximizing
    # returns
    #   row player policy
    #   column player policy
    #   value
    N = len(A)
    row = one_way_nash(A)
    col = one_way_nash(-A.T)
    assert abs(row[N] + col[N]) < 1e-07
    return row[0:N], col[0:N], row[N]

A = np.array([
    [0,1,3],
    [4,2,1],
    [1,3,1]
])

# print(getNash(A))