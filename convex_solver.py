
import cvxpy as cp
import numpy as np

def test(size):
    vars = []
    for i in range(0, size):
        vars.append(cp.Variable())
    sum = 0
    for i in range(size):
        sum = sum + vars[i]
    constraints = [sum == 1]
    value = 0
    for i in range(size):
        value = value + (i+1) * vars[i]**2
    obj = cp.Minimize(value)
    print(cp.Problem(obj, constraints).solve())

# test(10000)

def test2(size):
    var = cp.Variable(size)
    constraints = [cp.sum(var) == 1]
    c = np.zeros(size)
    for i in range(size):
        c[i] = i+1
    obj = cp.Minimize(c.T @ var**2)
    print(cp.Problem(obj, constraints).solve())

# test2(10000)

def test3(A):
    n, m = A.shape
    var = cp.Variable(m)
    constraints = [cp.sum(var) == 1, cp.min(var) >= 0]

    obj = cp.Maximize(cp.min(A @ var))
    print(cp.Problem(obj, constraints).solve())

# test3(np.array([
#     [1, 2, 3],
#     [4, 2, 1]
# ]))

def test4():
    # var = cp.Variable()
    # exp = [1, var]
    # for i in range(40):
    #     exp.append(exp[i] + exp[i+1])
    # constraints = []
    # obj = cp.Minimize(exp[11]**2)
    # print(cp.Problem(obj, constraints).solve())
    vars = []
    constraints = []
    for i in range(40):
        vars.append(cp.Variable())
        if i >= 2:
            constraints.append(vars[i] == vars[i-1] + vars[i-2])
    constraints.append(vars[0] == 1)
    obj = cp.Minimize(vars[39]**2)
    print(cp.Problem(obj, constraints).solve())

# test4()

def test5():
    a = cp.Variable()
    b = a + 3
    c = a * b
    constraints = []
    obj = cp.Minimize(c)
    print(cp.Problem(obj, constraints).solve())

test5()