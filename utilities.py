import math
from math import pi
import numpy as np
from numpy import random
import pulp
from pulp import LpMinimize, LpProblem, LpStatus, lpSum, LpVariable


def func_sign(x, std_noise=0):
    return np.sign(np.sin(2 * pi * x)) + std_noise * random.uniform(-1, 1)


def func_sin(x, std_noise=0):
    return np.sin(2 * pi * x) + std_noise * random.uniform(-1, 1)



def create_sign_dataset(n, std_noise=0):
    # xs = random.random(n)
    xs = [random.uniform(-1, 1) for i in range(n)]
    xs.sort()
    ys = [func_sign(x, std_noise) for x in xs]
    return xs, ys


def create_sin_dataset(n, std_noise=0):
    xs = [random.uniform(-1, 1) for i in range(n)]
    xs.sort()
    ys = [func_sin(x, std_noise) for x in xs]
    return xs, ys


def generate_experiment(func_type=0, n1=32, n2=32, n3=32, std_noise=0.1):
    """
        :param std_noise: std of noise around func
        :param func_type: 0 = sin, 1 = sign
        :param n: dataset size
        :return: train & test
        """
    std_noise = 0
    if func_type == 0:
        return create_sin_dataset(n1, std_noise),create_sin_dataset(n2, std_noise), create_sin_dataset(n3)

    if func_type == 1:
        return create_sign_dataset(n1, std_noise), create_sign_dataset(n2, std_noise), create_sign_dataset(n3)

    print("Wrong func_type")
    exit(1)
    return -1


def smooth_input(xs, ys, L):
    """
    Create & solve a linear programming model, minimize the sum of w_i
    s.t:
     1. sum of Li <= L,                         0 <= i <= n
     2. w_i >= |z_i - y_i|,                     0 <= i <= n
     3. |z_i - z_j| <= L_i * dist(x_i, x_j)     0 <= i,j <= n
     4. 0 <= w_i, z_i <= 1                      0 <= i <= n
    :param xs: x values of the given input
    :param ys: x values of the given input
    :param L: Liphshitz upper bound
    :return: A list of points after smoothing (e.g. (x, z) for each (x,y)) or terminate if no solution exist
    """
    n = len(xs)

    # obj = [1 for i in range(n)]
    # for i in range(2 * n):
    #     obj.append(0)

    # Create the model
    model = LpProblem(name="small-problem", sense=LpMinimize)
    ws = [LpVariable(name="w_{}".format(i), lowBound=0, upBound=1) for i in range(n)]
    ls = [LpVariable(name="L_{}".format(i), lowBound=0) for i in range(n)]
    zs = [LpVariable(name="z_{}".format(i)) for i in range(n)]

    # objective
    model += lpSum(ws)

    # constraint 1:
    # sum of Li <= L
    model += (lpSum(ls) <= L * n, "sum of Li <= L")

    # Constraint 2:
    # w_i >= |z_i - y_i|
    for i in range(n):
        model += (ws[i] + zs[i] >= ys[i], "C2.a_{}".format(i))
        model += (ws[i] - zs[i] >= -ys[i], "C2.b_{}".format(i))

    # Constraint 3
    # |z_i - z_j| <= L_i * dist(x_i, x_j)
    for i in range(n):
        for j in range(n):
            if i != j:
                model += (zs[i] - zs[j] - abs(xs[i] - xs[j]) * ls[i] <= 0, "C3.a_{}_{}".format(i, j))
                model += (zs[j] - zs[i] - abs(xs[i] - xs[j]) * ls[i] <= 0, "C3.b_{}_{}".format(i, j))

    if model.solve() == 1:
        print(
            "------------------------------------\nFound solution for the linear program\n------------------------------------\n")
        return [[xs[i], zs[i].value()] for i in range(n)]
        # return [zi.value() for zi in zs], [li.value() for li in ls]

    print("Linear program: no solution found")
    exit(1)
    return -1


def calc_squared_loss(predictions, label):
    total_error = 0
    for p, l in zip(predictions, label):
        error = math.sqrt((p - l) ** 2)
        total_error += error
        # print("predicted=", p, " label=", l, "error=", error)

    # print("\n---------------------------------\nTotal error = {}\nAverage error = {} "
    #       "\n---------------------------------\n".format(total_error, (total_error / len(predictions))))
    return total_error

def test_linear_program():
    xs, ys = create_sin_dataset(32)
    print("Start train()")
    # zs, ls = smooth_input(xs, ys, 100)
    points = smooth_input(xs, ys, 100)
    print(points)
    # print("xs:", xs,"\nys:", ys,"\nzs:",zs,"\nls",ls)

    exit(0)
    return 0
