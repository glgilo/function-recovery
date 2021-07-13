import math
import time
from matplotlib import pyplot as plt
import numpy as np
import utilities


class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def setR(self, R):
        self.R = R

    def __eq__(self, other):
        if self.x == other.x and self.y == other.y:
            return True
        return False

    def __lt__(self, other):
        return self.R < other.R

    def calcDist(self, other):
        return math.dist([self.x, self.y], [other.x, other.y])


class AverageSmoothnessLearner:
    xs: list
    ys: list
    net: list
    epsilon: float
    L: int

    def __init__(self, l=0, epsilon=0.1, xs=None, ys=None):
        self.L = l
        self.epsilon = epsilon
        self.xs = xs
        self.ys = ys
        self.net = []

    def train(self, xs=None, ys=None):
        # if received new data:
        if xs is not None and ys is not None:
            self.xs = xs
            self.ys = ys

        # Linear program:
        smoothed_points = utilities.smooth_input(self.xs, self.ys, self.L)

        # throw epsilon percent points, store with new y values (e.g. zs)
        points = throwEpsilon([Point(p[0], p[1]) for p in smoothed_points], self.epsilon)

        # build epsilon net:
        self.net = buildNet(points, self.epsilon)
        print("Finished train")

    def PMSE(self, x):
        """
        Predict y, based on the paper
        :param x: input x
        :return: prediction y
        """
        R_max = -1
        u_star = Point(0, 0)
        v_star = Point(0, 0)
        for u in self.net:
            for v in self.net:
                if u != v:
                    R_temp = calculate_Rx(x, u, v)
                    if R_temp > R_max:
                        R_max = R_temp
                        u_star = u
                        v_star = v
        return fx(x, u_star, v_star)

    def predict(self, xtest, ytest):
        predictions = []
        for x, y in zip(xtest, ytest):
            predictions.append(self.PMSE(x))

        return predictions

    def calc_loss(self, predictions, label):
        total_error = 0
        for p, l in zip(predictions, label):
            error = math.sqrt((p - l)**2)
            total_error += error
            print("predicted=", p, " label=", l, "error=", error)

        print("\n---------------------------------\nTotal error = {}\nAverage error = {} "
              "\n---------------------------------\n".format(total_error, (total_error / len(predictions))))
        return total_error

    def test(self, xtest, ytest):
        t0 = time.time()
        predictions = self.predict(xtest, ytest)
        predict_time = time.time() - t0
        print("ASL prediction for %d inputs in %.3f s\n"
              % (len(xtest), predict_time))

        self.plot_predictions(xtest, predictions)
        return self.calc_loss(predictions, ytest)

    def plot_predictions(self, xtest, preds):
        # x = np.arange(0, 1, 0.05)
        x = np.linspace(0, 1)
        plt.scatter(self.xs, self.ys, c='b', label='Train sample')
        plt.plot(xtest, preds, c='r', label='Average smoothness learner')
        plt.scatter(xtest, utilities.func_sin(xtest), c='g', label='Test sample')
        plt.plot(x, utilities.func_sin(x), c='y', label='sin(2Pi*x)')
        plt.xlabel('x - axis')
        plt.ylabel('y - axis')
        plt.title(' ASL n={}, L={} epsilon={}'.format(len(self.xs), self.L, self.epsilon))
        plt.legend()
        plt.show()


def calculateR(u, v):
    if u.x == v.x:
        return 0
    return (u.y - v.y) / (u.x - v.x)  # abs?


def throwEpsilon(points, epsilon):
    """
    Throw epsilon * (size of points set) from all points, with the highest R_X values
    :param points: points after the smoothing phase
    :param epsilon: percentage of highest R_x values out of the points list
    :return: new set of points, size = (1 - epsilon) * n
    """
    for u in points:
        r_max = -1
        for v in points:
            if u != v:
                R = calculateR(u, v)
                if R > r_max:
                    r_max = R
                    u.setR(R)
    points.sort()
    new_points_set = []
    for i in range(0, round(len(points) * (1 - epsilon))):
        new_points_set.append(points[i])
    return new_points_set


def buildNet(points, epsilon):
    net = []
    for point in points:
        covered = False
        for net_point in net:
            if point.calcDist(net_point) < epsilon:
                covered = True
                break
        if not covered:
            net.append(point)
    return net


def calculate_Rx(x, u, v):
    return (v.y - u.y) / (math.dist([x], [u.x]) + math.dist([x], [v.x]))  # abs?


def fx(x, u, v):
    return u.y + calculate_Rx(x, u, v) * math.dist([x], [u.x])


def test_learner(n=32, L=10, epsilon=0.1, std_error=0.001, func_type=0):
    print('Test learner for n={}, L={} epsilon={} std_error:'.format(n, L, epsilon, std_error))
    [xtrain, ytrain], [xtest, ytest] = utilities.generate_experiment(func_type, n, std_error)
    learner = AverageSmoothnessLearner(L, epsilon, xtrain, ytrain)
    learner.train()
    learner.test(xtest, ytest)
