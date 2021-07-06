# import torch
# import torch.nn as nn
# import torch.nn.functional as F
#
#
# class Net(nn.Module):
#
#     def __init__(self):
#         super(Net, self).__init__()
#         # 1 input image channel, 6 output channels, 5x5 square convolution
#         # kernel
#         self.conv1 = nn.Conv2d(1, 6, 5)
#         self.conv2 = nn.Conv2d(6, 16, 5)
#         # an affine operation: y = Wx + b
#         self.fc1 = nn.Linear(16 * 5 * 5, 120)  # 5*5 from image dimension
#         self.fc2 = nn.Linear(120, 84)
#         self.fc3 = nn.Linear(84, 10)
#
#     def forward(self, x):
#         # Max pooling over a (2, 2) window
#         x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
#         # If the size is a square, you can specify with a single number
#         x = F.max_pool2d(F.relu(self.conv2(x)), 2)
#         x = torch.flatten(x, 1) # flatten all dimensions except the batch dimension
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x
import math
import operator
from utilities import generate_experiment
from averagesmoothnesslearner import AverageSmoothnessLearner
from averagesmoothnesslearner import *

# [xtrain, ytrain], [xtest, ytest] = utilities.generate_experiment(0, 32)
# print(xtrain, ytrain, xtest, ytest)
# utilities.test_linear_program()

if __name__ == '__main__':
    test_learner(300)

    # A = Point(1,2)
    # B = Point(3, 4)
    # C = Point(1, 7)
    # D = Point(13, 5)
    # E = Point(2, 23)
    # pointss = []
    # pointss.append(A)
    # pointss.append(B)
    # pointss.append(C)
    # pointss.append(D)
    # pointss.append(E)
    # print(pointss)
    # new = throwEpsilon(pointss,0.2)
    # for point in new:
    #     print(point.x, point.y)
    # new_new = buildNet(new, 5)
    #
    # for point in new_new:
    #     print(point.x, point.y)
    # print(PMSE(2,new_new))
    #

    # net = Net()
    # print(net)
