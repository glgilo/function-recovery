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


class Point():
    def __init__(self,x , y):
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
        return math.dist([self.x,self.y], [other.x, other.y])


def calculateR(u, v):
    if u.x == v.x:
        return 0
    return (u.y - v.y)/ (u.x - v.x) # abs?


def throwEpsilon(points, epsilon):
    for point1 in points:
        max = -1
        for point2 in points:
            if(point1 != point2):
                R = calculateR(point1,point2)
                if R > max:
                    max = R
                    point1.setR(R)
    points.sort()
    new_points_set = []
    for i in range(0, round(len(points)*(1-epsilon))):
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
    return (v.y - u.y)/(math.dist([x], [u.x]) + math.dist([x], [v.x])) #abs?


def fx(x, u, v):
    return u.y + calculate_Rx(x, u, v)*math.dist([x], [u.x])


def PMSE(x, points):
    maxR = -1
    max_point1 = Point(0,0)
    max_point2 = Point(0, 0)
    for point1 in points:
        for point2 in points:
            if point1 != point2:
                tempR = calculate_Rx(x, point1, point2)
                if tempR > maxR:
                    maxR = tempR
                    max_point1 = point1
                    max_point2 = point2
    return fx(x, max_point1, max_point2)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    A = Point(1,2)
    B = Point(3, 4)
    C = Point(1, 7)
    D = Point(13, 5)
    E = Point(2, 23)
    pointss = []
    pointss.append(A)
    pointss.append(B)
    pointss.append(C)
    pointss.append(D)
    pointss.append(E)
    print(pointss)
    new = throwEpsilon(pointss,0.2)
    for point in new:
        print(point.x, point.y)
    new_new = buildNet(new, 5)

    for point in new_new:
        print(point.x, point.y)
    print(PMSE(2,new_new))


    # net = Net()
    # print(net)
