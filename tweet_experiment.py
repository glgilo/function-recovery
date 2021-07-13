
import time
import array
import numpy as np
import utilities
from sklearn.kernel_ridge import KernelRidge
from sklearn.neural_network import MLPRegressor
import matplotlib.pyplot as plt

import os
import imageio


# alpha = 0.1

def run_test():
    filenames = []
    for alpha in np.arange(0, 0.0001, 0.00001):
        mlp = MLPRegressor(alpha=alpha, max_iter=1000, hidden_layer_sizes=(1000))
        krr = KernelRidge(alpha=alpha, kernel='rbf')

        [xtrain, ytrain], [xtest, ytest] = utilities.generate_experiment(0, 32, 0.1)

        xtrain = np.array(xtrain).reshape(-1, 1)
        ytrain = np.array(ytrain).reshape(-1, 1)
        xtest = np.array(xtest).reshape(-1, 1)
        ytest = np.array(ytest).reshape(-1, 1)

        mlp.score()

        mlp.fit(xtrain, ytrain)
        mlp_predictions = mlp.predict(xtest).reshape(-1, 1)

        krr.fit(xtrain, ytrain)
        krr_predictions = krr.predict(xtest).reshape(-1, 1)

        print('prediction = {}\nytest = {}\nscore = {}'.format(mlp_predictions, ytest, mlp.score(xtest, ytest)))


        x = np.linspace(0, 1)
        plt.scatter(xtrain, ytrain, c='b', label='Train sample')
        plt.scatter(xtest, utilities.func_sin(xtest), c='g', label='Test sample')
        plt.plot(xtest, mlp_predictions, c='r', label='mlp')
        plt.plot(xtest, krr_predictions, c='k', label='krr')

        plt.plot(x, utilities.func_sin(x), c='y', label='sin(2Pi*x)')
        plt.xlabel('x - axis')
        plt.ylabel('y - axis')
        # plt.title(' ASL n={}, L={} epsilon={}'.format(len(self.xs), self.L, self.epsilon))
        plt.title('alpha = {}'.format(alpha))
        plt.legend()
        # plt.show()

        filename = f'images/frame_{alpha}.png'
        filenames.append(filename)
        # save img
        plt.savefig(filename)
        plt.close()

    print('Charts saved\n')
    # Build GIF
    print('Creating gif\n')
    with imageio.get_writer('mybars.gif', mode='I') as writer:
        for filename in filenames:
            image = imageio.imread(filename)
            writer.append_data(image)
    print('Gif saved\n')
    print('Removing Images\n')
    # Remove files
    for filename in set(filenames):
        os.remove(filename)
    print('DONE')

run_test()
