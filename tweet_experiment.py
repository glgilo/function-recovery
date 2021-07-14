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

def run_test(n=32, std_noise=0.1):
    filenames = []
    mlp_best_result = {'alpha': -1, 'xtrain': [], 'ytrain': [], 'xtest': [], 'ytest': [], 'score': 0}
    krr_best_result = {'alpha': -1, 'xtrain': [], 'ytrain': [], 'xtest': [], 'ytest': [], 'score': 0}

    for alpha in np.arange(0, 0.0001, 0.00001):
        [xtrain, ytrain], [xtest, ytest] = utilities.generate_experiment(0, n, std_noise)

        xtrain = np.array(xtrain).reshape(-1, 1)
        ytrain = np.array(ytrain).reshape(-1, 1)
        xtest = np.array(xtest).reshape(-1, 1)
        ytest = np.array(ytest).reshape(-1, 1)

        krr_predictions, krr_score, mlp_predictions, mlp_score = run_krr_mlp(alpha, xtrain, ytrain, xtest, ytest)
        create_fig(alpha, filenames, krr_predictions, mlp_predictions, xtest, xtrain, ytrain)

        # save best results for each method
        if krr_score > krr_best_result['score']:
            krr_best_result = {'alpha': alpha, 'xtrain': xtrain, 'ytrain': ytrain, 'xtest': xtest, 'ytest': ytest,
                               'score': krr_score}
        if mlp_score > mlp_best_result['score']:
            mlp_best_result = {'alpha': alpha, 'xtrain': xtrain, 'ytrain': ytrain, 'xtest': xtest, 'ytest': ytest,
                               'score': mlp_score}

    print('Charts saved\n')
    generate_gif(filenames)

    print("Best results:\nmlp: alpha={} score={}\nkrr: alpha={} score={}".format(mlp_best_result['alpha'],
                                                                                 mlp_best_result['score'],
                                                                                 krr_best_result['alpha'],
                                                                                 krr_best_result['score']))
    return mlp_best_result, krr_best_result


def create_fig(alpha, filenames, krr_predictions, mlp_predictions, xtest, xtrain, ytrain):
    # print('prediction = {}\nytest = {}\nscore = {}'.format(mlp_predictions, ytest, mlp.score(xtest, ytest)))

    x = np.linspace(0, 1)
    plt.scatter(xtrain, ytrain, c='b', marker='x' , label='Train sample')
    # plt.scatter(xtest, utilities.func_sin(xtest), c='g', label='Test sample')
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
    for i in range(10):
        filenames.append(filename)
    # save img
    plt.savefig(filename)
    plt.close()


def generate_gif(filenames):
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
    print('DONE\n')


def run_krr_mlp(alpha, xtrain, ytrain, xtest, ytest):
    mlp = MLPRegressor(alpha=alpha, max_iter=1000, hidden_layer_sizes=(1000))
    krr = KernelRidge(alpha=alpha, kernel='rbf')
    mlp.fit(xtrain, ytrain)
    mlp_predictions = mlp.predict(xtest).reshape(-1, 1)
    krr.fit(xtrain, ytrain)
    krr_predictions = krr.predict(xtest).reshape(-1, 1)
    return krr_predictions, krr.score(xtest, ytest), mlp_predictions, mlp.score(xtest, ytest)


run_test(32, 0.1)
