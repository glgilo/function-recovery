import time
import array
import numpy as np
import utilities
import sklearn as sk
from sklearn.kernel_ridge import KernelRidge
from sklearn.neural_network import MLPRegressor
import matplotlib.pyplot as plt

import os
import imageio


# alpha = 0.1

def create_mlp_krr_fig(alpha, filenames, krr_predictions, mlp_predictions, xtrain, ytrain, xtest, ytest, func_type=0):
    # set function to desired func
    func = utilities.func_sin
    if func_type == 1:
        func = utilities.func_sign

    x = np.linspace(-1, 1, 100)
    plt.scatter(xtrain, ytrain, c='b', marker='x' , label='Train sample')
    plt.scatter(xtest, ytest, c='g', marker='s', label='Validation sample')
    plt.plot(xtest, mlp_predictions, c='g', label='mlp')
    plt.plot(xtest, krr_predictions, c='k', label='krr')
    plt.plot(x, func(x), c='r', label='sin(2Pi*x)')
    plt.xlabel('x - axis')
    plt.ylabel('y - axis')
    # plt.title(' ASL n={}, L={} epsilon={}'.format(len(self.xs), self.L, self.epsilon))
    plt.title('n={} Krr and Mlp \nalpha ={}'.format(len(xtrain), alpha))
    plt.legend()
    # plt.show()
    filename = f'images/tweet_frame_{alpha}.png'
    for i in range(2):
        filenames.append(filename)
    # save img
    plt.savefig(filename)
    plt.close()


def generate_mlp_krr_gif(filenames):
    # Build GIF
    print('Creating KRR MLP gif\n')
    with imageio.get_writer('krr_mlp.gif', mode='I') as writer:
        for filename in filenames:
            image = imageio.imread(filename)
            writer.append_data(image)
    print('Gif krr mlp saved\n')
    print('Removing Images\n')
    # Remove files
    for filename in set(filenames):
        os.remove(filename)
    print('DONE\n')


def run_krr_mlp(alpha, xtrain, ytrain, xtest, ytest, case):
    if case == 0:
        mlp = MLPRegressor(alpha=alpha, max_iter=1000, hidden_layer_sizes=(1000))
        krr = KernelRidge(alpha=alpha, kernel='rbf')
    else:
        mlp = MLPRegressor(alpha=alpha[0], max_iter=1000, hidden_layer_sizes=(1000))
        krr = KernelRidge(alpha=alpha[1], kernel='rbf')
    mlp.fit(xtrain, ytrain)
    mlp_predictions = mlp.predict(xtest).reshape(-1, 1)
    krr.fit(xtrain, ytrain)
    krr_predictions = krr.predict(xtest).reshape(-1, 1)
    # ------- just debug --------
    # sk.model_selection.KFold
    # ---------------------------
    return krr_predictions, krr.score(xtest, ytest), mlp_predictions, mlp.score(xtest, ytest)


def experiment_mlp_krr(xlearn, ylearn, xvalid, yvalid, xtest, ytest, regularisation_variables, func_type=0):
    filenames = []
    mlp_best_result = {'alpha': 0, 'xtrain': [], 'ytrain': [], 'xtest': [], 'ytest': [],'predictions': [], 'score': 0}
    krr_best_result = {'alpha': 0, 'xtrain': [], 'ytrain': [], 'xtest': [], 'ytest': [],'predictions': [], 'score': 0}
    xcombine = xlearn + xvalid
    ycombine = ylearn + yvalid

    xlearn = np.array(xlearn).reshape(-1, 1)
    ylearn = np.array(ylearn).reshape(-1, 1)
    xvalid = np.array(xvalid).reshape(-1, 1)
    yvalid = np.array(yvalid).reshape(-1, 1)
    # for each alpha, run train & test, and save the optimal one (based on squared loss)
    for alpha in regularisation_variables:

        krr_predictions, krr_score, mlp_predictions, mlp_score = run_krr_mlp(alpha, xlearn, ylearn, xvalid, yvalid, 0)
        create_mlp_krr_fig(alpha, filenames, krr_predictions, mlp_predictions, xlearn, ylearn, xvalid, yvalid, func_type)

        # save best results for each method
        if krr_score > krr_best_result['score']:
            krr_best_result = {'alpha': alpha, 'xtrain': xlearn, 'ytrain': ylearn, 'xtest': xvalid, 'ytest': yvalid,
                               'predictions': krr_predictions, 'score': krr_score}
        if mlp_score > mlp_best_result['score']:
            mlp_best_result = {'alpha': alpha, 'xtrain': xlearn, 'ytrain': ylearn, 'xtest': xvalid, 'ytest': yvalid,
                               'predictions': mlp_predictions, 'score': mlp_score}

    # create gif for data visualisation


    xtrain = np.array(xcombine).reshape(-1, 1)
    ytrain = np.array(ycombine).reshape(-1, 1)
    xtest = np.array(xtest).reshape(-1, 1)
    ytest = np.array(ytest).reshape(-1, 1)

    krr_predictions, krr_score, mlp_predictions, mlp_score = run_krr_mlp([mlp_best_result['alpha'], krr_best_result['alpha']],
                                                                         xtrain, ytrain, xtest, ytest, 1)
    krr_best_result = {'alpha': krr_best_result['alpha'], 'xtrain': xtrain, 'ytrain': ytrain, 'xtest': xtest, 'ytest': ytest,
                       'predictions': krr_predictions, 'score': krr_score}
    mlp_best_result = {'alpha': mlp_best_result['alpha'], 'xtrain': xtrain, 'ytrain': ytrain, 'xtest': xtest, 'ytest': ytest,
                       'predictions': mlp_predictions, 'score': mlp_score}

    print('plots saved\n')
    generate_mlp_krr_gif(filenames)

    print("Best results:\nmlp: alpha={} score={}\nkrr: alpha={} score={}\n".format(mlp_best_result['alpha'],
                                                                                 mlp_best_result['score'],
                                                                                 krr_best_result['alpha'],
                                                                                 krr_best_result['score']))
    return mlp_best_result, krr_best_result




# if __name__ == '__main__':
#     [xtrain, ytrain], [xtest, ytest] = utilities.generate_experiment(0, 32, 0.1)
#     experiment_mlp_krr(xtrain, ytrain, xtest, ytest, 32, 0.1)
