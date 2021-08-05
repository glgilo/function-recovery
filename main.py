from averagesmoothnesslearner import *
from utilities import *
from tweet_experiment import *
import random





def plot_best_result():
    global x
    # create plot of best result:
    func = utilities.func_sin
    if func_type == 1:
        func = utilities.func_sign

    x = np.linspace(-1, 1, 100)
    # plt.scatter(xtrain, ytrain, c='b', marker='x', label='Train sample')
    # plt.scatter(xtest, ytest, c='g', marker='s', label='Test sample')
    plt.plot(xtest, mlp_best_result['predictions'], c='g', label='mlp')
    plt.plot(xtest, krr_best_result['predictions'], c='m', label='krr')
    plt.plot(xtest, asl_best_result['predictions'], c='k', label='Average smoothness learner')
    plt.plot(x, func(x), c='r', label='sin(2Pi*x)')
    plt.xlabel('x - axis')
    plt.ylabel('y - axis')
    # plt.title(' ASL n={}, L={} epsilon={}'.format(len(self.xs), self.L, self.epsilon))
    plt.title('n={}, ASL: L={} epsilon={}\nKRR: alpha = {}, MLP: alpha={}'.format(n_learn+n_valid, asl_best_result['L'],
                                                                                  asl_best_result['epsilon'],
                                                                                  krr_best_result['alpha'],
                                                                                  mlp_best_result['alpha']))
    plt.legend()
    plt.show()


def compare_squared_loss():
    # calc squared loss:
    krr_sl = calc_squared_loss(krr_best_result['predictions'], krr_best_result['ytest'])
    mlp_sl = calc_squared_loss(mlp_best_result['predictions'], mlp_best_result['ytest'])
    print('Squared loss (best result):\nasl={}\nkrr={}\nmlp={}'.format(asl_best_result['squared_loss'], krr_sl, mlp_sl))

def split_train(xtrain, ytrain, num):
    randomlist = random.sample(range(0, len(xtrain)), num)
    xlearn, ylearn, xvalid, yvalid = [], [], [], []
    for i in range(len(xtrain)):
        if i in randomlist:
            xvalid.append(xtrain[i])
            yvalid.append(ytrain[i])
        else:
            xlearn.append(xtrain[i])
            ylearn.append(ytrain[i])
    return xlearn, ylearn, xvalid, yvalid

if __name__ == '__main__':

    # In order to run the test, please:
    #
    #  1. set function as:
    #     0 := sin(2*pi*x), n = 32
    #     1 := sign(sin(2*pi*x)),
    #
    #  2. set n (default n=32)
    #  3. set L values and lambda values

    func_type = 1
    n_learn = 64
    n_valid = 64
    n_test = 64
    epsilon = 0.01

    # Set variables values
    L_values = np.arange(1, 21, 1)
    regularisation_variables = np.arange(0.000005, 0.0005, 0.000005)

    # generate sample
    [xlearn, ylearn],[xvalid, yvalid], [xtest, ytest] = utilities.generate_experiment(func_type, n_learn, n_valid , n_test)

    # xlearn, ylearn, xvalid, yvalid = split_train(xtrain,ytrain, round(n*0.5))


    # run experiments:
    asl_best_result = asl_experiment(xlearn, ylearn, xvalid, yvalid, xtest, ytest, L_values, epsilon, func_type)
    mlp_best_result, krr_best_result = experiment_mlp_krr(xlearn, ylearn, xvalid, yvalid, xtest, ytest,
                                                          regularisation_variables, func_type)


    # print and plot results
    compare_squared_loss()
    plot_best_result()
