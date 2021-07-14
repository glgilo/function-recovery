from averagesmoothnesslearner import *
from utilities import *
from tweet_experiment import *
# [xtrain, ytrain], [xtest, ytest] = utilities.generate_experiment(0, 32)
# print(xtrain, ytrain, xtest, ytest)
# utilities.test_linear_program()

# def run_full_experiment():


def plot_best_result():
    global x
    # create plot of best result:
    func = utilities.func_sin
    if func_type == 1:
        func = utilities.func_sign

    x = np.linspace(0, 1, 100)
    plt.scatter(xtrain, ytrain, c='b', marker='x', label='Train sample')
    plt.scatter(xtest, ytest, c='g', marker='s', label='Test sample')
    plt.plot(xtest, mlp_best_result['predictions'], c='g', label='mlp')
    plt.plot(xtest, krr_best_result['predictions'], c='m', label='krr')
    plt.plot(xtest, asl_best_result['predictions'], c='k', label='Average smoothness learner')
    plt.plot(x, func(x), c='r', label='sin(2Pi*x)')
    plt.xlabel('x - axis')
    plt.ylabel('y - axis')
    # plt.title(' ASL n={}, L={} epsilon={}'.format(len(self.xs), self.L, self.epsilon))
    plt.title('n={}, ASL: L={} epsilon={}\nKRR: alpha = {}, MLP: alpha={}'.format(n, asl_best_result['L'],
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


if __name__ == '__main__':
    # demo_test_learner(n=32, L=10, epsilon=0.1, std_error=0.1, func_type=1)
    # utilities.test_linear_program()

    # set function as:
    #   0 := sin(2*pi*x), n = 32
    #   1 := sign(sin(2*pi*x)), n = 32
    func_type = 0
    n = 32
    epsilon = 0.1

    # Set variables values
    L_values = np.arange(1, 11, 1)
    regularisation_variables = np.arange(0, 0.0001, 0.000005)

    # generate sample
    [xtrain, ytrain], [xtest, ytest] = utilities.generate_experiment(func_type, n)

    # run experiments:
    asl_best_result = asl_experiment(xtrain, ytrain, xtest, ytest, L_values, epsilon, func_type)
    mlp_best_result, krr_best_result = experiment_mlp_krr(xtrain, ytrain, xtest, ytest, regularisation_variables, func_type)

    # print and plot results
    compare_squared_loss()
    plot_best_result()
