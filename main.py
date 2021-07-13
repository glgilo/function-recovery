from averagesmoothnesslearner import *

# [xtrain, ytrain], [xtest, ytest] = utilities.generate_experiment(0, 32)
# print(xtrain, ytrain, xtest, ytest)
# utilities.test_linear_program()

if __name__ == '__main__':
    test_learner(n=32, L=10, epsilon=0.1, std_error=0.1, func_type=0)
    # utilities.test_linear_program()
