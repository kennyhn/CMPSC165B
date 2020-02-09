#!/usr/bin/env python3

import numpy                    as np
from sgd                        import SGDSolver
import binary_classification    as bc
import csv
import comparison_functions     as cf
import time
import svm_functions            as sf
import math

import matplotlib.pyplot          as plt

def load_data(filename):
    """
    arg: filename - filename of file you want to load data from
            e.g. red_train.npy 
    Return: x values (numpy array: n x k)
            y values (numpy array: n x 1)
    """
    data_x  = []
    data_y  = []
    # TODO: Finish this function here.
    if ".csv" in filename:
        data    = np.genfromtxt(filename, delimiter = ",", skip_header = 1)
        data_x  = data[:, :-1]
        data_y  = data[:, -1:]
    elif ".npy" in filename:
        data    = np.load(filename)
        data_x  = data[:, :-1]
        data_y  = data[:, -1:]
    return data_x, data_y

def red_wine_run(train_red_x, train_red_y, test_red_x, test_red_y):
    # Red wine data
    print('---------------\nRed Wine Data\n---------------')

    # Training Phase
    # values for 2D-grid search
    lam     = [1e-3, 2]           # regularization weight [min, max]
    alpha   = [1e-6, 1e-4]        # learning rate [min, max]
    nepochs = 50        # sample # of epochs
    epsilon = 0.0       # epsilon value
    param   = np.random.standard_normal(size = np.shape(train_red_x)[1]+1)
    param   = np.reshape(param, newshape = (np.shape(train_red_x)[1]+1, 1,))
    # end TODO

    # using this alpha and lambda values run the training
    print(f"alpha: {alpha}, lambda:{lam}")
    print("Running Training phase")
    # return param and optimal values for alpha and lambda from SGDSolver
    training_start      = time.time()
    param, alpha, lam = SGDSolver('Training', train_red_x, train_red_y, alpha, lam, nepochs, epsilon, param)
    training_time       = time.time()-training_start
    # optimal values from 2-D search
    print(f"optimal values\nalpha: {alpha}, lambda: {lam}")

    # Note: validation and testing phases only take a single value for (alpha, lam) and not a list. 
    # Validation Phase
    x_mse_val = SGDSolver('Validation', test_red_x, test_red_y, alpha, lam, nepochs, epsilon, param)
    print(f"Current Red Wine Data MSE is: {x_mse_val}.")

    # Testing Phase
    test_start      = time.time()
    red_wine_predicted = SGDSolver('Testing', test_red_x, test_red_y, alpha, lam, nepochs, epsilon, param)
    test_time       = time.time() - test_start
    #for i in range(100, 150):
        #print(f"Predicted: {red_wine_predicted[i]}, Real: {test_red_y[i]}")
    print(f'Training took: {training_time:.2f}\nTesting took: {test_time:.3f}s')
    test_y              = bc.classify_real_result(test_red_y)
    confusion_matrix    = cf.calculate_confusion_matrix(red_wine_predicted, test_y)
    accuracy            = cf.calculate_accuracy(red_wine_predicted, test_y)
    cf.print_confusion_matrix(confusion_matrix)
    cf.print_accuracy(accuracy)
    #cf.save_to_file("data_test_long.txt", run_number, "Red wine LR", confusion_matrix, accuracy, training_time, test_time)


def white_wine_run(train_white_x, train_white_y, test_white_x, test_white_y):
    # White wine data
    print('---------------\nWhite Wine Data\n---------------')

    # TODO: Change hyperparameter values here as needed 
    # similar to red_wine_run
    # values for 2D-grid search
    lam             = [1e-3, 2]           # regularization weight [min, max]
    alpha           = [1e-6, 1e-5]        # learning rate [min, max]
    nepochs         = 500                  # sample # of epochs
    epsilon         = 0.05                 # epsilon value
    param           = np.random.standard_normal(size = np.shape(train_white_x)[1]+1)
    param           = np.reshape(param, newshape = (np.shape(train_white_x)[1]+1, 1,))
    # end TODO

    # Training Phase
    print(f"alpha: {alpha}, lambda:{lam}")
    print("Running Training phase")
    # return param and optimal values for alpha and lambda from SGDSolver
    training_start      = time.time()
    param, alpha, lam = SGDSolver('Training', train_white_x, train_white_y, alpha, lam, nepochs, epsilon, param)
    training_time       = time.time() - training_start
    # optimal values from 2-D search
    print(f"optimal values\nalpha: {alpha}, lambda: {lam}")

    # Note: validation and testing phases only take a single value for (alpha, lam) and not a list. 
    # Validation Phase
    
    x_mse_val = SGDSolver('Validation', test_white_x, test_white_y, alpha, lam, nepochs, epsilon, param)
    print(f"Current White Wine Data MSE is: {x_mse_val}.")

    # Testing Phase
    test_start      = time.time()
    white_wine_predicted = SGDSolver('Testing', test_white_x, test_white_y, alpha, lam, nepochs, epsilon, param)
    test_time       = time.time() - test_start
    #for i in range(100, 150):
    #    print(f"Predicted: {white_wine_predicted[i]}, Real: {test_white_y[i]}")
    print(f'Training took: {training_time:.2f}\nTesting took: {test_time:.3f}s')
    test_white_y              = bc.classify_real_result(test_white_y)
    confusion_matrix    = cf.calculate_confusion_matrix(white_wine_predicted, test_white_y)
    accuracy            = cf.calculate_accuracy(white_wine_predicted, test_white_y)
    cf.print_confusion_matrix(confusion_matrix)
    cf.print_accuracy(accuracy)
    #cf.save_to_file("data_test_long.txt", run_number, "White wine LR", confusion_matrix, accuracy, training_time, test_time)


def main():
    # import all the data
    # TODO: call the load_data() function here and load data from file
    """
    x, y                            = load_data("classifier_test.csv")
    white_wine_run(x, y, x, y, 0)
    """

    
    train_red_x, train_red_y        = load_data('hw2_winequality-red_train.npy')
    test_red_x, test_red_y          = load_data('hw2_winequality-red_test.npy')
    train_white_x, train_white_y    = load_data('hw2_winequality-white_train.npy')
    test_white_x, test_white_y      = load_data('hw2_winequality-white_test.npy')
    
    """
    n_train_red, _                  = np.shape(train_red_x)
    n_test_red, _                   = np.shape(test_red_x)
    n_train_white, _                = np.shape(train_white_x)
    n_test_white, _                 = np.shape(test_white_x)
    
    

    
    partition_factor                = 5
    
    for i in range(partition_factor):
        # Red wine
        partitioned_train_red_x     = train_red_x[math.floor(n_train_red*(i/partition_factor)):math.floor(n_train_red*(i+1)/partition_factor), :]
        partitioned_train_red_y     = train_red_y[math.floor(n_train_red*(i/partition_factor)):math.floor(n_train_red*(i+1)/partition_factor), :]
        partitioned_test_red_x      = test_red_x[math.floor(n_test_red*(i/partition_factor)):math.floor(n_test_red*(i+1)/partition_factor), :]
        partitioned_test_red_y      = test_red_y[math.floor(n_test_red*(i/partition_factor)):math.floor(n_test_red*(i+1)/partition_factor), :]

        red_wine_run(partitioned_train_red_x, partitioned_train_red_y, partitioned_test_red_x, partitioned_test_red_y, i+1)
        
        partitioned_train_red_y     = bc.classify_real_result(partitioned_train_red_y)
        partitioned_test_red_y      = bc.classify_real_result(partitioned_test_red_y)
        training_start              = time.time()
        clf0, clf1, clf2 = sf.training_with_svm(partitioned_train_red_x, partitioned_train_red_y)
        training_time               = time.time() - training_start
        sf.validate_with_svm(partitioned_test_red_x, partitioned_test_red_y, clf0, clf1, clf2)
        test_start               = time.time()
        sf.test_with_svm(partitioned_test_red_x, partitioned_test_red_y, clf0, clf1, clf2, i+1, "Red wine", training_time, test_start)
        # White wine
        partitioned_train_white_x     = train_white_x[math.floor(n_train_white*(i/partition_factor)):math.floor(n_train_white*(i+1)/partition_factor),:]
        partitioned_train_white_y     = train_white_y[math.floor(n_train_white*(i/partition_factor)):math.floor(n_train_white*(i+1)/partition_factor),:]
        partitioned_test_white_x      = test_white_x[math.floor(n_test_white*(i/partition_factor)):math.floor(n_test_white*(i+1)/partition_factor),:]
        partitioned_test_white_y      = test_white_y[math.floor(n_test_white*(i/partition_factor)):math.floor(n_test_white*(i+1)/partition_factor),:]

        white_wine_run(partitioned_train_white_x, partitioned_train_white_y, partitioned_test_white_x, partitioned_test_white_y, i+1)

        partitioned_train_white_y     = bc.classify_real_result(partitioned_train_white_y)
        partitioned_test_white_y      = bc.classify_real_result(partitioned_test_white_y)
        training_start                = time.time()
        clf0, clf1, clf2 = sf.training_with_svm(partitioned_train_white_x, partitioned_train_white_y)
        training_time                 = time.time()-training_start
        sf.validate_with_svm(partitioned_test_white_x, partitioned_test_white_y, clf0, clf1, clf2)
        test_start                    = time.time()
        sf.test_with_svm(partitioned_test_white_x, partitioned_test_white_y, clf0, clf1, clf2, i+1, "White wine", training_time, test_start)
    
    cf.add_lines_to_file("data_test_long.txt", 5)
    """
    # Tests
    time_red    = time.time()
    red_wine_run(train_red_x, train_red_y, test_red_x, test_red_y)
    print("Time it took for code to run on red wine: {}".format(time.time()-time_red))

    time_white  = time.time()
    white_wine_run(train_white_x, train_white_y, test_white_x, test_white_y)
    print("Time it took for code to run on white wine: {}".format(time.time()-time_white))
    
    
    """
    start_time      = time.time()
    train_red_y     = bc.classify_real_result(train_red_y)
    test_red_y      = bc.classify_real_result(test_red_y)
    training_start  = time.time()
    clf0, clf1, clf2 = sf.training_with_svm(train_red_x, train_red_y)
    training_time   = time.time() - training_start
    sf.validate_with_svm(test_red_x, test_red_y, clf0, clf1, clf2)
    test_start      = time.time()
    sf.test_with_svm(test_red_x, test_red_y, clf0, clf1, clf2, 1, 'Red wine', training_time, test_start)

    start_time      = time.time()
    train_white_y   = bc.classify_real_result(train_white_y)
    test_white_y    = bc.classify_real_result(test_white_y)
    training_start  = time.time()
    clf0, clf1, clf2 = sf.training_with_svm(train_white_x, train_white_y)
    training_time   = time.time()-training_start
    sf.validate_with_svm(test_white_x, test_white_y, clf0, clf1, clf2)
    test_start      = time.time()
    sf.test_with_svm(test_white_x, test_white_y, clf0, clf1, clf2, 1, 'White wine', training_time, test_start)
    """
 
if __name__ == "__main__":
    main()

    