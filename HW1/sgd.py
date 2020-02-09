#!/usr/bin/env python3
import numpy as np


def SGDSolver(x=None, y=None, alpha=[1e-4, 1e-2], lam=[1e-2, 1], nepoch=None, epsilon=10 ** (-6),
              param=None):
    # normalize data
    if x is not None:
        x = massage_data(x)
    if y is not None:
        y = massage_data(y)


    # Run SGD solver phases
    if (x is not None) and (y is not None) and (nepoch is not None):
        # Returns parameter vector with weights and b
        return training_phase(x, y, alpha, lam, nepoch, epsilon)
    elif (x is not None) and (y is not None) and (param is not None):
        # returns error
        return validation_phase(x, y, param)
    elif (x is not None) and param is not None:
        # Returns vector with predicted y
        return testing_phase(x, param)
    else:
        print("Something's wrong")


def training_phase(x, y, alpha, lam, nepoch, epsilon):

    # How many steps should be taken when searching for the best lr and rt
    stepalph        = (alpha[1] - alpha[0]) / 100
    steplam         = (lam[1] - lam[0]) / 10

    n               = np.shape(y)[0] # y have n rows
    k               = np.shape(x)[1] # x have k columns and n rows
    

    lowest_error    = np.inf
    best_parameters = np.zeros(shape = (k+1, 1,)) # length of w + one value for the bias

    w_start         = np.random.standard_normal(size = k) # w has dimension kx1
    w_start         = np.reshape(w_start, newshape = (k, 1,))
    bscalar_start   = np.random.standard_normal(size = 1)
    ones_n          = np.ones(shape = (n, 1,))

    for regterm in np.arange(lam[0], lam[1]+steplam, steplam):
        for lr in np.arange(alpha[0], alpha[1]+stepalph, stepalph):
            # 2D grid search for best learning rate and regularization factor
            
            w           = np.copy(w_start)
            b_vector    = np.copy(bscalar_start*ones_n)
            b_scalar    = bscalar_start

            # Start gradient descent method
            for _ in range(nepoch):
                # Calculate the gradient with the current weighting matrix w
                grad_w      = -(2/n)*np.matmul(np.transpose(x), y)
                grad_w      += (2/n)*np.matmul(np.matmul(np.transpose(x), x), w)
                grad_w      += (2/n)*np.matmul(np.transpose(x), b_vector)
                grad_w      += 2*regterm*w

                # Calculate the gradient with the current scalar bias b    
                grad_b      = -(2/n)*np.matmul(np.transpose(ones_n), y)
                grad_b      += (2/n)*np.matmul(np.transpose(ones_n), np.matmul(x,w))
                grad_b      += 2*b_scalar


                # Update w and b_scalar for next iteration
                w           += -lr*grad_w
                b_scalar    += -lr*grad_b[0][0]

                # calculate new b_vector
                b_vector    = b_scalar*ones_n

                # Calculate MSE with new w and b
                error       = y-np.matmul(x, w)-b_vector
                mse         = (1/n)*np.matmul(np.transpose(error), error)
                mse         += regterm*np.matmul(np.transpose(w), w)

                if mse[0][0] < epsilon:
                    return np.append(np.array(w, copy = True), b_scalar)

                # Save the best parameter values w and b
                if mse[0][0] < lowest_error:
                    lowest_error    = mse[0][0]
                    best_parameters = np.append(np.array(w, copy = True), b_scalar)
                    best_parameters = np.reshape(best_parameters, newshape = (k+1, 1,))
    return best_parameters


def validation_phase(x, y, param):
    # sum of error
    n,k         = np.shape(x)
    ones_n      = np.ones(shape = (n, 1,))
    w           = param[:k]  # retrieve elements from 0 to k-1
    w           = np.reshape(w, newshape = (k,1,))
    b           = param[-1]  # last element is b

    # Calculate the mean square error
    error       = y - np.matmul(x,w)-b*ones_n

    mse         = (1/n)*np.matmul(np.transpose(error), error)
    return mse[0][0]


def testing_phase(x, param):
    n,k         = np.shape(x)
    w           = param[:k]
    w           = np.reshape(w, newshape = (k,1,))
    b           = param[-1]
    ones_n      = np.ones(shape = (n,1))
    return np.matmul(x,w) + b*ones_n


def massage_data(x): # kxn array:
    for n_column in range(np.shape(x)[1]):

        # Reset max_value and min_value
        max_value = -np.inf
        min_value = np.inf

        # Find min and max
        for m_row in range(np.shape(x)[0]):
            if x[m_row][n_column] < min_value:
                min_value      = x[m_row][n_column]
            if x[m_row][n_column] > max_value:
                max_value      = x[m_row][n_column]
        
        # Normalize data in each column
        for m_row in range(np.shape(x)[0]):
            x[m_row][n_column] = (x[m_row][n_column] - min_value)/(max_value-min_value)
            
    return x