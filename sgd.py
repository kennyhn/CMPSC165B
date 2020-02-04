#!/usr/bin/env python3

import numpy as np
import math


def SGDSolver(x=None, y=None, alpha=[10 ** (-6), 5 * 10 ** (-6)], lam=[0.1, 0.2], nepoch=None, epsilon=10 ** (-6),
              param=None):
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

    n = np.shape(y)[0]
    k = x.shape[1]  # k is number of columns in x
    ones = np.ones(n)
    stepalph = (alpha[1] - alpha[0]) / 100
    steplam = (lam[1] - lam[0]) / 10
    # Assumed x is passed through as an numpy matrix
    firstiteration = True
    lowestError = 0
    bestParameters = np.zeros(k + 1)  # need to make parameter array for the best w and b
    w_start = np.random.standard_normal(k)
    b_scalar_start = np.random.standard_normal(1)
    b_start = np.multiply(b_scalar_start, ones) # vector of b
    y  = y.flatten()

    for regterm in np.arange(lam[0], lam[1]+steplam, steplam):
        for lr in np.arange(alpha[1], alpha[0]-stepalph, -stepalph):
            w = np.array(w_start, copy=True)
            b_scalar = b_scalar_start
            b_vec = np.array(b_start, copy=True)
            # create a random weighting matrix with length k and random b (as initial values)
            #w = np.array([0.02520948,  0.35102952, -0.19441002, -0.13571329,  0.10601841,
            #            -0.19589803,  0.18104781])
            #b = np.array([-0.1274908])
            for _ in range(nepoch):
                # gradient = -2/n*x^T*(y-xw-b*1)+2*lambda*w
                ypred = np.add(np.matmul(x, w), b_vec)
                e = np.subtract(y, ypred)
                wgradient = np.add(-(2/n)*np.matmul(np.transpose(x), e), 2*regterm*w)
                # gradient for b := -2/n*1^T*(y-xw-b*1)
                bgradient = -(2/n)*np.matmul(np.transpose(ones), e)
                w        -= lr * wgradient
                b_scalar -= lr * bgradient
                b_vec     = np.multiply(b_scalar, ones)

                mserror = 1 / n * np.dot(y - ypred, y - ypred) + regterm * np.dot(w, w)
                #print("regterm: {} mseerror: {}".format(regterm,mserror))
                if mserror < epsilon:
                    # Just exit if error is smaller than epsilon
                    param = np.append(w, b_scalar)
                    print("error is smaller than epsilon: {}".format(mserror))
                    return np.reshape(param, (np.size(param), 1))
            param = np.append(w, b_scalar)
            # Error is scalar

            if firstiteration:
                lowestError = mserror
                bestAlpha = lr
                bestLam = regterm
                bestParameters = np.array(param, copy=True)
                firstiteration = False
            if mserror < lowestError:
                lowestError = mserror
                bestParameters = np.array(param, copy=True)
                bestAlpha = lr
                bestLam = regterm
    return np.reshape(bestParameters, (bestParameters.size, 1))


def validation_phase(x, y, param):
    # sum of error
    n     = np.size(y)
    param = param.flatten()
    if x.shape[0] == np.size(x):
        k = 1
    else:
        k = x.shape[1]
    ones = np.ones(n)
    w = param[:k]  # retrieve elements from 0 to k-1
    b = param[-1]  # last element is b
    ypred          = np.matmul(x, w) + np.multiply(b, ones)
    print(ypred)
    e = np.subtract(y, ypred)

    # calculate the error
    mse = (1 / n)*np.dot(e, e)
    print("error: {}".format(mse))
    return mse


def testing_phase(x, param):
    n = x.shape[0]  # n rows
    k = x.shape[1]  # k columns
    param = np.reshape(param, (np.shape(param)[0]))
    w = param[:k]
    b = param[-1]
    b_ones = np.ones(n)
    return np.add(np.matmul(x, w), np.multiply(b, b_ones))