#!/usr/bin/env python3
import numpy        as np
import math
from tree_class     import Tree
from tree           import testTree

def calculateEpsilon(training_y_real, training_y_predicted, weight):
    # Here y is either the class we want to classify (1) or the other classes (0)
    z1      = np.sum(weight)
    # If training_y_real = training_y_predicted we get +-2 if it is not equal we get 0
    ones_n  = np.ones(shape = (np.shape(training_y_real)[0], 1,), dtype = int)
    delta   = ones_n - np.abs(training_y_real+training_y_predicted)/2
    epsilon = np.matmul(np.transpose(delta), weight)

    # Return the error rate
    return (epsilon/z1) 

def calculateAlpha(epsilon):
    if epsilon == 0:
        # Correctly classified all training data so the error rate is 0
        return 0
    return math.log(math.sqrt((1-epsilon)/epsilon))

def reweight(training_y_real, training_y_predicted, weight):
    alpha       = calculateAlpha(calculateEpsilon(training_y_real, training_y_predicted, weight))
    rows, _     = np.shape(training_y_real)

    for row in range(rows):
        if training_y_real[row, 0] == training_y_predicted[row, 0]:
            weight[row, 0] = weight[row, 0]*math.exp(alpha)
        else:
            weight[row, 0] = weight[row, 0]*math.exp(-alpha)
    # New weights for next tree
    return weight

def convertOneVsAllToBinary(y, class_name):
    rows, _ = np.shape(y)
    # +1 if y is class_name -1 if y is not clas_name
    y_temp  = np.empty(shape = (rows, 1), dtype = int)
    for row in range(rows):
        if y[row, 0] == class_name:
            y_temp[row, 0] = 1
        else:
            y_temp[row, 0] = -1
    return y_temp



def trainAdaBoost(x, y, num_of_trees):
    rows, _         = np.shape(x)
    class_names     = np.unique(y)
    class_names     = np.flip(class_names)
    #Weights for each class in each 'parallell' tree
    list_of_ensembles   = [] # Three ensembles for each class_name poor, median, excellent

    for i in range(len(class_names)):
        y_binary        = convertOneVsAllToBinary(y, class_names[i])
        ensemble        = []
        weights         = np.ones(shape = (rows, 1,), dtype = float)

        for _ in range(num_of_trees):
            stump           = Tree(x, y_binary, 1, weights)
            stump.train(x, y_binary, 'boosting', weights)
            y_pred          = testTree(x, stump)
            y_pred          = y_pred.astype(int)
            epsilon         = calculateEpsilon(y_binary, y_pred, weights)
            alpha           = calculateAlpha(epsilon)
            ensemble.append((alpha, stump,))

            #reweights
            weights     = reweight(y_binary, y_pred, weights)
        # should make three ensembles
        list_of_ensembles.append(ensemble)
    
    return list_of_ensembles





        




    