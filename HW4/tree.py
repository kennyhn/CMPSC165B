#!/usr/bin/env python3
import math
import numpy as np
from tree_class import Tree
    
def DecisionTree(phase, x = None, y = None, depth = -1, ensemble = None):
    # TBD: Change so we can do bagging and boosting training/testing

    # Features x, class y, tree depth and root node of tree
    if phase.lower() == 'training' and x is not None and y is not None and depth != -1:
        return trainTree(x, y, depth)
        
    elif phase.lower() == 'validation' and x is not None and ensemble is not None:
        return testTree(x, ensemble) # Returns the results of all data
    else:
        print("Inputs are not valid")

def trainTree(x, y, depth, weights = None):
    # return tree that have been trained
    
    # Initiate tree object
    dTree = Tree(x, y, depth) # Need to create the tree
    dTree.train(x,y, 'boosting', weights) # Train it bruh
    # TBD
    return dTree

def testTree(x, tree):
    rows, _ = np.shape(x)
    result  = np.empty(shape = (0,1,), dtype = str)
    for row in range(rows):
        node = tree.findLeafNode(x[row, :], tree.root_node)
        class_number = node.probabilities.index(max(node.probabilities))
        key_list = list(tree.class_name_dict.keys())
        val_list = list(tree.class_name_dict.values())
        name     = key_list[val_list.index(class_number)]
        result   = np.append(result, np.reshape(np.array(name), (1,1,)), axis = 0)
    return result

def trainBaggingEnsemble(x, y, depth, num_of_trees):
    percentage              = 3/4 
    ensemble                = []
    rows, _                 = np.shape(x)
    row_indexes             = np.arange(rows)
    num_of_training_data    = math.ceil(rows*percentage)
    for _ in range(num_of_trees):
        # I want to only use 75% of the data when training
        # Randomize which data to pull 75% out of x for each tree
        np.random.shuffle(row_indexes)
        x     = x[row_indexes, :]
        y     = y[row_indexes, :]

        training_x = x[0:num_of_training_data, :]
        training_y = y[0:num_of_training_data, :]
        dTree = Tree(training_x, training_y, depth)
        dTree.train(training_x, training_y, 'bagging')
        ensemble.append(dTree)
    
    return ensemble
    

def testBaggingEnsemble(x, ensemble):
    # ensemble is a list of (5) trees
    rows, _ = np.shape(x)
    results = np.empty(shape = (rows, len(ensemble),), dtype = '<U9')
    for i in range(len(ensemble)):
        result = testTree(x, ensemble[i])
        for row in range(rows):
            results[row, i] = result[row, 0]
    prediction = np.empty(shape = (rows, 1,), dtype = '<U9')
    for row in range(rows):
        values, counts      = np.unique(results[row, :], return_counts = True)
        index               = np.argmax(counts)
        prediction[row, 0]  = values[index]
    return prediction
