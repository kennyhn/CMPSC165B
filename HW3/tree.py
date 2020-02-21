#!/usr/bin/env python3

import math
import numpy as np


class Tree:
    def __init__(self, x, y):
        # Assumed we have balanced data for each class
        _, self.num_of_features = np.shape(x)
        self.num_of_classes     = 0
        self.depth              = 0 # With only root node the depth is 0

        class_names             = []
        for classes in y:
            if classes not in class_names:
                class_names.append(classes)
                self.num_of_classes += 1
        
        # Probability is equal for all classes at root
        self.root_node = Node([1/self.num_of_classes for _ in range(self.num_of_classes)])


    # tree functions    
    def traverseTree(self, sample): # A row in x
        pass

class Node:
    # Each node can only have maximum two children nodes
    def __init__(self, probabilities):

        self.probabilities  = probabilities # Probability for each of the classes in the tree
        self.feature_index  = -1            # Leaf node will have feature index -1
        self.limit          = -1            # Limit to see which node 
        self.left_branch    = None
        self.right_branch   = None

    

def DecisionTree(x = None, y = None, depth = 0, tree = None):
    # Features x, class y, tree depth and root node of tree
    if x != None and y != None and depth != 0:
        return trainTree(x, y, depth) # Return a tree object
        
    elif x != None and tree != None:
        return testTree(x, tree)
    else:
        print("Inputs are not valid")

def trainTree(x, y, depth):
    # return tree that have been trained
    
    # Initiate tree object
    dTree = Tree(x, y) # Need to create the tree

    # TBD
    return dTree

def testTree(x, tree):
    # return classification prediction
    # TBD
    return 0

def calculateProbability(x, y, num_of_classes, feature_index, limit):
    # What is the probability it will become class w_j given feature limit
    # Return a list with the probabilities of all in this case three classes
    pass




# three different impurity functions
def giniImpurity(probabilities):
    # takes in a list of the probabilities for the classes 0, 1, 2
    # then calculate the gini entropy
    ent = 1
    for probability in probabilities:
        ent -= probability**2
    return ent

def entropyImpurity(probabilities):
    ent = 0
    for probability in probabilities:
        ent += probability*math.log2(probability)
    return -ent

def misclassificationEntropy(probabilities):
    max_prob =0
    for probability in probabilities:
        if probability > max_prob:
            max_prob = probability
    return 1-max_prob

