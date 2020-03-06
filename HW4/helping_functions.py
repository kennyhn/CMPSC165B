#!/usr/bin/env python3
import math

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