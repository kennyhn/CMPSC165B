#!/usr/bin/env python3

import math
import numpy as np
import queue as q

class Tree:
    def __init__(self, x, y, max_depth):
        _, self.num_of_features = np.shape(x)
        self.num_of_classes     = 0
        self.max_depth          = max_depth
        self.class_name_dict    = {} # Dictionary tells which index a class has

        probability = []
        num_of_data = 0
        for classes in y:
            if classes[0] not in self.class_name_dict.keys():
                self.class_name_dict[classes[0]] = self.num_of_classes
                probability.append(0)
                self.num_of_classes += 1
            # count number of occurences in each
            num_of_data += 1
            probability[self.class_name_dict[classes[0]]] += 1    
        
        # have to divide the occurences by total number of data
        self.root_node = Node([x/num_of_data for x in probability], 0)


    ########################### Tree functions ###############################

    def findLeafNode(self, sample, node): # A row in x and always start at root when called
        index = node.feature_index
        limit = node.limit
        if limit != -1 and index != -1: # not a leaf node
            if sample[index] > limit:
                if node.right_branch is None:
                    print("Something is wrong")
                best_node = self.findLeafNode(sample, node.right_branch)
            else:
                if node.left_branch is None:
                    print("Something is wrong")
                best_node = self.findLeafNode(sample, node.left_branch)
        else:
            best_node = node
        return best_node # Returns the leaf node if at a leaf node

    def calculateProbability(self, x, y, feature_index, limit):
    # What is the probability it will become class w_j given feature limit
    # Returns two lists with the probabilities of all classes for left and right branch
        probability_right           = [0 for _ in range(self.num_of_classes)]
        probability_left            = [0 for _ in range(self.num_of_classes)]
        total_num_of_data_right     = 0
        total_num_of_data_left      = 0
        rows, _                     = np.shape(x)
        for row_index in range(rows):
            if x[row_index, feature_index] > limit:
                total_num_of_data_right += 1
                class_value = self.class_name_dict[y[row_index, 0]] 
                probability_right[class_value] += 1
            else:
                total_num_of_data_left += 1
                class_value = self.class_name_dict[y[row_index, 0]] 
                probability_left[class_value] += 1

        probability_right = [x/total_num_of_data_right if x!=0 else 0 for x in probability_right]
        probability_left = [x/total_num_of_data_left if x!= 0 else 0 for x in probability_left]

        return probability_left, total_num_of_data_left, probability_right, total_num_of_data_right

    def findMinMax(self, x, feature_index):
        rows, _ = np.shape(x)
        minimum_value = math.inf
        maximum_value = -math.inf
        for i in range(rows):
            if x[i, feature_index] > maximum_value:
                maximum_value = x[i, feature_index]
            if x[i, feature_index] < minimum_value:
                minimum_value = x[i, feature_index]
        return (minimum_value, maximum_value,)

    def updateNode(self, x, y, node):
        num_of_data, _ = np.shape(x)
        best_feature_index      = -1
        best_limit              = -1
        best_impurity           = 1000 # The real should not be higher than 1
        best_right_prob         = [0, 0, 0]
        best_left_prob          = [0, 0, 0]

        information_parent      = giniImpurity(node.probabilities)

        if information_parent == 0: # Data in this node only belongs to one class
            return (None, None,) # Do not update left and right branch
        if node.depth_level == self.max_depth:
            return (None, None,) # Stop adding more levels to tree


        for feature_index in range(self.num_of_features):
            minimum, maximum = self.findMinMax(x, feature_index)
            if (maximum-minimum) != 0:
                limit_step_length = (maximum-minimum)/5
            else:
                limit_step_length = 1
            for limit in np.arange(minimum, maximum+limit_step_length, limit_step_length):
                # Try out 10 different limits between min and max values of features
                # to see which one gives the greatest impurity
                data = self.calculateProbability(x, y, feature_index, limit)
                probability_left, num_data_left, probability_right, num_data_right = data
                p_l = num_data_left/num_of_data
                p_r = num_data_right/num_of_data
                
                impurity          = p_l*giniImpurity(probability_left)
                impurity         += p_r*giniImpurity(probability_right)
                if impurity < best_impurity:
                    # Want to minimize the impurity when splitting
                    if impurity > 1:
                        print("Something is wrong")
                    best_impurity           = impurity
                    best_feature_index      = feature_index
                    best_limit              = limit
                    best_right_prob         = probability_right
                    best_left_prob          = probability_left

        right_node = Node(best_right_prob, node.depth_level+1)
        left_node  = Node(best_left_prob, node.depth_level+1)
        node.right_branch   = right_node
        node.left_branch    = left_node
        node.feature_index  = best_feature_index
        node.limit          = best_limit
        return (left_node, right_node) # To get to a new depth

    def split_data(self, x, y, node):
        rows, columns   = np.shape(x)
        left_data_x     = np.empty(shape = (0, columns,), dtype = float) 
        left_data_y     = np.empty(shape = (0, 1,), dtype = float)
        right_data_x    = np.empty(shape = (0, columns,), dtype = float) 
        right_data_y    = np.empty(shape = (0, 1,), dtype = float)

        for row in range(rows):
            if x[row, node.feature_index] > node.limit:
                data_x = np.reshape(x[row, :], (1, columns,))
                data_y = np.reshape(y[row, :], (1,1,))
                right_data_x = np.append(right_data_x, data_x, axis = 0)
                right_data_y = np.append(right_data_y,data_y, axis = 0)
            else:
                data_x = np.reshape(x[row, :], (1, columns,))
                data_y = np.reshape(y[row, :], (1,1,))
                left_data_x = np.append(left_data_x, data_x, axis = 0)
                left_data_y = np.append(left_data_y,data_y, axis = 0)

        return (left_data_x, left_data_y, right_data_x, right_data_y,)


    def train(self, x, y):
        node_expansion_queue = q.Queue()
        node_expansion_queue.put(self.root_node)
        training_data_x      = q.Queue()
        training_data_y      = q.Queue()
        training_data_x.put(x)
        training_data_y.put(y)

        while not node_expansion_queue.empty():
            node_to_train = node_expansion_queue.get()
            x_data        = training_data_x.get()
            y_data        = training_data_y.get()
            left_node, right_node = self.updateNode(x_data, y_data, node_to_train)
            if left_node != None and right_node != None:
                node_expansion_queue.put(left_node)
                node_expansion_queue.put(right_node)
                xleft, yleft, xright, yright = self.split_data(x_data, y_data, node_to_train)
                training_data_x.put(xleft)
                training_data_y.put(yleft)
                training_data_x.put(xright)
                training_data_y.put(yright)


class Node:
    # Each node can only have maximum two children nodes
    def __init__(self, probabilities, depth_level):

        self.probabilities  = probabilities # List of probabilities for each of the classes in the tree
        # Feature index tells which features is tested to decide what the next node is
        self.feature_index  = -1 # Leaf node will have feature index -1
        self.limit          = -1         # Limit to see which node to go
        self.left_branch    = None   # left branch if feature < limit
        self.right_branch   = None   # Right branch if feature > limit
        self.depth_level    = depth_level   # What level the node is at in the tree
    


def DecisionTree(phase, x = None, y = None, depth = -1, tree = None):
    # Features x, class y, tree depth and root node of tree
    if phase.lower() == 'training' and x is not None and y is not None and depth != -1:
        return trainTree(x, y, depth) # Return a tree object
        
    elif phase.lower() == 'validation' and x is not None and tree is not None:
        return testTree(x, tree) # Returns the results of all data
    else:
        print("Inputs are not valid")

def trainTree(x, y, depth):
    # return tree that have been trained
    
    # Initiate tree object
    dTree = Tree(x, y, depth) # Need to create the tree
    dTree.train(x,y) # Train it bruh
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

# Returns accuracy given the classified data and the actual data
def calculateAccuracy(classified_y, actual_y):
    total_data                      = 0
    total_correct_classifications   = 0

    for row in range(np.shape(classified_y)[0]):
        total_data += 1
        if classified_y[row, 0] == actual_y[row, 0]:
            total_correct_classifications += 1
    # return float of 3 decimals

    return round((total_correct_classifications/total_data)*1000)/1000


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

