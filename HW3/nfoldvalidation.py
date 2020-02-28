#!/usr/bin/env python3

import tree
import math
import numpy as np

def splitDatainNBuckets(n, x, y):
    num_of_classes = 3
    rows, _ = np.shape(x)
    split_point = math.floor(rows/num_of_classes)

    class0_x = x[0:split_point, :]
    class0_y = y[0:split_point, :]
    class1_x = x[split_point:2*split_point, :]
    class1_y = y[split_point:2*split_point, :]
    class2_x = x[2*split_point:, :]
    class2_y = y[2*split_point:, :]


    bucket_size = math.floor(split_point/n)

    list_of_data_x = []
    list_of_data_y = []

    for i in range(n):
        if i == n-1:
            data_x = np.concatenate((class0_x[i*bucket_size:, :],
                                    class1_x[i*bucket_size:, :],
                                    class2_x[i*bucket_size:, :]), axis = 0)
            data_y = np.concatenate((class0_y[i*bucket_size:, :],
                                    class1_y[i*bucket_size:, :],
                                    class2_y[i*bucket_size:, :]), axis = 0)

            # Add the n data-sets in a list
            list_of_data_x.append(data_x)
            list_of_data_y.append(data_y)
        else:
            data_x = np.concatenate((class0_x[i*bucket_size:(i+1)*bucket_size, :],
                                    class1_x[i*bucket_size:(i+1)*bucket_size, :],
                                    class2_x[i*bucket_size:(i+1)*bucket_size, :]), axis = 0)
            data_y = np.concatenate((class0_y[i*bucket_size:(i+1)*bucket_size, :],
                                    class1_y[i*bucket_size:(i+1)*bucket_size, :],
                                    class2_y[i*bucket_size:(i+1)*bucket_size, :]), axis = 0)
            

            # Add the n data-sets in a list
            list_of_data_x.append(data_x)
            list_of_data_y.append(data_y)

    return (list_of_data_x, list_of_data_y,)

def nFoldValidationIrisData(n, x, y):
    # We know we have balanced and sorted data
    # so we split them up so we can equal distribute them into n-buckets
    

    list_of_data_x, list_of_data_y = splitDatainNBuckets(n, x, y)
    rows, _ = np.shape(list_of_data_x[0])

    depth = 30
    iterations = 100


    indexes_shuffle         = np.arange(rows)
    num_of_training_data    = math.ceil(rows*4/5)
    avg_acc_test            = [0]*depth
    avg_acc_train           = [0]*depth
    for _ in range(iterations):
        # shuffle the data 100 times for each of the buckets
        # Shuffle the rows the same for each of the datasets
        np.random.shuffle(indexes_shuffle)
        for j in range(len(list_of_data_x)):
            list_of_data_x[j] = list_of_data_x[j][indexes_shuffle, :]
            list_of_data_y[j] = list_of_data_y[j][indexes_shuffle, :]


        for d in range(depth):
            # for each depth calculate the accuracy 
            for j in range(len(list_of_data_x)):
                # For each of the buckets with data
                # take out random data for training and testing
                training_data_x     = list_of_data_x[j][0:num_of_training_data, :]
                training_data_y     = list_of_data_y[j][0:num_of_training_data, :]

                testing_data_x      = list_of_data_x[j][num_of_training_data:, :]
                testing_data_y      = list_of_data_y[j][num_of_training_data:, :]

                #Train
                dTree               = tree.DecisionTree(phase = 'Training', x = training_data_x, y = training_data_y, depth = d)

                #Test
                classified_y        = tree.DecisionTree(phase = 'Validation', x = testing_data_x, tree = dTree)
                accuracy            = tree.calculateAccuracy(classified_y, testing_data_y)
                avg_acc_test[d]     += accuracy

                classified_y_train  = tree.DecisionTree(phase = 'Validation', x = training_data_x, tree = dTree)
                accuracy_train      = tree.calculateAccuracy(classified_y_train, training_data_y)
                avg_acc_train[d]    += accuracy_train
    


    avg_acc_test    = [x/(iterations*len(list_of_data_x)) for x in avg_acc_test]
    avg_acc_train   = [x/(iterations*len(list_of_data_x)) for x in avg_acc_train]

    return (avg_acc_test, avg_acc_train,)


