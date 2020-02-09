#!/usr/bin/env python3

import math
import numpy as np

def find_largest_of_classes(class0, class1, class2):
    if class0 > class1 and class0 > class2:
        # if poor has highest value
        value = 0
    elif class1 > class0 and class1 > class2:
        # if median has highest value
        value = 1
    else:
        # otherwise excellent has highest value
        value = 2
    return value

def sigma_func(function_value):
    return 1/(1+math.exp(-function_value))

def cross_entropy(x, y, param):
    
    # Have to make some logic to classify the result as 1 and 0

    n, _            = np.shape(x)
    ones_n          = np.ones(shape = (n, 1))

    # Modify x so that x*param = sum(x_i*w_i) + b
    z_temp          = np.append(x, ones_n, axis = 1)
    
    # Calculate f_wb(x^i)=sigma(y_pred)
    y_pred          = np.matmul(z_temp, param)
    f               = np.vectorize(sigma_func)(y_pred)

    # Calculate each term in the cross entropy seperately
    term            = np.matmul(np.transpose(y), np.vectorize(math.log)(f))
    inverse_term    = np.matmul(np.transpose(ones_n - y), np.vectorize(math.log)(ones_n - f))

    return -(term+inverse_term)[0][0]
    
def partition_into_class(y, class_number):
    # It is assumed that the values y take are real numbers between 0 and 1
    # if the value in y is between min and max it is in class 1 denoted by value 1
    # or else it is in class 2 denoted by value 0

    n, _        = np.shape(y)
    y_temp      = np.zeros(shape = (n,1,))
    for i in range(n):
        if y[i][0] == class_number:
            y_temp[i][0] = 1
    return y_temp

def classify_real_result(y):
    n, _        = np.shape(y)
    y_temp      = np.zeros(shape = (n,1,))
    
    for i in range(n):
        if y[i][0] <= 4:                     # Class poor
            y_temp[i][0] = 0
        elif y[i][0] >= 5 and y[i][0] <= 6:  # Class median
            y_temp[i][0] = 1
        elif y[i][0] >= 7:                      # Class excellent
            y_temp[i][0] = 2
    return y_temp

def classify_prediction(x, param):
    n, k            = np.shape(x) # n rows and k columns
    ones_n          = np.ones(shape = (n, 1,))
    temp_x          = np.append(x, ones_n, axis = 1)
    param_class0    = param[:k+1][:]
    param_class1    = param[k+1:2*k+2][:]
    param_class2    = param[2*k+2:][:]
    
    z0              = np.matmul(temp_x, param_class0)
    z1              = np.matmul(temp_x, param_class1)
    z2              = np.matmul(temp_x, param_class2)

    f0              = np.vectorize(sigma_func)(z0)
    f1              = np.vectorize(sigma_func)(z1)
    f2              = np.vectorize(sigma_func)(z2)
    
    y_pred          = np.vectorize(find_largest_of_classes)(f0, f1, f2)
    return y_pred
    
def remove_correlated_columns(x):
    corr_limit          = 0.55
    _, k                = np.shape(x)
    high_corr_columns = np.zeros((k,k,))
    # Find correlated columns
    for first_column in range(k):
        column1             = x[:, first_column]
        for second_column in range(k):
            column2     = x[:, second_column]
            correlation = np.corrcoef(column1, column2)
            if abs(correlation[0, 1]) >= corr_limit:
                high_corr_columns[first_column][second_column] = 1
    corr_counter = [-1 for _ in range(k)]
    # Find the column that is most correlated with other columns
    for rows in range(k):
        for columns in range(k):
            if high_corr_columns[rows, columns] == 1:
                corr_counter[rows] += 1
    if corr_counter[np.argmax(corr_counter)] > 0:
        column      = np.argmax(corr_counter)
        x           = np.delete(x, column, axis = 1)
        x           = remove_correlated_columns(x)
    return x