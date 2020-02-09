#!/usr/bin/env python3

import numpy                as np
from sgd                    import massage_data
from sklearn                import svm
import comparison_functions as cf
import time


def training_with_svm(x, y_real):
    n, _                = np.shape(x)
    x                   = massage_data(x)
    classifier_class0   = svm.SVC(C = 5, kernel = 'rbf')
    classifier_class1   = svm.SVC(C = 5, kernel = 'rbf')
    classifier_class2   = svm.SVC(C = 5  , kernel = 'rbf')

    y_class0            = np.zeros(shape = (n, 1,), dtype = int)
    y_class1            = np.zeros(shape = (n, 1,), dtype = int)
    y_class2            = np.zeros(shape = (n, 1,), dtype = int)

    for i in range(n):
        if y_real[i][0]     == 0: # class 0
            y_class0[i][0] = 1
        elif y_real[i][0]   == 1: # class 1
            y_class1[i][0] = 1
        elif y_real[i][0]   == 2: # class 2
            y_class2[i][0] = 1
    
    # Find the curve splitting class from the other two classes
    classifier_class0.fit(x, y_class0.ravel())
    classifier_class1.fit(x, y_class1.ravel())
    classifier_class2.fit(x, y_class2.ravel())
    return (classifier_class0, classifier_class1, classifier_class2)

def validate_with_svm(x, y, classifier0, classifier1, classifier2):
    n, _                     = np.shape(x)
    y_pred0                 = classifier0.predict(x)
    y_pred1                 = classifier1.predict(x)
    y_pred2                 = classifier2.predict(x)

    y_pred_result       = np.zeros(shape = (n, 1,))
    
    class_count = [0, 0, 0]
    for i in range(n):
        class_count[int(y[i][0])] += 1

    fewest_class        = np.argmin(class_count)

    for i in range(n):
        
        if y_pred0[i] > y_pred1[i] and y_pred0[i] > y_pred2[i]:
            y_pred_result[i][0] = 0
        elif y_pred1[i] > y_pred0[i] and y_pred1[i] > y_pred2[i]:
            y_pred_result[i][0] = 1
        elif y_pred2[i] > y_pred0[i] and y_pred2[i] > y_pred1[i]:
            y_pred_result[i][0] = 2
        else:
            # If conflicting values between the predictions
            y_pred_result[i][0] = fewest_class
    
    mse     = (1/n)*np.matmul(np.transpose(y_pred_result-y), y_pred_result-y)
    return mse[0][0]

def test_with_svm(x, y, classifier0, classifier1, classifier2, run_number, run_type, training_time, test_start):
    n, _                    = np.shape(x)
    y_pred0                 = classifier0.predict(x)
    y_pred1                 = classifier1.predict(x)
    y_pred2                 = classifier2.predict(x)

    y_pred_result       = np.zeros(shape = (n, 1,))
    
    class_count = [0, 0, 0]
    for i in range(n):
        class_count[int(y[i][0])] += 1
    fewest_class        = np.argmin(class_count)

    for i in range(n):
        if y_pred0[i] > y_pred1[i] and y_pred0[i] > y_pred2[i]:
            y_pred_result[i][0] = 0
        elif y_pred1[i] > y_pred0[i] and y_pred1[i] > y_pred2[i]:
            y_pred_result[i][0] = 1
        elif y_pred2[i] > y_pred0[i] and y_pred2[i] > y_pred1[i]:
            y_pred_result[i][0] = 2
        else:
            # If conflicting values between the predictions
            y_pred_result[i][0] = fewest_class
    confusion_matrix = cf.calculate_confusion_matrix(y_pred_result, y)
    accuracy         = cf.calculate_accuracy(y_pred_result, y)

    cf.save_to_file("data_test_long.txt", run_number, f"{run_type} SVM", confusion_matrix, accuracy, training_time, time.time()-test_start)
