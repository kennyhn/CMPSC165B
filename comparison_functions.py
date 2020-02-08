#!/usr/bin/env python3
import numpy as np


def calculate_confusion_matrix(y_pred, y_real):
    n, _ = np.shape(y_pred)
    confusion_matrix = np.zeros(shape = (3, 3,), dtype = int)
    for i in range(n):
        # Row in cm is prediction and column is actual value
        confusion_matrix[int(y_pred[i][0])][int(y_real[i][0])] += 1
    return confusion_matrix

def calculate_accuracy(y_pred, y_real):
    confusion_matrix    = calculate_confusion_matrix(y_pred, y_real)
    total_correct       = np.trace(confusion_matrix)
    total_data          = np.shape(y_pred)[0]
    return total_correct/total_data

def print_confusion_matrix(confusion_matrix):
    print('Confusion Matrix: |{:4} {:4} {:4}|\n'.format(confusion_matrix[0][0], confusion_matrix[0][1], confusion_matrix[0][2]))
    print('                : |{:4} {:4} {:4}|\n'.format(confusion_matrix[1][0], confusion_matrix[1][1], confusion_matrix[1][2]))
    print('                : |{:4} {:4} {:4}|\n'.format(confusion_matrix[2][0], confusion_matrix[2][1], confusion_matrix[2][2]))

def print_accuracy(accuracy):
    print('Accuracy:       : {:.2f}'.format(accuracy))
