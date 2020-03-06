#!/usr/bin/env python3
import numpy as np
import time


def calculate_confusion_matrix(y_pred, y_real, class_dictionary):
    n, _ = np.shape(y_pred)
    confusion_matrix = np.zeros(shape = (3, 3,), dtype = int)
    for i in range(n):
        # Row in cm is prediction and column is actual value
        confusion_matrix[class_dictionary[y_pred[i][0]]][class_dictionary[y_real[i][0]]] += 1
    return confusion_matrix

# Returns accuracy given the classified data and the actual data
def calculateAccuracy(classified_y, actual_y):
    total_data                      = 0
    total_correct_classifications   = 0

    for row in range(np.shape(classified_y)[0]):
        total_data += 1
        if classified_y[row, 0] == actual_y[row, 0]:
            total_correct_classifications += 1
            
    return total_correct_classifications/total_data


def print_confusion_matrix(confusion_matrix):
    print('Confusion Matrix: |{:4} {:4} {:4}|\n'.format(confusion_matrix[0][0], confusion_matrix[0][1], confusion_matrix[0][2]))
    print('                : |{:4} {:4} {:4}|\n'.format(confusion_matrix[1][0], confusion_matrix[1][1], confusion_matrix[1][2]))
    print('                : |{:4} {:4} {:4}|\n'.format(confusion_matrix[2][0], confusion_matrix[2][1], confusion_matrix[2][2]))

def print_accuracy(accuracy):
    print('Accuracy:       : {:.4f}'.format(accuracy))

def save_to_file(filename, test_run, type_run, confusion_matrix, accuracy, training_time, test_time):
    with open(filename, "a+") as f:
        f.write(f"Running {type_run} run number {test_run} we get:\n\n")
        f.write(f'Training of the parameters used {training_time}s\n')
        f.write(f'Testing used {test_time}s\n\n')
        f.write('                  Confusion matrix                   \n')
        f.write('Actual class ->     {:4} {:4} {:4} \n'.format(0, 1, 2))
        f.write('Predicted class 0: |{:4} {:4} {:4}|\n'.format(confusion_matrix[0][0], confusion_matrix[0][1], confusion_matrix[0][2]))
        f.write('                1: |{:4} {:4} {:4}|\n'.format(confusion_matrix[1][0], confusion_matrix[1][1], confusion_matrix[1][2]))
        f.write('                2: |{:4} {:4} {:4}|\n\n'.format(confusion_matrix[2][0], confusion_matrix[2][1], confusion_matrix[2][2]))
        f.write(f'with accuracy   : {accuracy:.4f}\n')
        f.write('--------------------------------------------------------------------------------\n')

def add_lines_to_file(filename, num):
    with open(filename, 'a+') as f:
        for _ in range(num):
            f.write("\n")

def print_file(filename):
    with open(filename, 'r') as f:
        for line in f:
            print(line)