#!/usr/bin/env python3

import numpy as np
import math

def load_data(filename):
    """
    arg: filename - filename of file you want to load data from
            e.g. red_train.npy 
    Return: x values (numpy array: n x k)
            y values (numpy array: n x 1)
    """
    data_x  = []
    data_y  = []
    if ".csv" in filename:
        data    = np.genfromtxt(filename, delimiter = ",", skip_header = 1)
        data_x  = data[:, :-1]
        data_y  = data[:, -1:]
    elif ".npy" in filename:
        data    = np.load(filename)
        data_x  = data[:, :-1]
        data_y  = data[:, -1:]
    elif ".data" in filename:
        with open(filename, 'r') as f:
            counter = 0
            for line in f:
                data = line.split(',')
                if counter == 0:
                    data_x = np.array([data[:-1]], dtype = float)
                
                    data_y = np.array([data[-1].strip()], dtype = str)
                    data_y = np.reshape(data_y, newshape = (1,1,))
                    counter += 1
                else:
                    if data[-1].strip() != '':
                        x_add = np.array([data[:-1]], dtype = float)
                        data_x      = np.append(data_x, x_add, axis = 0)

                        y_add       = np.array([data[-1].strip()], dtype = str)
                        y_add       = np.reshape(y_add, newshape = (1,1,))
                        data_y      = np.append(data_y, y_add, axis = 0)
    return data_x, data_y


def convertWineDataToClasses(y):
    rows, _ = np.shape(y)
    y_temp  = np.empty(shape = (rows, 1,), dtype = '<U9')

    for row in range(rows):
        if y[row, 0] <= 4:
            y_temp[row] = 'poor'
        elif y[row, 0] < 7:
            y_temp[row] = 'median'
        elif y[row, 0] >= 7:
            y_temp[row] = 'excellent'
    return y_temp

def findMaxOccurences(countDict):
    maxOcc       = -np.inf
    for key in countDict.keys():
        if countDict[key] > maxOcc:
            maxOcc = countDict[key]
    return maxOcc

def balanceData(x, y, classes_name):
    countDict = {x:0 for x in classes_name}
    rows, _   = np.shape(x)
    data      = np.append(x, y, axis = 1)
    sorted_data = data[np.argsort(data[:, -1])]
    sorted_data = np.flip(sorted_data, axis = 0)
    for row in range(rows):    
        countDict[y[row, 0]] += 1
    maxOcc = findMaxOccurences(countDict)
    keys = list(countDict.keys())
    array_created = False
    for i in range(len(keys)):
        factor      = math.floor(maxOcc/countDict[keys[i]])
        if i == 0:
            copied_data = sorted_data[0:(countDict[keys[i]]), :]
            copied_data = np.repeat(copied_data, factor, axis = 0)
        elif i != 0:
            start = 0
            k = i
            while k > 0:
                start += countDict[keys[k-1]]
                k     -= 1
            copied_data = sorted_data[start:(start+countDict[keys[i]]), :]
            copied_data = np.repeat(copied_data, factor, axis = 0)

        # If array is not created create it:
        if not array_created:
            balanced_data = np.array(copied_data, copy = True)
            array_created = True
        else:
            balanced_data = np.append(balanced_data, copied_data, axis = 0)
    

    balanced_x = balanced_data[:, :-1]
    balanced_y = balanced_data[:, -1:]
    # Have to change x so that it displays float and not string
    balanced_x = balanced_x.astype(np.float)

    # Returns data duplicated so that it is relatively balanced for all classes
    return (balanced_x, balanced_y,)


def findMinOccurences(countDict):
    minOcc       = np.inf
    for key in countDict.keys():
        if countDict[key] < minOcc:
            minOcc = countDict[key]
    return minOcc


def balanceDataRemove(x, y, classes_name):
    countDict = {x:0 for x in classes_name}
    rows, _   = np.shape(x)
    data      = np.append(x, y, axis = 1)
    sorted_data = data[np.argsort(data[:, -1])]
    sorted_data = np.flip(sorted_data, axis = 0)
    for row in range(rows):    
        countDict[y[row, 0]] += 1
    minOcc = findMinOccurences(countDict)
    keys = list(countDict.keys())
    array_created = False

    for i in range(len(keys)):
        if i == 0:
            copied_data = sorted_data[0:(countDict[keys[i]]), :]
        elif i != 0:
            start = 0
            k = i
            while k > 0:
                start += countDict[keys[k-1]]
                k     -= 1
            extractedData = sorted_data[start:(start+countDict[keys[i]]), :]
            indexes       = np.arange(np.shape(extractedData)[0])
            np.random.shuffle(indexes)
            extractedData = extractedData[indexes, :]
            copied_data   = extractedData[:minOcc, :]

        # If array is not created create it:
        if not array_created:
            balanced_data = np.array(copied_data, copy = True)
            array_created = True
        else:
            balanced_data = np.append(balanced_data, copied_data, axis = 0)
    
    
    balanced_x = balanced_data[:, :-1]
    balanced_y = balanced_data[:, -1:]
    # Have to change x so that it displays float and not string
    balanced_x = balanced_x.astype(np.float)

    countDict = {x:0 for x in classes_name}
    rows, _   = np.shape(balanced_y)
    for row in range(rows):    
        countDict[balanced_y[row, 0]] += 1
    return (balanced_x, balanced_y,)