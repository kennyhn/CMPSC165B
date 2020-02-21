#!/usr/bin/env python3
import numpy as np

def load_data(filename): # for loading data files
    # .data is structured with 5 attributes divided by ','
    # last entry is the class the data belongs to
    # the entry before is the features of that data
    with open(filename, 'r') as f:
        counter = 0
        for line in f:
            data = line.split(',')
            if counter == 0:
                x = np.array([data[:-1]], dtype = float)
                
                y = np.array([data[-1].strip()], dtype = str)
                y = np.reshape(y, newshape = (1,1,))
                counter += 1
            else:
                if data[-1].strip() != '':
                    x_add = np.array([data[:-1]], dtype = float)
                    x       = np.append(x, x_add, axis = 0)

                    y_add = np.array([data[-1].strip()], dtype = str)
                    y_add = np.reshape(y_add, newshape = (1,1,))
                    y = np.append(y, y_add, axis = 0)
    return (x,y,)