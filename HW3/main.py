#!/usr/bin/env python3

import tree
import load_data
import numpy as np

def main():
    x, y = load_data.load_data('iris.data')
    x_test, y_test = load_data.load_data('bezdekIris.data')
    decTree = tree.trainTree(x, y, 4)
    result  = tree.testTree(x_test, decTree)
    for i in range(np.shape(result)[0]):
        print(f'Classified: {result[i,0]} Real: {y_test[i,0]}')
    return 0

if __name__ == "__main__":
     main()