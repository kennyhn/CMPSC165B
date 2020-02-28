#!/usr/bin/env python3


import load_data
import numpy as np
import nfoldvalidation as nfv
from matplotlib import pyplot as plt


def main():
    x, y = load_data.load_data('iris.data')
    depth = np.arange(30)

    acc_test, acc_train = nfv.nFoldValidationIrisData(5, x, y)
    
    x, y = load_data.load_data('bezdekIris.data')
    acc_test1, acc_train1 = nfv.nFoldValidationIrisData(5, x, y)
    


    plt.title('Accuracy with 5-fold validation of Iris data', fontsize = 32)
    plt.grid()
    plt.plot(depth, acc_test, 'r-')
    plt.plot(depth, acc_train, 'b-')
    plt.xlim(0,30)
    plt.ylim(0,1)
    plt.legend(('Testing data', 'Training data',), fontsize = 20)
    plt.show()

    plt.title('Accuracy with 5-fold validation of Bezdekiris data', fontsize = 32)
    plt.grid()
    plt.plot(depth, acc_test1, 'r-')
    plt.plot(depth, acc_train1, 'b-')
    plt.legend(('Testing data', 'Training data',), fontsize = 20)
    plt.xlim(0, 30)
    plt.ylim(0, 1)
    plt.show()
    
    return 0

if __name__ == "__main__":
     main()