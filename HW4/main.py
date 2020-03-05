#!/usr/bin/env python3

import data_handling as dh
import comparison_functions as cf
import tree

def main():
    red_x, red_y            = dh.load_data('hw2_winequality-red_train.npy')
    red_y                   = dh.convertWineDataToClasses(red_y)
    red_test_x, red_test_y  = dh.load_data('hw2_winequality-red_test.npy')
    red_test_y              = dh.convertWineDataToClasses(red_test_y)

    balanced_x, balanced_y  = (red_x, red_y,)

    class_dictionary        = {'poor': 0, 'median': 1, 'excellent': 2}
    #balanced_x, balanced_y  = dh.balanceData(red_x, red_y, class_dictionary.keys())



    ensemble                = tree.trainBaggingEnsemble(balanced_x, balanced_y, 15, 5)
    y_pred                  = tree.testBaggingEnsemble(red_test_x, ensemble)
    confusion_matrix        = cf.calculate_confusion_matrix(y_pred, red_test_y, class_dictionary)
    cf.print_confusion_matrix(confusion_matrix)

    print(tree.calculateAccuracy(y_pred, red_test_y))
    return 0



if __name__ == "__main__":
    main()
    