#!/usr/bin/env python3
import numpy                    as np
import binary_classification    as bc
from sklearn                    import svm
import comparison_functions     as cf

def SGDSolver(phase, x=None, y=None, alpha=[1e-5, 1e-3], lam=[1e-3, 2], nepoch=None, epsilon=10 ** (-6),
              param=None):
    
    lam             = [1e-6, 0.2]           # regularization weight [min, max]
    alpha           = [1e-6, 2e-4]
    nepoch          = 25                  # sample # of epochs

    y               = bc.classify_result(y) # Separate data into classes

    # normalize data
    if x is not None:
        x = massage_data(x)
    
    _, k            = np.shape(x)
    # alpha[0] = alpha[0]*(10**(-1))
    # alpha[1] = alpha[1]*(10**(-1))

    # Generate a random start parameters vector
   

    
    # Run SGD solver phases
    if phase.lower()    == 'training':
        w_start         = np.random.standard_normal(size = k) # w has dimension kx1
        w_start         = np.reshape(w_start, newshape = (k, 1,))
        b_start         = np.random.standard_normal(size = 1)
        param           = np.append(np.array(w_start, copy = True), b_start)
        param           = np.reshape(param, newshape = (k+1, 1,))
        y_class0            = bc.partition_into_class(y, 0)
        y_class1            = bc.partition_into_class(y, 1)
        y_class2            = bc.partition_into_class(y, 2)

        # Returns parameter matrix with weights and b for each class and also best alpha and lambda for each class
        #return training_phase(x, y, alpha, lam, param, nepoch, epsilon)
        param0, alpha0, lam0    = training_logistic_regression(x, y_class0, alpha, lam, param, nepoch, epsilon)
        param1, alpha1, lam1    = training_logistic_regression(x, y_class1, alpha, lam, param, nepoch, epsilon)
        param2, alpha2, lam2    = training_logistic_regression(x, y_class2, alpha, lam, param, nepoch, epsilon)

        param                   = np.append(param0, param1, axis = 0)
        param                   = np.append(param, param2, axis = 0)

        alpha                   = (alpha0 + alpha1 + alpha2)/3
        lam                     = (lam0 + lam1 + lam2)/3
        #alpha                   = np.array([[alpha0], [alpha1], [alpha2]])
        #lam                     = np.array([[lam0], [lam1], [lam2]])

        return param, alpha, lam
    elif phase.lower()  == 'validation':
        param                   = np.reshape(param, newshape = (3*k+3, 1))
        # returns error
        return validation_logistic_reg(x, y, param)
    elif phase.lower() == 'testing':
        # Returns vector with predicted y
        return testing_logistic_reg(x, param)
    else:
        print("You have not given a valid phase")



def training_logistic_regression(x, y, alpha, lam, param, nepoch, epsilon):
    # y are values that are either 0 or 1 for class 2 and class 1 respectively


    # How many steps should be taken when searching for the best lr and rt
    stepalph                = (alpha[1] - alpha[0]) / 15
    steplam                 = (lam[1] - lam[0]) / 10

    n, k                    = np.shape(x) # x have n rows and k columns
    
    lowest_crossentropy     = np.inf
    w_start                 = param[:k, :]
    bscalar_start           = param[-1, 0]

    best_parameters         = np.zeros(shape = (k+1, 1,)) # length of w + one value for the bias
    ones_n                  = np.ones(shape = (n, 1,))

    for regterm in np.arange(lam[0], lam[1]+steplam, steplam):
        for lr in np.arange(alpha[0], alpha[1]+stepalph, stepalph):
            # 2D grid search for best learning rate and regularization factor
            
            w           = np.copy(w_start)
            b_scalar    = bscalar_start

            parameters  = np.append(w, b_scalar)
            parameters  = np.reshape(param, newshape = (k+1, 1))

            # Start gradient descent method
            for _ in range(nepoch):
                # Calculate the gradient
                z_temp          = np.matmul(np.append(x, ones_n, axis = 1), parameters)
                f               = np.vectorize(bc.sigma_func)(z_temp)

                grad_w          = -np.matmul(np.transpose(y-f), x)
                grad_w          = np.reshape(grad_w, newshape = (k, 1))
                grad_w          += 2*regterm*w
                

                # Calculate the gradient with the current scalar bias b    
                grad_b          = -np.matmul(np.transpose(y-f), ones_n)

                # Update w and b_scalar for next iteration
                w               += -lr*grad_w
                b_scalar        += -lr*grad_b[0][0]
                param           = np.append(w, b_scalar)
                param           = np.reshape(param, newshape = (k+1, 1))
                
                # Calculate cross entropy with new w and b
                crossentropy    = bc.cross_entropy(x, y, param)

                if crossentropy < epsilon:
                    return param, lr, regterm

                # Save the best parameter values w and b
                if crossentropy < lowest_crossentropy:
                    lowest_crossentropy = crossentropy
                    best_alpha          = lr
                    best_lam            = regterm
                    best_parameters     = parameters
    return (best_parameters, best_alpha, best_lam,)


def validation_logistic_reg(x, y, param):
    # Take in parameters for all three classes
    n, _    = np.shape(x)
    y_pred  = bc.classify_prediction(x, param)
    mse     = (1/n)*np.matmul(np.transpose(y-y_pred), y-y_pred)
    return mse[0][0]


def testing_logistic_reg(x, param):
    return bc.classify_prediction(x, param)


def training_phase(x, y, alpha, lam, param, nepoch, epsilon):

    # How many steps should be taken when searching for the best lr and rt
    stepalph        = (alpha[1] - alpha[0]) / 100
    steplam         = (lam[1] - lam[0]) / 10

    n, k            = np.shape(x) # x have n rows and k columns
    
    
    lowest_error    = np.inf
    w_start         = param[:k, :]
    bscalar_start   = param[-1, 0]
    best_parameters = np.zeros(shape = (k+1, 1,)) # length of w + one value for the bias
    ones_n          = np.ones(shape = (n, 1,))

    for regterm in np.arange(lam[0], lam[1]+steplam, steplam):
        for lr in np.arange(alpha[0], alpha[1]+stepalph, stepalph):
            # 2D grid search for best learning rate and regularization factor
            
            w           = np.copy(w_start)
            b_vector    = np.copy(bscalar_start*ones_n)
            b_scalar    = bscalar_start
            
            # Start gradient descent method
            for _ in range(nepoch):
                # Calculate the gradient with the current weighting matrix w
                grad_w      = -(2/n)*np.matmul(np.transpose(x), y)
                grad_w      += (2/n)*np.matmul(np.matmul(np.transpose(x), x), w)
                grad_w      += (2/n)*np.matmul(np.transpose(x), b_vector)
                grad_w      += 2*regterm*w

                # Calculate the gradient with the current scalar bias b    
                grad_b      = -(2/n)*np.matmul(np.transpose(ones_n), y)
                grad_b      += (2/n)*np.matmul(np.transpose(ones_n), np.matmul(x,w))
                grad_b      += 2*b_scalar


                # Update w and b_scalar for next iteration
                w           += -lr*grad_w
                b_scalar    += -lr*grad_b[0][0]

                # calculate new b_vector
                b_vector    = b_scalar*ones_n

                # Calculate MSE with new w and b
                error       = y-np.matmul(x, w)-b_vector
                mse         = (1/n)*np.matmul(np.transpose(error), error)
                mse         += regterm*np.matmul(np.transpose(w), w)

                if mse[0][0] < epsilon:
                    return np.append(np.array(w, copy = True), b_scalar)

                # Save the best parameter values w and b
                if mse[0][0] < lowest_error:
                    lowest_error    = mse[0][0]
                    best_parameters = np.append(np.array(w, copy = True), b_scalar)
                    best_parameters = np.reshape(best_parameters, newshape = (k+1, 1,))
    return best_parameters



def validation_phase(x, y, param):
    # sum of error
    n,k         = np.shape(x)
    ones_n      = np.ones(shape = (n, 1,))
    w           = param[:k]  # retrieve elements from 0 to k-1
    w           = np.reshape(w, newshape = (k,1,))
    b           = param[-1]  # last element is b

    # Calculate the mean square error
    error       = y - np.matmul(x,w)-b*ones_n

    mse         = (1/n)*np.matmul(np.transpose(error), error)
    return mse[0][0]


def testing_phase(x, param):
    n,k         = np.shape(x)
    w           = param[:k]
    w           = np.reshape(w, newshape = (k,1,))
    b           = param[-1]
    ones_n      = np.ones(shape = (n,1))
    return np.matmul(x,w) + b*ones_n


def massage_data(x): # kxn array:
    for n_column in range(np.shape(x)[1]):

        # Reset max_value and min_value
        max_value = -np.inf
        min_value = np.inf

        # Find min and max
        for m_row in range(np.shape(x)[0]):
            if x[m_row][n_column] < min_value:
                min_value      = x[m_row][n_column]
            if x[m_row][n_column] > max_value:
                max_value      = x[m_row][n_column]
        
        # Normalize data in each column
        for m_row in range(np.shape(x)[0]):
            x[m_row][n_column] = (x[m_row][n_column] - min_value)/(max_value-min_value)
            
    return x



def training_with_svm(x, y_real):
    n, _                = np.shape(x)
    x                   = massage_data(x)
    classifier_class0   = svm.SVC(C = 0.05, kernel = 'linear')
    classifier_class1   = svm.SVC(C = 0.05, kernel = 'linear')
    classifier_class2   = svm.SVC(C = 0.05  , kernel = 'linear')

    y_class0            = np.zeros(shape = (n, 1,), dtype = int)
    y_class1            = np.zeros(shape = (n, 1,), dtype = int)
    y_class2            = np.zeros(shape = (n, 1,), dtype = int)

    class_count = [0, 0, 0]


    for i in range(n):
        class_count[int(y_real[i][0])] += 1
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

    y_pred0                 = classifier_class0.predict(x)
    y_pred1                 = classifier_class1.predict(x)
    y_pred2                 = classifier_class2.predict(x)

    y_pred_result       = np.zeros(shape = (n, 1,))
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
    
    mse     = (1/n)*np.matmul(np.transpose(y_pred_result-y_real), y_pred_result-y_real)
    print("The MSE with SVM: {}".format(mse))
    return (classifier_class0, classifier_class1, classifier_class2)


def test_with_svm(x, y, classifier0, classifier1, classifier2):
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

    cf.print_confusion_matrix(confusion_matrix)
    cf.print_accuracy(accuracy)