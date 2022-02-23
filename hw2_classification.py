from __future__ import division
import numpy as np
import sys

X_train = np.genfromtxt(sys.argv[1], delimiter=",")
y_train = np.genfromtxt(sys.argv[2])
X_test = np.genfromtxt(sys.argv[3], delimiter=",")

## can make more functions if required


def make_prediction(x,classProbs,classMeans,classDets,classCovInv):
    
    predictions = [0,0,0,0,0,0,0,0,0,0]
    
    for y in range(0,10):
        shiftedX = x - classMeans[y]
        coef = classProbs[y]/sqrt(classDets[y])
        gamma = np.dot(np.transpose(shiftedX),np.dot(classCovInv[y],shiftedX))*(-0.5);
        predictions[y] = coef*exp(gamma)
    return predictions

def pluginClassifier(X_train, y_train, X_test):
    # this function returns the required output
    classCounts = [0,0,0,0,0,0,0,0,0,0]
    classProbs = [0,0,0,0,0,0,0,0,0,0]
    classMeans = [0,0,0,0,0,0,0,0,0,0]
    classDets = [0,0,0,0,0,0,0,0,0,0]
    classCovariance = [0,0,0,0,0,0,0,0,0,0]
    classBuckets = [[],[],[],[],[],[],[],[],[],[]]
    #
    n  = y_train.shape[0]
    fn = X_train[0].shape[0]
    #
    for i in range(0,n):
        y = y_train[i]
        classBuckets[y].append(X_train[i]);
        classProbs[y] += 1
        classCounts[y] += 1

    for y in range(0,10):
        classProbs[y] = classProbs[y]/n
    #print classProbs
    
    # means for classes
    for y in range(0,10):
        bn = classCounts[y]  # how many vectors in this class
        bsum = np.array([0]*fn)
        for i in range(0,bn):
            bsum += classBuckets[y][i]
        bsum = bsum/bn
        classMeans[y] = bsum
    #
    for y in range(0,10):
        bn = classCounts[y]  # how many vectors in this class
        SigmaY = np.array([[0]*fn]*fn)
        for i in range(0,bn):
            x_i = classBuckets[y][i] - classMeans[y]
            kron = np.dot(x_i,np.transpose(x_i));
            SigmaY += kron;
        classCovariance[y] = SigmaY/bn
        classDets[y] =  np.linalg.det(classCovariance[y])
        if ( classDets[y] > 0 ):
            classCovariance[y] = np.linalg.inv(classCovariance[y])
    #
    

    outputs = [make_prediction(x,classProbs,classMeans,classDets,classCovariance) for x in X_test]
    return outputs



final_outputs = pluginClassifier(X_train, y_train, X_test) # assuming final_outputs is returned from function

np.savetxt("probs_test.csv", final_outputs, delimiter=",") # write output to file
