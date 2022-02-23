import numpy as np
import sys

lambda_input = int(sys.argv[1])
sigma2_input = float(sys.argv[2])
X_train = np.genfromtxt(sys.argv[3], delimiter = ",")
y_train = np.genfromtxt(sys.argv[4])
X_test = np.genfromtxt(sys.argv[5], delimiter = ",")


#numpy.linalg.pinv


    ## Input : Arguments to the function
    ## Return : wRR, Final list of values to write in the file
    


## Solution for Part 1
def part1():
    global y_train
    
    N = y_train.shape[0]
    
    # center Y
    mY = y_train.mean()
    y_train = y_train - mY
    
    # normalize X
    x_rrTrain = X_train[0:,1:]
    for i in range(0,N):
        rowMean = x_rrTrain[i].mean()
        sigma_i = np.std(x_rrTrain[i], dtype=np.float64)
        #
        x_rrTrain[i] = x_rrTrain[i] - rowMean
        x_rrTrain[i] = x_rrTrain[i]/sigma_i
    # 
    X_transpose = np.transpose(x_rrTrain)
    
    XXT = np.dot(X_transpose,x_rrTrain)
    d = XXT.shape[0]
    
    ident = np.identity(d)
    ident = lambda_input*ident
    #
    M = XXT + ident;
    M = np.linalg.inv(M);
    
    wwRR = np.dot(M,X_transpose)
    wwRR = np.dot(wwRR,y_train)
    
    #print wwRR
    
    return wwRR



wRR = part1()  # Assuming wRR is returned from the function
np.savetxt("wRR_" + str(lambda_input) + ".csv", wRR, delimiter="\n") # write output to file


## Solution for Part 2
def part2():
    ## Input : Arguments to the function
    ## Return : active, Final list of values to write in the file
    topTen = [0,0,0,0,0,0,0,0,0,0]
    N = y_train.shape[0]
    d = X_train.shape[1]
    
    covariences = np.identity(d)
    covariences = lambda_input*covariences
    for i in range(0,N):
        x_i = X_train[i]
        covary_x_i = np.dot(np.transpose(x_i),x_i);
        covariences += (1.0/sigma2_input)*covary_x_i
        
    SigInvOp = np.linalg.inv(covariences)
    
    for itr in range(0,10):
        maxVary = -1
        x_0 = 0
        ii = 0
        for i in range(0,N):
            if i in topTen:
                continue
            x_i = X_train[i]
            x_i_T = np.transpose(x_i)
            vary0 = sigma2_input + np.dot(x_i,np.dot(SigInvOp,x_i_T))
            if ( vary0 > maxVary ):
                maxVary = vary0
                x_0 = x_i
                ii = i
                
        topTen[itr] = ii
        #
        covary_x_i = np.dot(np.transpose(x_0),x_0);
        covariences += (1.0/sigma2_input)*covary_x_i
        SigInvOp = np.linalg.inv(covariences)
     
    
    return topTen

active = part2()  # Assuming active is returned from the function
np.savetxt("active_" + str(lambda_input) + "_" + str(int(sigma2_input)) + ".csv", active, delimiter=",") # write output to file


