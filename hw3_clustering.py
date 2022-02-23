import numpy as np
import pandas as pd
import scipy as sp
import sys

import randint from random

X = np.genfromtxt(sys.argv[1], delimiter = ",")

def findClosest(xdat,centerslist):
    mindist = -1
    mink = -1
    for k in range(0,k):
        meanX = centerslist[k]
        xdiff = xdat - meanX;
        dist = sqrt(np.dot(xdiff,xdiff))
        mindist = dist if ( mindist == -1 ) else mindist
        if ( dist < mindist ):
            mindist = dist;
            mink = k
    return mink

def findMean(cluster):
    meanX = cluster[0]
    n = len(cluster)
    for i in range(1,n):
        meanX += cluster[i]
    meanX = meanX/n
    return meanX

def KMeans(data):
	#perform the algorithm with 5 clusters and 10 iterations...you may try others for testing purposes, but submit 5 and 10 respectively
	
	clusters = [[],[],[],[],[]]
	centerslist = [0,0,0,0,0]
	
	n = data.shape[0]
	indecies = [0,0,0,0,0]
	while (indecies[0] == 0) or (indecies[1] == 0) or(indecies[2] == 0) or(indecies[3] == 0) or (indecies[4] == 0):
        idx = randint(0,n)
        if (indecies[0] == idx) or (indecies[1] == idx) or(indecies[2] == idx) or(indecies[3] == idx) or (indecies[4] == idx):
            continue
        for j in range(0,5):
            if ( indecies[j] == 0 ):
                indecies[j] = idx
                break
            
    centerslist = [ data[indecies[k]] for k in range(0,5) ]
	
	for i in range(0,10):
        for l in range(0,n):
            xdat = data[l]
            k = findClosest(xdat,centerslist)
            clusters[k].append(xdat)
        
        centerslist = [ findMean(clusters[k]) for k in range(0,5) ]
        
        filename = "centroids-" + str(i+1) + ".csv" #"i" would be each iteration
        np.savetxt(filename, centerslist, delimiter=",")

def calcNormals(x_i,pis,mu_s,Sigmas):
    probs = [0,0,0,0,0]
    #
    for k in range(0,5):
        #
        pi = pis[k]
        mu = mu_s[k]
        Sigma = Sigmas[k]
        #
        shiftedX = x_i - mu
        detS = np.linalg.det(Sigma)
        SigmaInv = np.linalg.inv(Sigma)
        #
        if ( det <> 0 ):
            coef = pi/sqrt(det)
            gamma = np.dot(np.transpose(shiftedX),np.dot(SigmaInv,shiftedX))*(-0.5);
            probs[k] = coef*exp(gamma)
        #
    return probs


def makeCovarience(covarienceM,mu,data,phi,d,N):
    #
    covarienceM = np.matrix([0]*d,[0]*d)
    #
    for i in range(0,N):
        x_i = data[i]
        x_i -= mu;
        x_i_trns = np.transpose(x_i)
        cv_i = np.dot(x_i_trns,x_i)
        cv_i = phi*cv_i
        covarienceM += cv_i
    covarienceM = covarienceM/covarienceM
    return covarienceM

def EMGMM(data):
    
    d = data.shape[1]
    N = data.shape[0]
    
    pis = [(1.0/5),(1.0/5),(1.0/5),(1.0/5),(1.0/5)]
    Sigmas = [ np.identity(d), np.identity(d), np.identity(d), np.identity(d), np.identity(d) ]
    mu_s = [ np.array([0]*d), np.array([0]*d), np.array([0]*d), np.array([0]*d), np.array([0]*d) ]
    phi_s = [[0]*5]*N

	for i in range(0,10):
        #
        #calculate initial phi's
        for ii in range(0,N):
            normalsProbs = calcNormals(data[ii],pis,mu_s,Sigmas)
            sumProbs = np.sum(normalsProbs)
            phi_s[ii] = [ normalsProbs[k]/sumProbs for k in range(0,5) ]
        #
        Nk_s = [0,0,0,0,0]
        for ii in range(0,N):
            for k in range(0,5):
                Nk_s[k] += phi_s[ii][k]
        
        pis = [Nk_s[k]/N for k in range(0,5)]
        mu_s = [ np.array([0]*d), np.array([0]*d), np.array([0]*d), np.array([0]*d), np.array([0]*d) ]
        #
        for ii in range(0,N):
            x_i = data[ii]
            for k in range(0,5):
                mu_s[k] += (phi_s[ii][k])*x_i
        for k in range(0,5):
            mu_s[k] =  mu_s[k]/Nk_s[k]
        #
        for k in range(0,5):
           Sigmas[k] = makeCovarience(Nk_s[k],mu_s[k],data,phi_s,d,N) 
        
        filename = "pi-" + str(i+1) + ".csv" 
        np.savetxt(filename, pis, delimiter=",") 
        filename = "mu-" + str(i+1) + ".csv"
        np.savetxt(filename, mu_s, delimiter=",")  #this must be done at every iteration
        for j in range(k): #k is the number of clusters 
            filename = "Sigma-" + str(j+1) + "-" + str(i+1) + ".csv" #this must be done 5 times (or the number of clusters) for each iteration
            np.savetxt(filename, Sigmas[j], delimiter=",")
    
    
