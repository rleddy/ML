from __future__ import division
import numpy as np
import sys

train_data = np.genfromtxt(sys.argv[1], delimiter = ",")

lam = 2
sigma2 = 0.1
d = 5


def randomVector(d,lmda):
    #
    vect = np.array([0]*d)
    #
    for k in range(0,d):
        vect[k] = np.random.normal(0.0,1.0/lmda)
    #
    return np.array(vect)


def buildMappingMatrix(lmda,sig2,d,vcts,N):
    scaleM = lmda*sig2*np.identity(d)
    for ii in range(0,N):
        v_i = vcts[ii]
        kronV = np.dot(v_i,np.transpose(v_i))
        scaleM += kronV
    mapper = np.linalg.inv(scaleM)
    return mapper

# Implement function here
def PMF(train_data):
    M = train_data
    N1 = train_data.shape[0]
    N2 = train_data.shape[1]
    V_matrices = [0]*50
    U_matrices = [0]*50
    L = [0]*50
    
    v_vectors = [ randomVector(d,lam) for ii in range(0,N2) ]
    u_vectors = [0]*N1
    
    for itr in range(0,50):
        V_matrices[itr] = np.matrix(v_vectors)
        #
        for i in range(0,N1):
            mapVector = np.sum([ M[i][j]*v_vectors[j] for j in range(0,N2) ])
            mapper = buildMappingMatrix(lam,sigma2,d,v_vectors,N2)
            u_vectors[i] = np.dot(mapper,mapVector)
        #
        U_matrices[itr] = np.matrix(u_vectors)
        #
        for j in range(0,N2):
            mapVector = np.sum([ M[i][j]*u_vectors[i] for i in range(0,N1) ])
            mapper = buildMappingMatrix(lam,sigma2,d,v_vectors,N2)
            v_vectors[j] = np.dot(mapper,mapVector)
        #
        v_ip_sum = 0.0
        for j in range(0,N2):
            v_j = v_vectors[j]
            v_ip_sum += np.dot(v_j,v_j)
        #
        u_ip_sum = 0.0
        for i in range(0,N1):
            u_i = u_vectors[i]
            u_ip_sum += np.dot(u_i,u_i)
        #
        ip_sum = (-0.5*lam)*(v_ip_sum + u_ip_sum)
        #
        ll = 0.0
        for i in range(0,N1):
            for j in range(0,N2):
                vv = M[i][j] - np.dot(u_vectors[i],v_vectors[j])
                ll += vv*vv
        #
        ll = (-0.5/sigma2)*ll
        L[itr] = ll + ip_sum
                
    return L, U_matrices, V_matrices


# Assuming the PMF function returns Loss L, U_matrices and V_matrices (refer to lecture)
L, U_matrices, V_matrices = PMF(train_data)

np.savetxt("objective.csv", L, delimiter=",")

np.savetxt("U-10.csv", U_matrices[9], delimiter=",")
np.savetxt("U-25.csv", U_matrices[24], delimiter=",")
np.savetxt("U-50.csv", U_matrices[49], delimiter=",")

np.savetxt("V-10.csv", V_matrices[9], delimiter=",")
np.savetxt("V-25.csv", V_matrices[24], delimiter=",")
np.savetxt("V-50.csv", V_matrices[49], delimiter=",")
