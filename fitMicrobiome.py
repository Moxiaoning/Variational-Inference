
import re
import numpy as np
import matplotlib.pyplot as plt
import sklearn as sk
import scipy.io
from scipy import stats
from sklearn.model_selection import train_test_split
import matplotlib.pylab as plt
from time import time
from numpy import savetxt
start = time()


###############################PREPARE DATA SET###################################
#For desktop
# filepath = 'C:\\Users\\Jack\\Downloads\\combined_data'
# #For laptop
# #filepath = "C:\\Users\\jackx\\Downloads\\david"
# data = scipy.io.loadmat(filepath)
# #print(data.keys())
# def superduperreduce(M):
# # inputs: otu data (n or xs)
# # outpus: reduced otu data with higher cutoff than reduce(M) for smaller working datasets
#     reduced = []
#     for i in range(np.shape(M)[1]):
#         if (np.sum(M[:, i])/np.shape(M)[0]) > 0.004:
#             reduced.append(M[:, i])
#     return np.transpose(np.asarray(reduced))
# def reduce(M):
# # inputs: otu data (n or xs)
# # outpus: reduced otu data
#     reduced = []
#     for i in range(np.shape(M)[1]):
#         if (np.sum(M[:, i])/np.shape(M)[0]) > 0.001:
#             reduced.append(M[:, i])
#     return np.transpose(np.asarray(reduced))
# def reduceSamples(M, N):
#     species = np.where(data['cow_sps'] == 1)[1]
#     Mreduce = np.asarray([M[i] for i in species])   
#     Nreduce = np.asarray([N[i] for i in species])
#     return Mreduce, Nreduce
# ################# DATA ############################
# # metx - [939, 53] 53 metadata features
# #xs - [939, 17364]
# #Get from data and do not change
# m = data['metx'] #metadata features (metx)
# n = reduce(data['xs'].toarray()) #num counts (xs)
# print(np.shape(n))

# m, n = reduceSamples(m, n)





n = np.genfromtxt("ndat.csv", delimiter = ',')





################## CONSTANTS#######################
#Get from data and do not change
s = np.shape(n)[0]
#p = np.shape(m)[1]
o = np.shape(n)[1]

N = np.ones(s) #total counts
N += 9999
#################### HYPERPARAMETER ################
#Model specific
a = 0
#a = 1 favors metadata
#a = 0 favors microbiome
k = 5
kc = k
ko = k #latent dimensions zo
#################################################
#Gradient Ascent

### Good eta values for learning from full dataset. Does not lead to convergence on the partial data set

etazo = 10**-6
etat = 10**-6

maxiter = 50000
# etazp = 10**-14
# etazo = 10**-14
# etat = 10**-14
# etap = 10**-14

#temp: (zp, zo, theta, phi)


################# LEARNED VARIABLES ###############
# phi = np.ones([kp, p])
# zp = np.ones([s, kp])

# theta = np.ones([ko, o])
# zo = np.ones([s, ko])


def dtheta(x, ZO, N, THETA):
# calculated as in eqtn (8)
#inputs: q, n, z
#outputs: dtheta (k x o)
    q = calcQ(ZO, THETA)

    #print(np.sum(N * (q[:, 0] - x[:, 0]) * ZO[:, 1]))
    dt = (1-a) * N[0] * np.matmul(np.transpose(ZO), (q - x))
    #dt = np.asarray([[(1 - a) * np.sum(N * (q[:, OO] - x[:, OO]) * ZO[:, KK]) for OO in range(np.shape(THETA)[1])] for KK in range(np.shape(THETA)[0])])
    #print(np.sum(np.sum(dtheta)))
    #dtheta = dtheta / np.linalg.norm(dtheta) 
    return dt
def dzo(ZO, THETA, x, N):
#calculates dzo as in eqtn (9)
#Output: dzo (s x ko)

    
    q = calcQ(ZO, THETA)
    #print(np.sum(x[0, :]))
    #term2 = [[(1-a) * N[SS] * np.sum((q[SS, :] - x[SS, :]) * THETA[KK, :]) for KK in range(ko)] for SS in range(np.shape(ZO)[0])]
    dzo = N[0] * np.matmul((q - x), np.transpose(THETA))
    #dzo = dzo / np.linalg.norm(dzo)
    return dzo
def calcX(n, N):
#calc x as counts over total counts
#inputs: n (counts for each otu), N (total counts)
#outpus: x (s x p)

    return np.asarray(calcN(n, N) / N[0])
def calcQ(ZO, THETA):
#Calc q in (4)
#inputs: zo, theta
#outputs: Q (s x p)
    #print(np.max(np.matmul(ZO, THETA)))
    Q = np.exp(-1 * np.matmul(ZO, THETA))
    Q = Q/np.linalg.norm(Q, ord=1, axis=1, keepdims=True)
    #Q = np.asarray([Q[i]/np.sum(Q[i]) for i in range(np.shape(Q)[0])])
    return Q
def calcN(Q, N):
#Calc n counts given prob q (3)
#inputs: Q, N
#outputs: n ()
    n = [np.random.multinomial(N[i], Q[i]) for i in range(np.shape(Q)[0])]
    return np.asarray(n)
def train(ndat):
#
#Initialize hyperparameters
    #while not converged
      #update x = x + eta * dx
    # phi = np.ones([kp, p])
    # zp = np.ones([s, kp])

    # theta = np.ones([ko, o])
    # zo = np.ones([s, ko])
    
    S = np.shape(ndat)[0]
    N = np.zeros(S) #total counts (s x 1)
    N += 10000
    xdat = calcX(ndat, N)
    
    #phi = np.random.rand(kp, p)
    theta = np.random.rand(ko, o)
    zc = np.random.rand(S, kc)

    zo = np.random.rand(S, ko - kc)
    zo = np.append(zc, zo, axis = 1)

    grad = [[], [], []]
    #grad[0]: dtheta
    #grad[1]: dzo
    #grad[2]: ELBO
    converged = False
    count = 0
    while(converged == False):
        temp = []
        #phi2 = phi
        #zp2 = zp
        theta2 = theta
        zo2 = zo

        #Update matrices##############
        #zp = zp + etazp * dzp(mdat, zp, phi, zo, theta, ndat, N)
        dt = dtheta(xdat, zo, N, theta)
        dz = dzo(zo, theta, xdat, N)
        grad[0].append(np.linalg.norm(dz)/np.linalg.norm(zo))
        grad[1].append(np.linalg.norm(dt)/np.linalg.norm(theta))
        grad[2].append(calcL(zo, theta, xdat))
        #grad[2].append(0)

        theta = theta + etat * dt
        zo = zo + etazo * dz 

        temp.append(np.linalg.norm(dz)/np.linalg.norm(zo))
        temp.append(np.linalg.norm(dt)/np.linalg.norm(theta))

        temp = [np.round(temp[i], 6) for i in range(np.shape(temp)[0])]
        #temp: (zp, zo, theta, phi)
        
        #print(temp)
        #temp = np.abs(np.sum(np.sum(zo[:, 0:kc] - zp[:, 0:kc])))


        ############################ TEST PEARSON COEFF
    
        #temp.append(compareMetadata(mdat, phi, zp)[0])
        temp.append(compareMicrobiome(xdat, theta, zo)[0])
        if (temp[0] < 2 and temp[1] < 2) or count > maxiter: #or count > maxiter:
            converged = True
        if np.isnan(np.sum(temp)):
            print("NaN")
            break

        count += 1

        print(count, temp) #change in learned variables
    #plots(grad)


        #if count == 10:
           # break
        #print(count, lossMu(zp, phi), lossQ(zo, theta)) #difference between m, mu and q, x
    
    #Converged criteria: 
    #zo ~ zp for 0 to kc
    #phi - phi(-1) < some value for all 4 matrices
    np.savetxt("theta nonvar.csv", theta, delimiter = ',')
    np.savetxt("Z nonvar.csv", zo, delimiter = ',')
    return theta, zo, xdat
def encodeZO(ndat, THETA):
#Input: otu counts (ndat), encoder (theta)
#Output: zo
    Zo = np.random.rand(np.shape(ndat)[0], ko)
    x = calcX(ndat, N)
    converged = False
    while(converged == False):
        q = calcQ(Zo, THETA)
        dzo = np.asarray([[(1-a) * N[SS] * np.sum((q[SS, :] - x[SS, :]) * THETA[KK, :]) for KK in range(ko)] for SS in range(np.shape(ndat)[0])])
        deltaZo = Zo
        Zo = Zo + etazo * dzo
        deltaZo = deltaZo - Zo
        if(np.abs(np.sum(np.sum(deltaZo))) < 1):
            converged = True
    return Zo
def testModel():
#Train test split model
    #print(np.shape(m))
    # ntrain, ntest, mtrain, mtest = train_test_split(n, m, test_size = 0.2)
    ntest = n
    Theta, zoo, xd = train(ntest)

    S = np.shape(ntest)[0]
    N = np.zeros(S) #total counts
    N += 10000
    xtest = calcX(ntest, N)
    zotest = encodeZO(xtest, Theta)
    print("END: ", compareMicrobiome(xtest, Theta, zotest))
    savetxt('latents.csv', zoo, delimiter=',')
    savetxt('theta.csv', Theta, delimiter=',')
    plt.plot(zoo[:, 0], zoo[:, 1], 'o')
    plt.xlabel("z1")
    plt.ylabel("z2")
    plt.show()
    # plt.plot(zotest[:, 0], zotest[:, 1], 'o')
    # plt.xlabel("z1")
    # plt.ylabel("z2")
    # plt.show()
    return 0
def compareMicrobiome(ndat, THETA, ZO):
#See compareMetadata
    qpred = calcQ(ZO, THETA)
    npred = calcN(qpred, N)
    return pearsonCoeff(ndat, npred)
def pearsonCoeff(dat, pred):
#input: data matrix and prediction matrix
#output: pearson corr coeff between two matrices
    dat = np.ndarray.flatten(dat)
    pred = np.ndarray.flatten(pred)
    return stats.pearsonr(dat, pred)
def learnZO(ndat, THETA):
    Zo = np.random.rand(np.shape(ndat)[0], np.shape(THETA)[0])
    converged = False
    while(converged == False):
        x = calcX(ndat, N)
        q = calcQ(Zo, THETA)
        dzo = [[(1-a) * N[SS] * np.sum((q[SS, :] - x[SS, :]) * THETA[KK, :]) for KK in range(ko)] for SS in range(np.shape(ndat)[0])]
        deltaZo = Zo
        Zo = Zo + etazo * dzo
        deltaZo = deltaZo - Zo
        if(np.abs(np.sum(np.sum(deltaZo))) < 0.0001):
            converged = True
    return Zo
def learnZP(mdat, PHI):
#input: mu (data), PHI (learned)
#output: zp given as mu times right inverse of phi
    return np.matmul(mdat, np.linalg.pinv(PHI))
def calcL(ZO, THETA, X):
#input: ZO, THETA, X
#output: likelihood of X given ZO, THETA
    L = -1 * np.sum(np.sum(X * (np.matmul(ZO, THETA) + np.log(calcOmega(ZO, THETA)))))
    return L
def calcOmega(ZO, THETA):
#input: mu, sigma, theta
#output: normalization factor omega
    pi = np.exp(-1 * np.matmul(ZO, THETA))
    PI = np.linalg.norm(pi, ord=1, axis=1, keepdims=True)


    return PI
def plots(GRADIENT):
    x = np.linspace(1, np.shape(GRADIENT)[1], np.shape(GRADIENT)[1])
    plt.plot(x, GRADIENT[0], 'bo')
    plt.plot(x, GRADIENT[1], 'go')
    
    plt.legend(['dTheta', 'dZo'])
    plt.yscale('log')
    plt.show()
    plt.clf()
    plt.plot(x, GRADIENT[2], 'ro')
    plt.show()
    return 0

testModel()
print(time()-start, " seconds")
        #6/1/2022
        #remove loops in code
        #Helpful: Run on 6/9 for MNIST 
        #ELBO as function of iteration
        #Gradient magnitudes as function of iter
        #Want convergence based on magnitude of gradient

#6/3/2022
#Local correlations in the neighborhood of any given microbiome
#ex: take two different points in latent space. find probabilities
#at these points, and find correlation between specific bacteria abundances
# #for these two points
# ^Question that can be asked w/ variational model and can't w/ nonvariational model
#  Evaluate gradient for given z, theta two different times. and ensure that they are 
#sufficiently correlated
#Check for sign (gradient ascent vs descent)
#Try fixing sigma to 1 (only learn mu and theta) (check accuracy of this)
#Try lower learning rates (10**-5/10**-6)
#epsilon should be s x k matrix (one epsilon for each gradient)