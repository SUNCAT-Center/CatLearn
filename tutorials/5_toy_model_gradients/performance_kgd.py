import numpy as np
from atoml.regression.gpfunctions.gkernels import gkernels as gkernels
from timeit import default_timer as timer
from scipy.spatial import distance

# Method 1: broadcast, Method 2: for_train_loop, Method3: old for loops.
# Method 4: broadcast+cdist

#T500/D100, 1=0.715  2=0.209  3=45.8  4=0.384  5=1.46 6=0.201  7=0.184 #2,7,6
#T100/D100, 1=0.026  2=0.007  3=1.21  4=0.016  5=0.05 6=0.006  7=0.006 #6,7,2
#T10/D1000, 1=0.006  2=0.001  3=1.27  4=0.003  5=0.002 6=0.001 7=0.003 #2,5,6,7
#T50/D1000, 1=0.067  2=0.041  3=39.6  4=0.054  5=0.049 6=0.046 7=0.053 #2,6,7,5,1
#T1000/D10, 1=0.221  2=0.098  3=47.79 4=0.169  5=4.15  6=0.107 7=0.145 #2,6,7,4
#T1000/D50, 1=1.28   2=0.376  3=84.87 4=0.860  5=4.56  6=0.352 7=0.464 #6,2,7
#T10000/D2, 1=18.8   2=5.698  3=BAD   4=10.29  5=BAD   6=5.53 7=5.43  #6,7,2
#T10/D10000,1=0.019  2=0.006  3=NT    4=0.012  5=0.005 6=0.0079 7=0.019 #2,6,7
#T500/D500, 1=2.989  2=1.34   3=NT    4=2.24   5=1.96  6=1.34  7=1.39 #2,6,7
#T1000/D500,1=103.7  2=98.06  3=NT    4=53.74  5=8.15  6=4.31  7=4.38 #6,7
#T1000/D1000, 1=NT   2=  3=   4=  5=  6=21.45  7=21.58 # 6,7

method = '7'
np.random.seed(1)
m1 = []
train_points = 1000
dimensions = 500
iterations = 1

m1= 1.2*np.random.randint(5.0, size=(train_points,
dimensions))


kwidth = np.zeros(np.shape(m1)[1])+2.0
time=[]


for i in range(0,iterations):
    start=timer()
    loop_number = i

    if method=='1':
        size = np.shape(m1)
        d = (m1[np.newaxis,:,:] - m1[:,np.newaxis,:])
        invsqkwidth = 1/kwidth**2
        den = np.sum(invsqkwidth*d**2,axis=-1)
        k1 = np.exp(-den/2)
        # print(k1)
        big_kgd = -(invsqkwidth*d*k1[:,:,np.newaxis]).reshape(size[0],
        size[0]*size[1])
        # print(big_kgd)

    if method=='2':
        size = np.shape(m1)
        big_kgd = np.zeros((size[0],size[0]*size[1]))
        invsqkwidth = 1/kwidth**2
        k1 = distance.pdist(m1 / kwidth, metric='sqeuclidean')
        k1 = distance.squareform(np.exp(-.5 * k1))
        np.fill_diagonal(k1, 1)
        # print(k1)
        for i in range(size[0]):
            d = m1[:,:]-m1[i,:]
            big_kgd_i = ((invsqkwidth * d).T * k1[i]).T
            big_kgd[:,(size[1]*i):(size[1]+size[1]*i)] = big_kgd_i
        # print(big_kgd)

    if method=='3':
         k1 = distance.pdist(m1 / kwidth, metric='sqeuclidean')
         k1 = distance.squareform(np.exp(-.5 * k1))
         np.fill_diagonal(k1, 1)
         # print(k1)
         big_kgd = gkernels.big_kgd('squared_exponential',kwidth,m1)
         # big_kdd = gkernels.big_kdd('squared_exponential',kwidth,m1)
         # print(big_kgd)

    if method=='4':
        size = np.shape(m1)
        k1 = distance.pdist(m1 / kwidth, metric='sqeuclidean')
        k1 = distance.squareform(np.exp(-.5 * k1))
        np.fill_diagonal(k1, 1)
        d = (m1[np.newaxis,:,:] - m1[:,np.newaxis,:])
        invsqkwidth = 1/kwidth**2
        # print(k1)
        big_kgd = -(invsqkwidth*d*k1[:,:,np.newaxis]).reshape(size[0],
        size[0]*size[1])
        # print(big_kgd)

    if method=='5':
        k = distance.pdist(m1 / kwidth, metric='sqeuclidean')
        k = distance.squareform(np.exp(-.5 * k))
        np.fill_diagonal(k, 1)
        size = np.shape(m1)
        big_kgd = np.zeros((size[0],size[0]*size[1]))
        l = np.array(kwidth)
        size = np.shape(m1)
        invkwidthsq = l**(-2)
        for i in range(len(m1)):
            for j in range(len(m1)):
                d = m1[i]-m1[j]
                k_gd = invkwidthsq * d * k[i][j]
                big_kgd[i:i+1,j*size[1]:(j+1)*size[1]] = k_gd
        # print(big_kgd)

# method 6 comes from method 2.
    if method=='6':
        size = np.shape(m1)
        big_kgd = np.zeros((size[0],size[0]*size[1]))
        invsqkwidth = 1/kwidth**2
        k1 = distance.pdist(m1 / kwidth, metric='sqeuclidean')
        k1 = distance.squareform(np.exp(-.5 * k1))
        np.fill_diagonal(k1, 1)
        # print(k1)
        for i in range(size[0]):
            ldist = (invsqkwidth * (m1[:,:]-m1[i,:]))
            big_kgd_i = ((ldist).T * k1[i]).T
            big_kgd[:,(size[1]*i):(size[1]+size[1]*i)] = big_kgd_i
        # print(big_kgd)

# method 7 comes from method 5.
    if method=='7':
        # m1 = np.array([[0.0,1.5],[1.0,1.0],[2.0,1.0]])
        # kwidth = [2.0,2.0]
        size = np.shape(m1)
        big_kgd = np.zeros((size[0],size[0]*size[1]))
        big_kdd = np.zeros((size[0]*size[1],size[0]*size[1]))
        k = distance.pdist(m1 / kwidth, metric='sqeuclidean')
        k = distance.squareform(np.exp(-.5 * k))
        np.fill_diagonal(k, 1)
        l = np.array(kwidth)
        invkwidthsq = l**(-2)
        I_m = np.identity(size[1])*invkwidthsq
        for i in range(size[0]):
            ldist = (invkwidthsq * (m1[:,:]-m1[i,:]))
            k_gd = ldist*k[i,:].reshape(size[0],1)
            big_kgd[:,size[1]*i:size[1]+size[1]*i] = k_gd
            # for j in range(i,size[0]):
            #     k_dd = (I_m-np.outer(ldist[j],ldist[j].T))*k[i,j]
            #     big_kdd[i*size[1]:(i+1)*size[1],j*size[1]:(j+1)*size[1]] = k_dd
            #     if j!=i:
            #         big_kdd[j*size[1]:(j+1)*size[1],i*size[1]:(i+1)*size[1]]= k_dd.T
        # print(big_kdd)

# Method 8 same as 7 but using broadcast for bigkdd
    if method=='8':
        # m1 = np.array([[0.0,1.5],[1.0,1.0],[2.0,1.0]])
        # kwidth = [2.0,2.0]
        size = np.shape(m1)
        big_kgd = np.zeros((size[0],size[0]*size[1]))
        big_kdd = np.zeros((size[0]*size[1],size[0]*size[1]))
        k = distance.pdist(m1 / kwidth, metric='sqeuclidean')
        k = distance.squareform(np.exp(-.5 * k))
        np.fill_diagonal(k, 1)
        l = np.array(kwidth)
        invkwidthsq = l**(-2)
        I_m = np.identity(size[1])
        for i in range(size[0]):
            ldist = (invkwidthsq * (m1[:,:]-m1[i,:]))
            k_gd = ldist*k[i,:].reshape(size[0],1)
            big_kgd[:,size[1]*i:size[1]+size[1]*i] = k_gd
            # k_dd = ((I_m*invkwidthsq - (ldist[:,None,:]*ldist[:,:,None]))*(
            # k[i,None,None].T)).reshape(-1,size[1])
            # big_kdd[:,size[1]*i:size[1]+size[1]*i] = k_dd



################################# TILDE #############

    if method=='20':
        # m1 = np.array([[0.0,1.5],[1.0,1.0],[2.0,1.0]])
        # m2 = np.array([[1.1,2.3],[0.3,0.4]])
        k = distance.cdist(m1 / kwidth, m2 / kwidth, metric='sqeuclidean')
        k = np.exp(-.5 * k)
        size_m1 = np.shape(m1)
        size_m2 = np.shape(m2)
        kgd_tilde = np.zeros((size_m1[0], size_m2[0] * size_m2[1]))
        invsqkwidth = kwidth**(-2)
        for i in range(size_m1[0]):
            kgd_tilde_i = -((invsqkwidth * (m2[:,:]-m1[i,:])* k[i,
            :].reshape(size_m2[0],1)).reshape(1,size_m2[0]*size_m2[1]))
            kgd_tilde[i,:] = kgd_tilde_i
        k = np.block([k, kgd_tilde])


    end = timer()
    time_one_loop = end-start
    time.append(time_one_loop)
    print('Elapsed time iteration',loop_number,time_one_loop)


print('Average time', np.average(time))
print('Best', np.min(time))









