import numpy as np
from atoml.regression.gpfunctions.gkernels import gkernels as gkernels
from timeit import default_timer as timer
from scipy.spatial import distance

# Test the performance of the construction of Kdd (train/train).
# Returns elapsed time for each iteration, average and fastest iteration times.

method = 'i' # Methods 1,2,3,4 and i (combination of methods 2 and 3).
train_points = 1000 # Number of training points
dimensions = 10 # Dimensions of the training points
iterations = 1 # Number of iterations for average time.


np.random.seed(1) # Random seed.
m1 = []
m1= 1.2*np.random.randint(5.0, size=(train_points,dimensions)) # Train
kwidth = np.zeros(np.shape(m1)[1])+2.0 # Length scale.

time=[]

for loop in range(0,iterations):
    start=timer()
    loop_number = loop

    if method=='1':
         k1 = distance.pdist(m1 / kwidth, metric='sqeuclidean')
         k1 = distance.squareform(np.exp(-.5 * k1))
         np.fill_diagonal(k1, 1)
         # print(k1)
         big_kdd = gkernels.big_kdd('squared_exponential',kwidth,m1)
         # print(big_kdd)

    if method=='2':
        size = np.shape(m1)
        big_kdd = np.zeros((size[0]*size[1],size[0]*size[1]))
        k = distance.pdist(m1 / kwidth, metric='sqeuclidean')
        k = distance.squareform(np.exp(-.5 * k))
        np.fill_diagonal(k, 1)
        l = np.array(kwidth)
        invkwidthsq = l**(-2)
        I_m = np.identity(size[1])
        for i in range(size[0]):
            ldist = (invkwidthsq * (m1[:,:]-m1[i,:]))
            for j in range(i,size[0]):
                k_dd = (I_m*invkwidthsq-np.outer(ldist[j],ldist[j].T))*k[i,j]
                big_kdd[i*size[1]:(i+1)*size[1],j*size[1]:(j+1)*size[1]] = k_dd
                if j!=i:
                    big_kdd[j*size[1]:(j+1)*size[1],i*size[1]:(i+1)*size[1]]= k_dd.T
        # print(big_kdd)


    if method=='3':
        size = np.shape(m1)
        big_kdd = np.zeros((size[0]*size[1],size[0]*size[1]))
        k = distance.pdist(m1 / kwidth, metric='sqeuclidean')
        k = distance.squareform(np.exp(-.5 * k))
        np.fill_diagonal(k, 1)
        l = np.array(kwidth)
        invkwidthsq = l**(-2)
        I_m = np.identity(size[1])*invkwidthsq
        for i in range(size[0]):
            ldist = (invkwidthsq * (m1[:,:]-m1[i,:]))
            k_dd = ((I_m - (ldist[:,None,:]*ldist[:,:,None]))*(k[i,None,None].T)).reshape(-1,size[1])
            big_kdd[:,size[1]*i:size[1]+size[1]*i] = k_dd
        # print(big_kdd)


    if method=='4':
        size = np.shape(m1)
        big_kdd = np.zeros((size[0]*size[1],size[0]*size[1]))
        k = distance.pdist(m1 / kwidth, metric='sqeuclidean')
        k = distance.squareform(np.exp(-.5 * k))
        np.fill_diagonal(k, 1)
        l = np.array(kwidth)
        invkwidthsq = l**(-2)
        I_m = np.identity(size[1])*invkwidthsq
        for i in range(size[0]):
            ldist = (invkwidthsq * (m1[:,:]-m1[i,:]))
            k_dd = ((I_m - (np.einsum('ij,ik->ijk',ldist,ldist)))*(k[i,None,
            None].T)).reshape(-1,size[1])
            big_kdd[:,size[1]*i:size[1]+size[1]*i] = k_dd
        # print(big_kdd)


    if method=='i':
        k = distance.pdist(m1 / kwidth, metric='sqeuclidean')
        k = distance.squareform(np.exp(-.5 * k))
        np.fill_diagonal(k, 1)
        size = np.shape(m1)
        big_kgd = np.zeros((size[0], size[0] * size[1]))
        big_kdd = np.zeros((size[0] * size[1], size[0] * size[1]))
        invsqkwidth = kwidth**(-2)
        I_m = np.identity(size[1]) * invsqkwidth
        for i in range(size[0]):
            ldist = (invsqkwidth * (m1[:,:]-m1[i,:]))
            # big_kgd_i = ((ldist).T * k[i]).T
            # big_kgd[:,(size[1]*i):(size[1]+size[1]*i)] = big_kgd_i
            if size[1]<=30: # (Method 3) Broadcasting requires large memory.
                k_dd = ((I_m - (ldist[:,None,:]*ldist[:,:,None]))*(k[i,None,None].T)).reshape(-1,size[1])
                big_kdd[:,size[1]*i:size[1]+size[1]*i] = k_dd
            if size[1]>30: # (Method 2) Loop when the number of features is
            # large.
                for j in range(i, size[0]):
                    k_dd = (I_m - np.outer(ldist[j], ldist[j].T)) * k[i,j]
                    big_kdd[i*size[1]:(i+1)*size[1],j*size[1]:(j+1)*size[1]] = k_dd
                    if j!=i:
                        big_kdd[j*size[1]:(j+1)*size[1],i*size[1]:(i+1)*size[1]]= k_dd.T


    end = timer()
    time_one_loop = end-start
    time.append(time_one_loop)
    print('Elapsed time iteration',loop_number,time_one_loop)


print('Average time', np.average(time))
print('Best', np.min(time))









