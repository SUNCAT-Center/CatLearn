import numpy as np
from atoml.regression.gpfunctions.gkernels import gkernels as gkernels
from timeit import default_timer as timer
from scipy.spatial import distance


method = '1'
train_points = 200
test_points = 1
dimensions = 200
iterations = 1

np.random.seed(1)
m1 = []
m2 = []
m1= 1.2*np.random.randint(5.0, size=(train_points,
dimensions))
m2= 2.3*np.random.randint(6.0, size=(test_points,
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


# Method 9 same as 7 but using broadcast for bigkdd
    if method=='9':
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


################################# TILDE ###################################

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



    if method=='21':
        k = distance.cdist(m1 / kwidth, m2 / kwidth, metric='sqeuclidean')
        k = np.exp(-.5 * k)
        size_m1 = np.shape(m1)
        size_m2 = np.shape(m2)
        d = (m1[np.newaxis,:,:] - m2[:,np.newaxis,:])
        invsqkwidth = 1/kwidth**2
        sqd = np.sqrt(np.sum(invsqkwidth*d**2,axis=-1))
        k1 = np.exp(-(sqd)**2/2)
        kgd_tilde = ((invsqkwidth*d*k1[:,:,np.newaxis]).swapaxes(size_m1[1],
        1).reshape(size_m2[0]*size_m2[1],size_m1[0])).T




    end = timer()
    time_one_loop = end-start
    time.append(time_one_loop)
    print('Elapsed time iteration',loop_number,time_one_loop)


print('Average time', np.average(time))
print('Best', np.min(time))









