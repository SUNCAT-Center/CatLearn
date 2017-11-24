import numpy as np
from atoml.regression.gpfunctions.gkernels import gkernels as gkernels
from timeit import default_timer as timer
from scipy.spatial import distance

# TEST METHODS FOR BIG KDD

#T500/D100,    1=31.18     2=26.15      3=37.50     4=42.20
#T100/D100,    1=0.819     2=0.819      3=1.12      4=1.16
#T10/D1000,    1=1.31      2=1.146      3=1.80      4=1.77
#T50/D1000,    1=37.92     2=36.31      3=77.79     4=78.41
#T1000/D10,    1=23.69     2=6.54       3=0.99      4=1.06
#T1000/D50,    1=53.84     2=33.05      3=41.39     4=39.83
#T10000/D2,    1=NA        2=546.0      3=12.44     4=12.34

method = '1'
np.random.seed(1)
m1 = []
train_points = 10000
dimensions = 2
iterations = 5

m1= 1.2*np.random.randint(5.0, size=(train_points,
dimensions))


kwidth = np.zeros(np.shape(m1)[1])+2.0
time=[]


for i in range(0,iterations):
    start=timer()
    loop_number = i



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
        # m1 = np.array([[0.0,1.5],[1.0,1.0],[2.0,1.0]])
        # kwidth = [2.0,2.0]
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

    end = timer()
    time_one_loop = end-start
    time.append(time_one_loop)
    print('Elapsed time iteration',loop_number,time_one_loop)


print('Average time', np.average(time))
print('Best', np.min(time))









