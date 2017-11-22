import numpy as np
from atoml.regression.gpfunctions.gkernels import gkernels as gkernels

method = 'forloops'
np.random.seed(1)
m1 = []
train_points = 3
dimensions = 2
m1= 1.2*np.random.randint(5.0, size=(train_points,
dimensions))
kwidth = np.zeros(np.shape(m1)[1])+2.0


if method=='broadcast':
    size = np.shape(m1)
    d = (m1[np.newaxis,:,:] - m1[:,np.newaxis,:])
    invsqkwidth = 1/kwidth**2
    sqd = np.sqrt(np.sum(invsqkwidth*d**2,axis=-1))
    k1 = np.exp(-(sqd)**2/2)
    big_kgd = -(invsqkwidth*d*k1[:,:,np.newaxis]).reshape(size[0],
    size[0]*size[1])

if method=='forloops':
     big_kgd = gkernels.big_kgd('squared_exponential',kwidth,m1)

else:
    exit()