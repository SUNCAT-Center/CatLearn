import numpy as np

# Linear kernel for k:
def kernel_k(l, xm, xn):
    k = l + np.outer(xm,xn)
    return k


# Linear kernel for kgd:
def kernel_kgd(l, xm, xn):
    kgd = xm
    return kgd


# Linear kernel for kdd:
def kernel_kdd(l, xm, xn):
    kdd = 1.0
    return kdd


###############################################
######### k little and kgd_tilde   ############
###############################################

# To build k_little:
def k_little(l,train,test):
    l = np.array(l)
    size1 = np.shape(train)
    size2 = np.shape(test)
    k_little = np.zeros((size1[0],size2[0]))
    for i in range(len(train)):
        for j in range(len(test)):
            k_little[i][j] = kernel_k(l,test[j],train[i])
    return k_little

# To build kgd_tilde:
def kgd_tilde(l, train, test):
    l = np.array(l)
    size1 = np.shape(train)
    size2 = np.shape(test)
    kgd = np.zeros((size1[0]*size1[1],size2[0]))
    for j in range(len(test)):
        kgd_thistest = np.zeros(size1[0]*size1[1])
        for i in range(len(train)):
            kgd_thistest[i*size1[1]:(i+1)*size1[1]] = kernel_kgd(
            l,test[j],train[i])
        kgd[:,j] = kgd_thistest
    return kgd


#########################################
############    bigKs    ################
#########################################

# To build big_k:
def bigk(l, train):
    l = np.array(l)
    size = np.shape(train)
    bigk = np.zeros((size[0],size[0]))
    for i in range(len(train)):
        for j in range(len(train)):
            bigk[i][j] = kernel_k(l,train[i],train[j])
    return bigk


# To build big_kgd:
def big_kgd(l,train):
    l = np.array(l)
    size = np.shape(train)
    big_kgd = np.zeros((size[0],size[0]*size[1]))
    for i in range(len(train)):
        basis_vect1 = np.zeros(size[0])
        basis_vect1[i] = 1.0
        for j in range(len(train)):
            basis_vect2 = np.zeros(size[0])
            basis_vect2[j] = 1.0
            k_gd = kernel_kgd(l, train[i],train[j])
            prekron = np.outer(basis_vect1,np.transpose(basis_vect2))
            kron = np.kron(prekron,k_gd)
            big_kgd = big_kgd + kron
    return big_kgd


# To build big_kdd:
def big_kdd(l, train):
    l = np.array(l)
    size = np.shape(train)
    big_kdd = np.zeros((size[0]*size[1],size[0]*size[1]))
    for i in range(len(train)):
        basis_vect1 = np.zeros(size[0])
        basis_vect1[i] = 1.0
        for j in range(len(train)):
            basis_vect2 = np.zeros(size[0])
            basis_vect2[j] = 1.0
            k_dd = kernel_kdd(l, train[i],train[j])
            prekron = np.outer(basis_vect1,np.transpose(basis_vect2))
            kron = np.kron(prekron,k_dd)
            big_kdd = big_kdd + kron
    return big_kdd


#########################################
#######  k TILDE and bigK TILDE    ######
#########################################

# To build k_tilde:
def k_tilde(l,test,train):
    l = np.array(l)
    k_tilde = []
    k_tilde = np.vstack([k_little(l,train,test),kgd_tilde(l,
    train,test)])
    return k_tilde.T

# To build bigK_tilde:
def bigk_tilde(l,train):
    l = np.array(l)
    bigk_tilde = []
    bigk_tilde = np.block([[bigk(l,train),big_kgd(l,
    train)],
    [np.transpose(
    big_kgd(l,train)),big_kdd(l,train)]])
    return bigk_tilde

#########################################
#########################################