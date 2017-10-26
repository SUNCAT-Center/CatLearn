import numpy as np
import matplotlib.pyplot as plt
import gradient_kernels as gkernels

gradients = True
gradients_mode = 'Analytical' # This is the only mode available for 2D plots.
train_points = 6 # Number of training points.
test_points = 50 # Length of the grid test points (nxn).


l = np.array([3.0,3.0]) # Length-scale parameters.
sigma = 0.0001 # Noise parameter.
scaling = 1.0

###############################
#### SET TRAINING POINTS ###
###############################

# Random
# train = np.random.uniform(low=-6.0, high=6.0, size=(train_points,2))

# Equaly spaced
train = []
trainx = np.linspace(-5.0, 5.0, train_points)
trainy = np.linspace(-5.0, 5.0, train_points)
for i in range(len(trainx)):
    for j in range(len(trainy)):
        train1 = [trainx[i],trainy[j]]
        train.append(train1)

train = np.array(train)


# Append more points
train = np.append(train,([[0.0,0.0]]),axis=0)

# Sort in 2d
# train = train[::, train[0,].argsort()[::-1]]


###############################
#### SET TEST POINTS #######
###############################

test = []
testx = np.linspace(-5.0, 5.0, test_points)
testy = np.linspace(-5.0, 5.0, test_points)
for i in range(len(testx)):
    for j in range(len(testy)):
        test1 = [testx[i],testy[j]]
        test.append(test1)

test = np.array(test)


###############################
#### SET REAL FUNCTION #######
###############################

def afunc(x,y):
    """S FUNCTION"""
    z = -(12.0)*(x**2.0) + (1.0/3.0)*(x**4.0)
    z = z -(12.0)*(y**2.0) + (1.0/2.0)*(y**4.0)
    return z

def first_derivative(x,y):
    dx = -24.0*x + (4.0/3.0)*x**3.0
    dy = -24.0*y + 2.0*y**3.0
    return [dx,dy]


###############################
#### CALCULATE TARGETS  #######
###############################

target = []
for i in train:
    target.append([afunc(i[0],i[1])])
target = np.array(target)
targetplot = target


######################################################################
##### BUILD MATRICES #################################################
######################################################################

k_little = gkernels.k_little(scaling, l,train,test)
if gradients == True:
    k_little = gkernels.k_tilde(scaling, l,train,test)

bigK = gkernels.bigk(scaling, l,train)
if gradients == True:
    bigK = gkernels.bigk_tilde(scaling, l,train)

if gradients == True:
    if gradients_mode == 'Analytical':
        gradients = []
        for i in train:
            gradients.append(first_derivative(i[0],i[1]))
        deriv = []
        for i in range(len(gradients)):
            for d in range(np.shape(gradients)[1]):
                deriv.append(gradients[i][d])
        gradients = deriv
    y_tilde = []
    # Build y_tilde as:
    # y_tilde = y1, y2...yN, delta1, delta2...deltaN
    y_tilde = np.append(target, gradients)
    y_tilde = np.reshape(y_tilde,(np.shape(y_tilde)[0],1))
    target = y_tilde


######################################################################
# Posterior mean:
# klittle.T • ( bigK + sigma**2 * I)^-1 • targets
######################################################################

ktb = np.dot(k_little.T,np.linalg.inv(bigK + sigma**2 * np.identity(np.shape(
bigK)[0])))
pred = np.dot(ktb,target)
pred = pred


######################################################################
# Predictive GP variance:
# k_xx - (klittle.T • ( bigK + sigma**2 * I)^-1 • k_little)
######################################################################
#
# k_xx = gkernels.k_little(scaling, l,test,test)
# var = k_xx - np.dot(ktb,k_little)
#
# # I need to print the diagonal
# var2 = np.zeros((np.shape(var)[0],1))
# for i in range(len(var)):
#    var2[i] = var[i][i]
#
# upper = var2 + sigma**2
# lower = var2 - sigma**2

###############################
######### PLOTS ###############
###############################

plt.figure(figsize=(12.0, 5.0))

######################################################
# Contour plot for real function.
######################################################

plt.subplot(131)
x = np.linspace(-5.0, 5.0, test_points)
y = np.linspace(-5.0, 5.0, test_points)
X,Y = np.meshgrid(x, y)
plt.contourf(X, Y, afunc(X, Y), 6, alpha=.70, cmap='PRGn',vmin=np.min(afunc(
X,Y)),
vmax=np.max(afunc(X,Y)))
plt.colorbar(orientation="horizontal", pad=0.1)
plt.clim(np.min(afunc(
X,Y)),np.max(afunc(
X,Y)))
C = plt.contour(X, Y, afunc(X, Y), 6, colors='black', linewidths=1)
plt.clabel(C, inline=1, fontsize=9)
plt.title('Real function',fontsize=10)

######################################################
# Contour plot for predicted function.
######################################################

plt.subplot(132)

x = []
for i in range(len(test)):
    t = test[i][0]
    t = x.append(t)
y = []
for i in range(len(test)):
    t = test[i][1]
    t = y.append(t)

zi = plt.mlab.griddata(x, y, pred[:,0], testx, testy, interp='linear')

plt.contourf(testx, testy, zi, 6, alpha=.70, cmap='PRGn',vmin=np.min(afunc(
X,Y)),
vmax=np.max(afunc(X,Y)))
plt.colorbar(orientation="horizontal", pad=0.1)
C = plt.contour(testx, testy, zi, 6, colors='k',linewidths=1.0)
plt.clabel(C, inline=0.1, fontsize=9)


# Print training points positions
plt.scatter(train[:,0],train[:,1],marker='o',s=5.0,c='black',
edgecolors='black',alpha=0.8)
plt.xlim(-5,5)
plt.ylim(-5,5)
plt.title('Predicted function',fontsize=10)

######################################################
# Difference
######################################################
plt.subplot(133)

diff = afunc(X,Y)-zi
diff = np.absolute(diff)

plt.contourf(testx, testy, diff, 6, alpha=.70, cmap='Reds',vmin=0.0,
vmax=5.0)
plt.colorbar(orientation="horizontal", pad=0.1)
C = plt.contour(testx, testy, diff, 6,colors='k',linewidths=1.0)
plt.clabel(C, inline=0.1, fontsize=9)
plt.title('Difference',fontsize=10)


plt.show()
