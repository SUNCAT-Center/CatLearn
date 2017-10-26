import numpy as np
import matplotlib.pyplot as plt
import gradient_kernels as gkernels

gradients = True
gradients_mode = 'Analytical' # Select: Central, Finite, Analytical

train = np.array([[1.5],[3.0],[4.0],[4.5]])
# train = np.append(train,([5.0]))
train = np.reshape(train,(np.shape(train)[0],1))
train = np.sort(train)

test = np.linspace(1,5,100)
test = np.reshape(test,(np.shape(test)[0],1))

# test = np.array([[3.0]])
test = np.sort(test)

scaling = 1.90581728
l = np.array([1.96519344])
sigma = 0.15673437



def afunc(x):
    """S FUNCTION"""
    y = 20*((x-3)**(2))*(x-3) + 19
    #y = (x-3)**(2) + 1
    return y

def first_derivative(x):
    dydx =  60*(x-3)**2
    return dydx


linex = []
linex = np.linspace(1.0,5.0,1000)
linex = np.sort(linex)
linex = np.array(linex)


liney = []
for i in linex:
    liney.append(afunc(i))

target = []
for i in train:
    target.append(afunc(i))
target = np.array(target)
targetplot = target



######################################################################
##### BUILD MATRICES #################################################
######################################################################

k_little = gkernels.k_little(scaling,l,train,test)
if gradients == True:
    k_little = gkernels.k_tilde(scaling, l,train,test)

bigK = gkernels.bigk(scaling, l,train)
if gradients == True:
    bigK = gkernels.bigk_tilde(scaling, l,train)

if gradients == True:
   gradients_plot = True
if gradients == False:
   gradients_plot = False

if gradients == True:
    if gradients_mode == 'Analytical':
        gradients = []
        for i in train:
            gradients.append(first_derivative(i))
    if gradients_mode == 'Finite':
        delta = np.abs((train[0]-train[np.shape(train)[0]-1]) / 4.0)
        gradients = []
        for i in range(0,np.shape(train)[0]):
            gradients_finite = afunc(train[i]+delta)-afunc(train[i]-delta)
            gradients_finite = gradients_finite / 2*delta
            gradients.append(gradients_finite)
    if gradients_mode == 'Central':
        gradients = np.gradient(target[:,0],train[:,0])
    y_tilde = []
    # 1) Build y_tilde as:
    # y_tilde = y1, y2...yN, delta1, delta2...deltaN
    y_tilde = np.append(target, gradients)
    y_tilde = np.reshape(y_tilde,(np.shape(y_tilde)[0],1))
    target = y_tilde
    # # 2) Build y_tilde as:
    # # y_tilde = y1,delta1,y2,delta2...yN,deltaN
    # for i in range(0,np.shape(target)[0]):
    #     y_tilde.append(target[i][0])
    #     y_tilde.append(gradients[i])
    # y_tilde = np.array(y_tilde)
    # y_tilde = np.reshape(y_tilde,(np.shape(y_tilde)[0],1))
    # target = y_tilde

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

k_xx = gkernels.k_little(scaling, l,test,test)
var = k_xx - np.dot(ktb,k_little)

var2 = np.zeros((np.shape(var)[0],1))
for i in range(len(var)):
   var2[i] = var[i][i]

error = (var2 + sigma**2)



# Plot

fig = plt.figure(figsize=(5.0, 5.0))
ax = fig.add_subplot(111)
ax.plot(linex, liney, '--', lw=2, color='black')
ax.plot(train, targetplot, 'o', alpha=1.0, color='black')
ax.plot(test, pred, '-', alpha=1.0, color='red')


# Plot gradients:
size_bar_gradients = (np.abs(np.max(linex) - np.min(linex)) / (2.0))/20.0

if gradients_plot == True:
    for i in range(0,np.shape(gradients)[0]):
        def lineary(m,linearx,train,target):
            """Define some linear function."""
            lineary = m*(linearx-train)+target
            return lineary
        linearx_i = np.linspace(train[i]-size_bar_gradients, train[
 i]+size_bar_gradients,
num=10)
        lineary_i = lineary(gradients[i],linearx_i,train[i],targetplot[i])
        ax.plot(linearx_i, lineary_i, '-', lw=3, alpha=1, color='black')

# End plot gradients.

plt.show()





