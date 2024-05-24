import numpy as np
import matplotlib.pyplot as plt

# function to return values - it represents our model
def fx(x):
    return 3*x**2 - 3*x + 4

# derivative
def deriv(x):
    return 6*x - 3

## x inputs
x = np.linspace(-2,2,2001)

### now try to find the min
localmin = np.random.choice(x,1)

# learning params
learning_rate = .01
training_epochs = 100

print(f'Starting with {localmin}')

for i in range(training_epochs):
    grad = deriv(localmin)
    localmin = localmin - learning_rate*grad
    #print(f'i={i:03} localmin={localmin}')

print(f'Ended with {localmin}')

# plot it
plt.plot(x,fx(x), x,deriv(x))
plt.plot(localmin, deriv(localmin), 'ro')
plt.plot(localmin, fx(localmin), 'ro')
plt.xlim(x[[0,-1]])
plt.grid()
plt.xlabel('x')
plt.ylabel('y')
plt.legend(['y', 'dy'])
#plt.show()
plt.savefig('gradient_1d')

# store all the parameters while we descend towards minimum
modelparams = np.zeros((training_epochs,2))
for i in range(training_epochs):
    grad = deriv(localmin)
    localmin = localmin - learning_rate*grad
    modelparams[i,0] = localmin[0]
    modelparams[i,1] = grad[0]

# plot
fig,ax = plt.subplots(1,2,figsize=(12,4))
for i in range(2):
    ax[i].plot(modelparams[:,i],'o-')
    ax[i].set_xlabel('Iteration')
    ax[i].set_title(f'Final estimated min: {localmin[0]:.5f}')

ax[0].set_ylabel('Local minimum')
ax[1].set_ylabel('Derivative')
plt.savefig('gradient_1d_params')
