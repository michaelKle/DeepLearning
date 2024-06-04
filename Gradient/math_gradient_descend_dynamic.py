import numpy as np
import matplotlib.pyplot as plt

# function to return values - it represents our model
def fx(x):
    return 3*x**2 - 3*x + 4

# derivative
def deriv(x):
    return 6*x - 3

# input values - from -2 to 2
x = np.linspace(-2,2,2001)

def const_learning_rate(starting_learning_rate, gradient, current_epoch, num_epochs, current_local_min):
    # standard model
    return starting_learning_rate

def gradient_based_learning_rate(starting_learning_rate, gradient, current_epoch, num_epochs, current_local_min):
    # in effect we now have gradient squared
    return starting_learning_rate * abs(gradient)

def epoch_based_learning_rate(starting_learning_rate, gradient, current_epoch, num_epochs, current_local_min):
    # could als use the max number of epochs to go from 1 to 0 in linear steps
    return starting_learning_rate * (1.0 - current_epoch/num_epochs)


def do_gradient_descend(starting_min, training_epochs, starting_learning_rate, learning_rate_function):
    localmin = starting_min
    modelparams = np.zeros((training_epochs,3))
    for i in range(training_epochs):
        grad = deriv(localmin)
        learning_rate = learning_rate_function(starting_learning_rate, grad, i, training_epochs, localmin)
        localmin = localmin - learning_rate * grad
        modelparams[i,0] = localmin[0]
        modelparams[i,1] = grad[0]
        modelparams[i,2] = learning_rate[0]

    return modelparams


# learning params
starting_min = np.random.choice(x,1)
learning_rate = np.array([.01])
training_epochs = 100

modelparams_const = do_gradient_descend(starting_min, training_epochs=training_epochs, starting_learning_rate=learning_rate, learning_rate_function=const_learning_rate)
modelparams_grad = do_gradient_descend(starting_min, training_epochs=training_epochs, starting_learning_rate=learning_rate, learning_rate_function=gradient_based_learning_rate)
modelparams_epoch = do_gradient_descend(starting_min, training_epochs=training_epochs, starting_learning_rate=learning_rate, learning_rate_function=epoch_based_learning_rate)


plt.figure()
plt.plot(modelparams_const[:,0],'o-', markerfacecolor='w')
plt.plot(modelparams_grad[:,0],'o-', markerfacecolor='w')
plt.plot(modelparams_epoch[:,0],'o-', markerfacecolor='w')
plt.xlabel('epoch')
plt.ylabel('localmin')
plt.legend(['const', 'gradient', 'epoch'])
plt.savefig('gradient_1d_dynamic_localmin')

plt.figure()
plt.plot(modelparams_const[:,1],'o-', markerfacecolor='w')
plt.plot(modelparams_grad[:,1],'o-', markerfacecolor='w')
plt.plot(modelparams_epoch[:,1],'o-', markerfacecolor='w')
plt.xlabel('epoch')
plt.ylabel('gradient')
plt.legend(['const', 'gradient', 'epoch'])
plt.savefig('gradient_1d_dynamic_gradient')

plt.figure()
plt.plot(modelparams_const[:,2],'o-', markerfacecolor='w')
plt.plot(modelparams_grad[:,2],'o-', markerfacecolor='w')
plt.plot(modelparams_epoch[:,2],'o-', markerfacecolor='w')
plt.xlabel('epoch')
plt.ylabel('learning rate')
plt.legend(['const', 'gradient', 'epoch'])
plt.savefig('gradient_1d_dynamic_rate')