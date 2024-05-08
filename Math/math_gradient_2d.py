import numpy as np
import sympy as sym
import matplotlib.pyplot as plt

# function to return values - it represents our model
def peaks(x,y):
    x,y = np.meshgrid(x,y)

    z = 3*(1-x)**2 * np.exp(-(x**2) - (y+1)**2) \
        - 10*(x/5 - x**3 - y**5) * np.exp(-x**2-y**2) \
        -1/3*np.exp(-(x+1)**2 - y**2)
    return z

# create the landscape
x = np.linspace(-3,3,201)
y = np.linspace(-3,3,201)

Z = peaks(x,y)

plt.imshow(Z, extent=[x[0],x[-1],y[0],y[-1]], vmin=-5, vmax=5, origin='lower')
plt.savefig('gradient_2d')

# now create function as sympy function
sx,sy = sym.symbols('sx,sy')
sZ = 3*(1-sx)**2 * sym.exp(-(sx**2) - (sy+1)**2) \
     - 10*(sx/5 - sx**3 - sy**5) * sym.exp(-sx**2-sy**2) \
     - 1/3*sym.exp(-(sx+1)**2 - sy**2)

# partial derivatives
df_x = sym.lambdify( (sx,sy), sym.diff(sZ,sx), 'sympy')
df_y = sym.lambdify( (sx,sy), sym.diff(sZ,sy), 'sympy')

print(f'Z\'(1,1)={df_x(1,1).evalf()}')

# random starting point
### now try to find the min
#localmin = np.random.rand(2)*4-2
localmin = np.array([np.random.choice(x,1)[0], np.random.choice(y,1)[0]])
startpnt = localmin[:] # copy not reassignment

# learning params
learning_rate = .01
training_epochs = 1000

print(f'Starting with {localmin}')

trajectory = np.zeros((training_epochs,2))
for i in range(training_epochs):
    grad = np.array( [df_x(localmin[0],localmin[1]).evalf(),
                      df_y(localmin[0],localmin[1]).evalf(),
                      ])
    localmin = localmin - learning_rate*grad
    trajectory[i,0] = localmin[0]
    trajectory[i,1] = localmin[1]

print(f'Ended with {localmin}')

plt.imshow(Z, extent=[x[0],x[-1],y[0],y[-1]], vmin=-5, vmax=5, origin='lower')
plt.plot(startpnt[0], startpnt[1], 'bs')
plt.plot(localmin[0], localmin[1], 'ro')
plt.plot(trajectory[:,0], trajectory[:,1], 'r')
plt.legend('rnd start', 'local min')
plt.colorbar()
plt.savefig('gradient_2d_trjectory')
