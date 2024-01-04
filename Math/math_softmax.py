import matplotlib.pyplot as plt
import numpy as np

# manual calc of simple vector
z = [1,2,3]
num = np.exp(z)
print(f"exp(z)={num}")
den = np.sum(np.exp(z))
print(f"sum(exp(z))={den}")
sigma = num / den
print(f"sigma=exp(z)/sum(exp(z))={sigma}")
print(f"sum(sigma)=sum(exp(z)/sum(exp(z)))={sum(sigma)}")

def softmax(z):
    num = np.exp(z)
    return num / np.sum(num)

# random
z = np.random.randint(-5,high=15, size=25)
print(f"z={z}")
sigma = softmax(z)
print(f"sigma(z)={sigma}")
print(f"sum(sigma(z))={sum(softmax(z))}")


## graph it
plt.plot(z, sigma, 'ko')
plt.xlabel('Original number (z)')
plt.ylabel('Softmaxified $\sigma$')
#plt.yscale('log')
plt.title(f'$\sum\sigma$ = {np.sum(sigma)}')
plt.savefig('sigma')


import torch
import torch.nn as nn

# create a function 
softfun = nn.Softmax(dim=0)
z = [1,2,3]
sigmaT = softfun(torch.Tensor(z))
print(f'softfun(z)={sigmaT}')