import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(.0001, 1, 200)
#print(x)

logx = np.log(x)
#print(logx)

fig = plt.figure(figsize=(10,4))
plt.rcParams.update({'font.size':15})
plt.plot(x, logx, 'ks-', markerfacecolor='w')
plt.xlabel('x')
plt.ylabel('log(x)')
plt.savefig('log')


x = np.linspace(.0001, 1, 20)
logx = np.log(x)
expx = np.exp(x)

fig = plt.figure(figsize=(10,4))
plt.plot(x, x, color=[.8,.8,.8])
plt.plot(x, np.exp(logx), 'o', markersize=8)
plt.plot(x, np.log(expx), 'x', markersize=8)
plt.xlabel('x')
plt.ylabel('f(g(x))')
plt.savefig('log2')
