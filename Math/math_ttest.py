import numpy as np
import matplotlib.pyplot as plt

import scipy.stats as stats

n1 = 30 # samples in dataset 1
n2 = 40 # samples in dataset 2
mu1 = 1 # population mean in dataset 1
mu2 = 2 # population mean in dataset 2

data1 = mu1 + np.random.randn(n1)
data2 = mu2 + np.random.randn(n2)

#print(f"data1={data1}")
#print(f"data2={data2}")

fig = plt.figure(figsize=(10,4))
plt.hist(data1, bins=10, density=True, alpha=.8, edgecolor='red')
plt.hist(data2, bins=10, density=True, alpha=0.7, edgecolor='yellow')
plt.plot([mu1,mu1],[0,.3], 'r--')
plt.plot([mu2,mu2],[0,.3], 'y-')
plt.savefig('t-test')

t,p = stats.ttest_ind(data1,data2)
print(f"t={t} p={p}")

fig = plt.figure(figsize=(10,4))
plt.rcParams.update({'font.size':12})
plt.plot(0+np.random.randn(n1)/15, data1, 'ro', markerfacecolor='w', markersize=14)
plt.plot(1+np.random.randn(n2)/15, data2, 'bs', markerfacecolor='w', markersize=14)
plt.xlim([-1,2])
plt.xticks([0,1], labels=['Group1', 'Group2'])
plt.title(f"t = {t:.2f}, p={p:.3f}")
plt.savefig('t-test2')

