import numpy as np

population = [1,2,4,6,5,4,0,-4,5,-2,6,10,-9,1,3,-6]
n = len(population)

# population mean
popmean = np.mean(population)
print(f"Poplation mean={popmean}")

# single sample
sample = np.random.choice(population, size=5, replace=True)
samplemean = np.mean(sample)
print(f"Sample mean={samplemean}")


# lots of samples
nExers = 10000

sampleMeans = np.zeros(nExers)
for i in range(nExers):
    sample = np.random.choice(population, size=5, replace=True)
    sampleMeans[i] = np.mean(sample)

import matplotlib.pyplot as plt

fig = plt.figure(figsize=(10,4))
plt.hist(sampleMeans, bins=40, density=True)
plt.plot([popmean,popmean],[0,.3], 'm--')
plt.savefig('sample')