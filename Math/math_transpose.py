import numpy as np
import torch

nv = np.array([[1,2,3,4]])
print(nv), print(' ')

print (nv.T), print(' ')

nvT = nv.T
print (nvT.T), print(' ')


print('#####')

nM = np.array([ [1,2,3,4], [5,6,7,8]])
print(nM), print (' ')

print(nM.T), print (' ')

print(nM.T.T), print (' ')