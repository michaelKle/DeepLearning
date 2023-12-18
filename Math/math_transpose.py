import numpy as np
import torch

print('##### Vector with numpy')

nv = np.array([[1,2,3,4]])
print(nv), print(' ')

print (nv.T), print(' ')

nvT = nv.T
print (nvT.T), print(' ')


print('##### Matrix with numpy')

nM = np.array([ [1,2,3,4], [5,6,7,8]])
print(nM), print (' ')

print(nM.T), print (' ')

print(nM.T.T), print (' ')

print('##### Vector with torch')

tv = torch.tensor([[1,2,3,4]])
print(tv), print(' ')
print (tv.T), print(' ')
print (tv.T.T), print(' ')


print('##### Matrix with torch')
tM = torch.tensor([ [1,2,3,4], [5,6,7,8]])
print(tM), print (' ')
print(tM.T), print (' ')
print(tM.T.T), print (' ')



print('##### Types')
print(f'nv is of type {type(nv)}')
print(f'nM is of type {type(nM)}')
print(f'tv is of type {type(tv)}')
print(f'tM is of type {type(tM)}')