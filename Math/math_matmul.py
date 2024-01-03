import numpy as np

m1 = np.array([ [1,2], [3,4]])
m2 = np.array([ [5,6], [7,8]])
print(f"m1@m2 = {m1@m2}")

print(f"m1[0]*m2[:,0]={m1[0].dot(m2[:,0])}")
print(f"m1[0]*m2[:,1]={m1[0].dot(m2[:,1])}")
print(f"m1[1]*m2[:,0]={m1[1].dot(m2[:,0])}")
print(f"m1[1]*m2[:,1]={m1[1].dot(m2[:,1])}")

# random matrices
# rows x columns
A = np.random.randn(3,4)
B = np.random.randn(4,5)
C = np.random.randn(3,7)

print(f"A@B=\n{np.round(A@B,2)}")
print(f"A.T@C=\n{np.round(A.T@C,2)}")

import torch

A = torch.randn(3,4)
B = torch.randn(4,5)
C1 = np.random.randn(4,7)
C2 = torch.tensor(C1, dtype=torch.float)

# try some mults
print(f"A@B=\n{np.round(A@B,2)}")
# print(f"A@B.T=\n{np.round(A@B.T,2)}") <- wont work because dimensions do not fit for mat mul
print(f"A@C1=\n{np.round(A@C1,2)}")
print(f"A@C2=\n{np.round(A@C2,2)}")