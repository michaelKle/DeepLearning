import numpy as np

values = [1, -1, 3, 0, 4, 3]

print(f"values={values}")
print(f"min(values)={min(values)}")
print(f"max(values)={max(values)}")
print(f"argmin(values)={np.argmin(values)}")
print(f"argmax(values)={np.argmax(values)}")

M = np.array([ [0,1,10], [20,8,5]])
print(f"M=\n{M}"), print(' ')

print(f'np.min(M)={np.min(M)}')
print(f'np.min(M,axis=0)={np.min(M,axis=0)}') # axis=0 -> 3 columns
print(f'np.min(M,axis=0)={np.min(M,axis=1)}') # axis=1 -> 2 row
print(f'np.argmin(M)={np.argmin(M)}')
print(f'np.argmin(M,axis=0)={np.argmin(M,axis=0)}') # axis=0 -> 3 columns
print(f'np.argmin(M,axis=0)={np.argmin(M,axis=1)}') # axis=1 -> 2 row

import torch

vt = torch.tensor(values)
print(f"vt={vt}")
print(f"torch.min(vt)={torch.min(vt)}")
print(f"torch.max(vt)={torch.max(vt)}")
print(f"torch.argmin(vt)={torch.argmin(vt)}")
print(f"torch.argmax(vt)={torch.argmax(vt)}")

M = torch.tensor(([ [0,1,10], [20,8,5]]))
print(f"M=\n{M}"), print(' ')

print(f'torch.min(M)={torch.min(M)}')
print(f'torch.min(M,axis=0)={torch.min(M,axis=0)}') # axis=0 -> 3 columns
print(f'torch.min(M,axis=0).indices={torch.min(M,axis=0).indices}') # axis=0 -> 3 columns
print(f'torch.min(M,axis=0)={torch.min(M,axis=1)}') # axis=1 -> 2 row
print(f'torch.argmin(M)={torch.argmin(M)}')
print(f'torch.argmin(M,axis=0)={torch.argmin(M,axis=0)}') # axis=0 -> 3 columns
print(f'torch.argmin(M,axis=0)={torch.argmin(M,axis=1)}') # axis=1 -> 2 row