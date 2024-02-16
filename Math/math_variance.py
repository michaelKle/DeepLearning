import numpy as np

values = [-2, 0, 4, 1, 7.0]
#  use ony float value so it will work with torch

print(f"values={values}")
print(f"mean={np.mean(values)}")
print(f"variance={np.var(values)}") # (1/n) * sum..
print(f"variance={np.var(values, ddof=1)}") # (1/(n-1)) * sum.. - also called unbiased measure



import torch

#t1 = torch.tensor([1,2,3,4])
t1 = torch.tensor(values)
print(f"t1={t1}")
print(f"mean={torch.mean(t1)}")
print(f"variance={torch.var(t1)}")
