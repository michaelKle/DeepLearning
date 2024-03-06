import numpy as np


sample = np.random.randn(5)
print(f"Sample={sample}")

np.random.seed(17)
sample = np.random.randn(5)
print(f"Sample(with seed)={sample}")

rg1 = np.random.RandomState(17)
rg2 = np.random.RandomState(19)

print(f"Sample(rg1)={rg1.randn(5)}")
print(f"Sample(rg2)={rg2.randn(5)}")

import torch

print(f"Torch={torch.randn(5)}")
#print(f"RNG-State={torch.random.get_rng_state()}")