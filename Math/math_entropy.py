import numpy as np
import matplotlib.pyplot as plt

# probability of an event happening:
p = .25
q = 1 - p

H = - (p*np.log(p) + q*np.log(q))
print(f"Entropy: {H}")

# also called "binary entropy"
Hb = -( p*np.log(p) + (1-p)*np.log(1-p))
print(f"binary entropy: {Hb}")

# general solution
ps = [.25, .75]
H2 = -sum(map(lambda x: x*np.log(x), ps))
print(f"Entropy Lambda: {H2}")


# now cross entropy - two sets of events
p = [1, 0]
q = [.25, .75]

assert(len(p) == len(q))
H = 0
for i in range(len(p)):
    H -= p[i]*np.log(q[i])

print(f"Cross-Entropy: {H}")

# simplification because of p[1] == 0
# p[0] == 1
H = -np.log(q[0])

## now pytorch
import torch
import torch.nn.functional as F

p_tensor = torch.Tensor(p)
q_tensor = torch.Tensor(q)

# F.binary_cross_entropy(p,q) <- does not work
# F.binary_cross_entropy(p_tensor,q_tensor) <- wrong value

H = F.binary_cross_entropy(q_tensor,p_tensor)
# q_tensor = model prediction
# p_tensor = category lables (is a cat / or not)
print(f'Pytorch: {H}')



