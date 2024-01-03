import torch

t1 = torch.tensor([1,2,3,4])
t2 = torch.tensor([1,2,3,4])

print(f"t1={t1}")
print(f"t2={t2}")

print(f"sum(t1*t2)={sum(t1*t2)}")
print(f"t1.dot(t2)={t1.dot(t2)}")

m1 = torch.tensor([[0,3,2],[-3,-3,1],[1, 0, 2]])
m2 = torch.tensor([[1,0,6],[2,-1,0],[5, 1, 4]])
print(f"m1={m1}")
print(f"m2={m2}")
print(f"m1*m2={m1*m2}")
#print(f"m1.mm(m2)={m1.mm(m2)}")
print(f"sum(sum(m1*m2))={sum(sum(m1*m2))}")