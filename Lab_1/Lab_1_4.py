import numpy as np
import torch as torch

'''

VVIPs

normalizing x is very imp... so divide by 1000

ALWAYS keep inputs between [0,1] to avoid errors
prioritize using matrix operations to do things

'''

size = 1000
x = torch.linspace(1,1000,size,requires_grad=False)
x = x / 1000

h_0 = 105
h_1 = 192
h_2 = 157
h_3 = 119


H = torch.tensor([h_0,h_1,h_2,h_3],dtype=torch.float,requires_grad=False).reshape(4,1)

C = torch.randint(0, 10, (4,1), dtype=torch.float, requires_grad=True)

print(C.shape,H.shape)

lr = 1e-4
epochs = 5000
Shyama = torch.vstack([torch.ones((size)),x,x**2,x**3])

Shyama = torch.matmul(Shyama , Shyama.T)

Shyam = torch.diag_embed(Shyama)

print(Shyama.shape)

optimizer = torch.optim.SGD([C],lr=lr)

for epoch in range(epochs):

    loss = (C-H).T @ Shyama @ (C-H)

    optimizer.zero_grad()

    loss.backward()
    
    optimizer.step()
    
    if(epoch%100 ==0):
        print(loss.item())


print(f"Final Learnt Values: c_0 : {C[0].item()}, c_1 : {C[1].item()}, c_2 : {C[2].item()}, c_3 : {C[3].item()}")