import numpy as np
import torch as torch

'''

VVIPs

normalizing x is very imp... so divide by 1000

ALWAYS keep inputs between [0,1] to avoid errors
prioritize using matrix operations to do things

'''

size = 1000
x = torch.linspace(1,1000,size)
x = x / 1000

h_0 = 105
h_1 = 192
h_2 = 157
h_3 = 119


H = torch.tensor([h_0,h_1,h_2,h_3],dtype=torch.float).reshape(4,1)

c_0 = 1
c_1 = 100
c_2 = -20
c_3 = 37

C = torch.tensor([c_0,c_1,c_2,c_3],dtype=torch.float).reshape(4,1)

print(C.shape,H.shape)

lr = 1e-4
epochs = 600
Shyama = torch.vstack([torch.ones((size)),x,x**2,x**3])

Shyama = torch.matmul(Shyama , Shyama.T)


Shyam  = torch.zeros_like(Shyama)
for i in range(Shyama.shape[0]):
    Shyam[i][i] = Shyama[i][i]

print(Shyam.shape)

for epoch in range(epochs):

    grad_mat = torch.clamp((2 * Shyam @ (C-H) ) * lr,-10,10)

    c_0  = c_0 - grad_mat[0] 
    c_1  = c_1 - grad_mat[1]
    c_2  = c_2 - grad_mat[2]
    c_3  = c_3 - grad_mat[3]
    
    loss = (C-H).T @ Shyama @ (C-H)

    C = torch.tensor([c_0,c_1,c_2,c_3],dtype=torch.float).reshape(4,1)

    print(loss)


print(f"Final Learnt Values: c_0 : {c_0}, c_1 : {c_1}, c_2 : {c_2}, c_3 : {c_3}")
    

