import torch as tr
import numpy as np

size = 500
X = tr.linspace(1,size+1,steps=size,dtype=tr.float)
X = X / size
h_0 = -50
h_1 = 157
h_2 = -290
h_3 = 372
h_4 = -128

H = tr.tensor([h_0,h_1,h_2,h_3,h_4],requires_grad=False,dtype=tr.float)
H = H.permute(*tr.arange(H.ndim - 1, -1, -1))
c_0 = -5
c_1 = 15
c_2 = 19
c_3 = 32
c_4 = 180

C = tr.tensor([c_0,c_1,c_2,c_3,c_4],requires_grad=False,dtype=tr.float)
C = C.permute(*tr.arange(C.ndim - 1, -1, -1))

lr = .2
Shyama = tr.vstack([tr.ones(size),X,X**2,X**3,X**4])
Shyama = Shyama.T # 500,5

epochs = 6000

Y_true = Shyama @ H

for i in range(epochs):
    
    Y_pred = Shyama @ C

    error = Y_pred - Y_true # 500,1
    
    grad = lr * tr.clamp( Shyama.T  @ (2/size * error )  ,-1000,1000)
    # print(grad.shape)
    C = C - grad

    if i % 1000 == 0:
        # lr *= 0.35
        print(f'Loss: {error.T @ error}')

print(f'final values of C : {C}')