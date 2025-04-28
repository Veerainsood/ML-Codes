from torchvision import datasets,transforms
from torch.utils.data import DataLoader,Dataset,Subset
from torch import randint, nn as nn
from torch.optim import Adam
from tqdm import tqdm as prettyprint
import torch as t
transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize([.5,],[.5,])
    ]
)
# https://colab.research.google.com/drive/1FVX3Skws0YstOgVFbowT87DH_JPklogq

train_set = datasets.MNIST(root='../Lab_3/Datasets',train=True,transform=transform,download=True)

test_set = datasets.MNIST(root='../Lab_3/Datasets',train=False,transform=transform,download=False)

train_set = Subset(train_set,randint(0,60000,(20000,)))
test_set = Subset(test_set,randint(0,10000,(1000,)))

train_loader = DataLoader(dataset=train_set,batch_size=64,shuffle=True)
test_loader = DataLoader(dataset=train_set,batch_size=64,shuffle=False)

# print(train_loader.dataset[0][0].shape)
# exit(0)
'''
Transposed Cov:
    (n - 1) * s + f - 2p + outputPadding = output size

Cov:
    (n - f + 2p)/s + 1 = output size

'''
# 



class Model(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.Conv1 = nn.Conv2d(1,64,kernel_size=2,stride=2)    #(28-2 )/2 + 1 -> 14
        self.Conv2 = nn.Conv2d(64,128,kernel_size=2,stride=2)  #(14-2)/2 + 1 -> 7
        self.Conv3 = nn.Conv2d(128,256,kernel_size=3,stride=2) #(7-3)/2 + 1 -> 3
        self.bn1   = nn.BatchNorm2d(64)
        self.bn2   = nn.BatchNorm2d(128)
        self.bn3   = nn.BatchNorm2d(256)

        self.revConv1 = nn.ConvTranspose2d(256,128,kernel_size=3,stride=2) # (3 - 1) * 2  + 3 -> 7
        self.revConv2 = nn.ConvTranspose2d(128,64,kernel_size=2,stride=2)  # (7 - 1) * 2 + 2 -> 14
        self.revConv3 = nn.ConvTranspose2d(64,1,kernel_size=2,stride=2)    # (14 -1) * 2 + 2 -> 28
        self.bn11   = nn.BatchNorm2d(128)
        self.bn22   = nn.BatchNorm2d(64)
        self.relu  = nn.ReLU()
        self.tan = nn.Tanh()
        
        self.Lin1  = nn.Linear(784,256)
        self.Lin2  = nn.Linear(256,64)
        self.Lin3  = nn.Linear(64,10)

    def encoder(self,x):
        x = self.relu(self.bn1(self.Conv1(x)))
        x = self.relu(self.bn2(self.Conv2(x)))
        x = self.bn3(self.Conv3(x))
        return x
    
    def decoder(self,x):
        x = self.relu(self.bn11(self.revConv1(x)))
        x = self.relu(self.bn22(self.revConv2(x)))
        x = self.tan(self.revConv3(x))
        return x
    
    def forward(self,x):
        x = self.encoder(x)
        x = self.decoder(x)
        decoded = x
        x = x.view(x.shape[0],-1)
        x = self.Lin1(x)
        x = self.Lin2(x)
        x = self.Lin3(x)
        return (decoded,x)

    
model = Model()
model.load_state_dict(t.load("q_1_model.pth"))
lr = 1e-3

epochs = 15

optimizer = Adam(model.parameters(),lr = lr)

device = t.device('cuda' if t.cuda.is_available() else 'cpu')

mse = t.nn.MSELoss()
cel = t.nn.CrossEntropyLoss()
# for epoch in range(epochs):
#     lossval = 0
#     correctPred = 0
#     totalPred = 0
#     for dataIndx , (image,labels) in prettyprint(enumerate(train_loader),total=len(train_loader),desc=f"Epoch [{epoch}/{epochs}]:"):
        
#         image,labels = image.to(device),labels.to(device)

#         optimizer.zero_grad()

#         decoded,y_pred = model(image)

#         mseloss = mse(decoded,image)
#         cesloss = cel(y_pred,labels.long())

#         loss = mseloss + cesloss

#         lossval += cesloss.item() + mseloss.item()

#         correctPred += t.sum( 
#             t.argmax(y_pred,dim=1) == labels
#         )

#         totalPred += len(labels)

#         loss.backward() # computes gradients

#         optimizer.step() # applies the grads..

#         lossval += loss.item()
        
#     print(f"Loss: {lossval}")
#     print(f"Accuracy : {correctPred/totalPred}")

# t.save(model.state_dict(),"q_1_model.pth")

# correctPred = 0
# totalPred = 0
# lossval = 0
# for dataIndx , (image,labels) in prettyprint(enumerate(test_loader),total=len(test_loader)):

#     image , labels = image.to(device),labels.to(device)

#     decoded,y_pred = model(image)

#     # mseloss = mse(decoded,image)
#     cesloss = cel(y_pred,labels.long())

#     lossval += cesloss.item()

#     correctPred += t.sum( 
#         t.argmax(y_pred,dim=1) == labels
#     )

#     totalPred += len(labels)
# print(f"Accuracy : {correctPred/totalPred}")


import matplotlib.pyplot as plt
import numpy as np

generated = []
actual = []
fig , axs = plt.subplots(3,2,figsize=(10,4))
for i in range(3):
    (images,labels) = next(iter(test_loader))
    # print(images[i].shape,images[i].unsqueeze(0).shape)
    (gen,pred) = model(images[i].unsqueeze(0))
    generated.append( 
        (
            gen.squeeze(0).squeeze(0).detach()
        )
    )
    actual.append(
        (
            images[i].squeeze(0).detach()
        )
    )
    axs[i,0].imshow(actual[i],cmap="Greys")
    axs[i,0].set_title(labels[i].item())
    axs[i,1].imshow(generated[i],cmap="Greys")
    axs[i,1].set_title(f"predicted:{(t.argmax(pred)).item()}")
    # axs[i,0].axis("off")
    # axs[i,1].axis("off")

generated = np.array(generated)
actual = np.array(actual)
    
plt.savefig("fig.png")
