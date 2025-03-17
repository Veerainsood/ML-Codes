from torchvision.datasets import CIFAR10
from torchvision import transforms
import torch as torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm
from torch.utils.data import DataLoader

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((.5,.5,.5),(.5,.5,.5))
])

trainset = CIFAR10(root='./data',train=True,transform=transform,download=True)
testset = CIFAR10(root='./data',train=False,transform=transform,download=False)

trainLoader = DataLoader(trainset,batch_size=64,shuffle=True)

class Model(nn.Module):
    def __init__(self,latent_dim, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.enc = nn.Sequential(
                nn.Conv2d(in_channels=3,out_channels=8,kernel_size=2,stride=2),#16
                nn.BatchNorm2d(8),
                nn.ReLU(),

                nn.Conv2d(in_channels=8,out_channels=16,kernel_size=2,stride=2),#8
                nn.BatchNorm2d(16),
                nn.ReLU(),

                nn.Conv2d(in_channels=16,out_channels=32,kernel_size=2,stride=2),#4
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.Flatten(start_dim=1) # (batch,32*4*4)
        )
        self.mu = nn.Linear(in_features=32*4*4,out_features=latent_dim)
        self.logvar = nn.Linear(in_features=32*4*4,out_features=latent_dim)
        self.defloat = nn.Linear(in_features=latent_dim,out_features=32*4*4)
        self.dec = nn.Sequential(
                nn.ConvTranspose2d(in_channels=32,out_channels=16,kernel_size=2,stride=2),#  16,8,8
                nn.BatchNorm2d(16),
                nn.LeakyReLU(.2),

                nn.ConvTranspose2d(in_channels=16,out_channels=8,kernel_size=2,stride=2),# 8, 16,16
                nn.BatchNorm2d(8),
                nn.LeakyReLU(.2),

                nn.ConvTranspose2d(in_channels=8,out_channels=3,kernel_size=2,stride=2),#3 , 32,32
                nn.BatchNorm2d(3),
                nn.Sigmoid(),
        )

    def encoder(self,x):
        x = self.enc(x)
        mean = self.mu(x)
        logvar = self.logvar(x)
        return mean,logvar

    def reparametrization(self,mean,logvar):
        std = torch.exp(.5 * logvar)
        eps = torch.randn_like(std)        
        return mean+ eps*std
    
    def decode(self,z):
        defl = self.defloat(z).view(-1,32,4,4)
        return self.dec(defl)
    
    def forward(self,x):
        mean , logvar = self.encoder(x)
        z = self.reparametrization(mean=mean,logvar=logvar)
        imp = self.decode(z)
        return imp,mean,logvar
    
model = Model(latent_dim=20)
lr = 1e-3
epochs = 5
optimizer = Adam(model.parameters(),lr=lr,betas=(.5,.999))

def vaeLoss(recon,x,mean,logvar):
    x = (x+1)/2
    BCE = torch.nn.functional.binary_cross_entropy(recon,x,reduction='sum')
    KLE = -.5 * torch.sum(1 + logvar - mean.pow(2) - torch.exp(logvar))
    return KLE + BCE

for epoch in range(epochs):
    lossVal = 0
    Img =0
    for idx,(img,label) in tqdm(enumerate(trainLoader),desc=f'Epoch[{epoch}/{epochs}]',total=len(trainLoader)):

        optimizer.zero_grad()
        
        recon_Img,mean,log_var  =  model(img)

        loss = vaeLoss(recon=recon_Img,x=img,mean=mean,logvar=log_var)
        
        lossVal += loss.item()
        
        loss.backward()

        optimizer.step()

    print(f"Loss : {lossVal}")
