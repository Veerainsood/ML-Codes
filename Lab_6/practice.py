from tqdm import tqdm

from torch.utils.data import Dataset,DataLoader,Subset
from torch.optim import Adam
import torch as torch
from torchvision.datasets import CIFAR10
from torchvision import transforms

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((.5,.5,.5),(.5,.5,.5)),
])


train_set = CIFAR10('./data',train=True,download=False,transform=transform) # 32,32

test_set = CIFAR10('./data',train=False,transform=transform,download=False)


train_loader = DataLoader(train_set,batch_size=64,shuffle=True)
test_loader = DataLoader(test_set,batch_size=64,shuffle=False)


class Model(torch.nn.Module):
    def __init__(self,latent_dim, *args, **kwargs):

        super().__init__(*args, **kwargs)

        self.enc = torch.nn.Sequential(
            torch.nn.Conv2d(3,8,kernel_size=2,stride=2), # 16 (32-2)/2 + 1
            torch.nn.BatchNorm2d(8),
            torch.nn.ReLU(),

            torch.nn.Conv2d(8,16,kernel_size=2,stride=2), # 8 (16-2)/2 + 1
            torch.nn.BatchNorm2d(16),
            torch.nn.ReLU(),

            torch.nn.Conv2d(16,32,kernel_size=2,stride=2), # 4
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),

            torch.nn.Flatten(start_dim=1) # batch,32 * 4 * 4
        )

        self.mu = torch.nn.Linear(32*4*4,latent_dim)
        self.logvar = torch.nn.Linear(32*4*4,latent_dim)
        self.deflatten = torch.nn.Linear(latent_dim,32*4*4)

        self.decod = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(32,16,kernel_size=2,stride=2), # 8 = (4-1)*2 + 2
            torch.nn.BatchNorm2d(16),
            torch.nn.LeakyReLU(.2),

            torch.nn.ConvTranspose2d(16,8,kernel_size=2,stride=2), # 16 = (8-1)*2 + 2
            torch.nn.BatchNorm2d(8),
            torch.nn.LeakyReLU(.2),

            torch.nn.ConvTranspose2d(8,3,kernel_size=2,stride=2), # 32 = (16-1)*2 + 2
            torch.nn.BatchNorm2d(3),
            torch.nn.Sigmoid(),
        )
    def encoder(self,x):
        x = self.enc(x)
        mean = self.mu(x)
        logVariance = self.logvar(x)
        return mean,logVariance
    
    def reparametrization(self,mean,log_variance):
        std = torch.exp(1/2 * log_variance)
        eps = torch.randn_like(std)
        return mean + eps * std
    
    def decoder(self,z):
        deflat = self.deflatten(z).view(-1,32,4,4)
        return self.decod(deflat)
    
    def forward(self,x):
        mean , log_variance = self.encoder(x)
        z = self.reparametrization(mean=mean,log_variance=log_variance)
        img = self.decoder(z)
        return img, mean , log_variance
    
model = Model(latent_dim=20)
lr = 1e-3
epochs = 5
optimizer = Adam(model.parameters(),lr=lr,betas=(.5,.999))

def vaeLoss(recon,x,mean,log_var):
    # bce + kme
    x = (x+1)/2 # to bring targets into correct shape
    BCE = torch.nn.functional.binary_cross_entropy(recon,x.view(-1,3,32,32),reduction='sum')
    KLD =  - .5 * torch.sum(1 + log_var - mean.pow(2) - torch.exp(log_var)) # for forcing the probability distr of latent space given x to become normal....
    return BCE + KLD

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('agg')

for epoch in range(epochs):
    lossVal = 0
    Img =0
    for idx,(img,label) in tqdm(enumerate(train_loader),desc=f'Epoch[{epoch}/{epochs}]',total=len(train_loader)):

        optimizer.zero_grad()
        
        recon_Img,mean,log_var  =  model(img)

        loss = vaeLoss(recon=recon_Img,x=img,mean=mean,log_var=log_var)
        
        lossVal += loss.item()
        
        loss.backward()

        optimizer.step()
        Img = recon_Img 

    print(f"Loss : {lossVal}")
    sample_img = Img[0].detach().cpu()  # Detach from computation graph
    sample_img = sample_img.permute(1,2,0)  # Convert from (C, H, W) to (H, W, C)
    
    # Undo normalization (if applied)
    sample_img = (sample_img * 0.5) + 0.5  # Rescale from [-1, 1] to [0,1]
    
    # Plot and save image
    fig, axis = plt.subplots(1, 1, figsize=(4, 4))  # Correct subplot usage
    axis.imshow(sample_img.numpy())  # Convert tensor to NumPy array
    axis.axis('off')  # Remove axis labels

    plt.savefig('current.png')
    plt.close(fig)







