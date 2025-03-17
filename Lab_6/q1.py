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

train_set = datasets.MNIST(root='../Lab_3/Datasets',train=True,transform=transform)

test_set = datasets.MNIST(root='../Lab_3/Datasets',train=False,transform=transform)

train_set = Subset(train_set,randint(0,60000,(20000,)))

test_set = Subset(test_set,randint(0,60000,(1000,)))

train_loader = DataLoader(dataset=train_set,batch_size=64,shuffle=True)
test_loader = DataLoader(dataset=train_set,batch_size=64,shuffle=False)

# print(train_loader.dataset[0][0].shape)
# exit(0)
# (n + 1) * s + f - 2p = output size


class Model(nn.Module):
    def __init__(self, latent_dims=20):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(784,256),
            nn.ReLU(),
            nn.Linear(256,64),
            nn.ReLU()
        )

        self.fc_mu      = nn.Linear(64,latent_dims)
        self.fc_log_var = nn.Linear(64,latent_dims)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dims,64),
            nn.ReLU(),
            nn.Linear(64,256),
            nn.ReLU(),
            nn.Linear(256,784),
            nn.Sigmoid()
        )

    def encod(self,x):
        x = x.view(x.shape[0],-1)
        x = self.encoder(x)
        mu, log_var = self.fc_mu(x),self.fc_log_var(x)
        return (mu,log_var)
    
    def reparameterize(self,mu,logvar):
        std = t.exp(.5*logvar)
        eps = t.rand_like(std)
        return mu + eps * std

    def decod(self,x):
        return self.decoder(x)

    def forward(self,x):
        mu,logvar = self.encod(x)
        z = self.reparameterize(mu,logvar=logvar)
        return (self.decod(z), mu,logvar)

    
model = Model(10)
model.load_state_dict(t.load("q_1_model.pth"))
lr = 1e-3

epochs = 15

optimizer = Adam(model.parameters(),lr = lr)

device = t.device('cuda' if t.cuda.is_available() else 'cpu')

def vae_loss_func(recon_x,x,mu,logvar):
    x = (x + 1) / 2 
    BCE = nn.functional.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')  # Reconstruction loss
    KLD = -0.5 * t.sum(1 + logvar - mu.pow(2) - logvar.exp())  # KL Divergence

    return BCE + KLD 

for epoch in range(epochs):
    lossval = 0
    for dataIndx , (image,labels) in prettyprint(enumerate(train_loader),total=len(train_loader),desc=f"Epoch [{epoch}/{epochs}]:"):
        
        image,labels = image.to(device),labels.to(device)

        optimizer.zero_grad()

        decoded,mu,logvar = model(image)

        loss = vae_loss_func(decoded, image, mu, logvar)

        lossval += loss.item()

        loss.backward() # computes gradients

        optimizer.step()
        lossval += loss.item()
    print(f"Loss: {lossval}")

t.save(model.state_dict(),"q_1_model.pth")

correctPred = 0
totalPred = 0
lossval = 0
for dataIndx , (image,labels) in prettyprint(enumerate(test_loader),total=len(test_loader)):

    with t.no_grad():
        image,labels = image.to(device),labels.to(device)

        optimizer.zero_grad()

        decoded,mu,logvar = model(image)

        loss = vae_loss_func(decoded, image, mu, logvar)

        lossval += loss.item()

    
    print(f"Loss: {lossval}")




import matplotlib.pyplot as plt
import numpy as np

generated = []
actual = []
fig , axs = plt.subplots(3,2,figsize=(10,4))
for i in range(3):
    (images,labels) = next(iter(test_loader))
    (decoded,mu,logvar) = model(images[i].unsqueeze(0))
    generated.append( 
        (
            decoded.squeeze(0).squeeze(0).reshape((28,28)).detach()
        )
    )
    actual.append(
        (
            images[i].squeeze(0).detach()
        )
    )
    axs[i,0].imshow(actual[i],cmap="Greys")
    axs[i,1].imshow(generated[i],cmap="Greys")

generated = np.array(generated)
actual = np.array(actual)
    
plt.show()
