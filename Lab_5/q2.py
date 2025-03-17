import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])  # Normalize to [-1, 1]
])

train_set = datasets.MNIST(root='../Lab_3/Datasets', train=True, transform=transform, download=True)
test_set = datasets.MNIST(root='../Lab_3/Datasets', train=False, transform=transform, download=True)

train_set = Subset(train_set, torch.randint(0, 60000, (20000,)))
test_set = Subset(test_set, torch.randint(0, 10000, (1000,)))

train_loader = DataLoader(dataset=train_set, batch_size=64, shuffle=True)
test_loader = DataLoader(dataset=test_set, batch_size=64, shuffle=False)


class VarAutoEncoder(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=400, latent_dim=20):
        super(VarAutoEncoder, self).__init__()
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc21 = nn.Linear(hidden_dim, latent_dim)  
        self.fc22 = nn.Linear(hidden_dim, latent_dim)  
        
        self.fc3 = nn.Linear(latent_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, input_dim)
        
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    
    def encoder(self, x):
        h1 = self.relu(self.fc1(x))
        mu = self.fc21(h1)
        log_var = self.fc22(h1)
        return mu, log_var
    
    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decoder(self, z):
        h3 = self.relu(self.fc3(z))
        return self.sigmoid(self.fc4(h3))
    
    def forward(self, x):
        mu, log_var = self.encoder(x.view(-1, 784))
        z = self.reparameterize(mu, log_var)
        return self.decoder(z), mu, log_var

model = VarAutoEncoder()
optimizer = optim.Adam(model.parameters(), lr=1e-4)


def loss_function(recon_x, x, mu, log_var):

    x_rescaled = (x + 1) / 2 
    BCE = nn.functional.binary_cross_entropy(recon_x, x_rescaled.view(-1, 784), reduction='sum')
    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return BCE + KLD


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

epochs = 15
for epoch in range(epochs):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(tqdm(train_loader)):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, log_var = model(data)
        loss = loss_function(recon_batch, data, mu, log_var)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
    
    print(f'Epoch {epoch + 1}, Loss: {train_loss / len(train_loader.dataset):.4f}')

model.eval()
with torch.no_grad():
    sample = torch.randn(64, 20).to(device)
    sample = model.decoder(sample).cpu()
    sample = sample.view(64, 1, 28, 28)

    sample = sample * 2 - 1

fig, axes = plt.subplots(8, 8, figsize=(10, 10))
for i, ax in enumerate(axes.flatten()):
    ax.imshow(sample[i].squeeze(), cmap='gray')
    ax.axis('off')
plt.show()