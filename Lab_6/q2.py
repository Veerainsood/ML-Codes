from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset, Subset
from torch import nn, optim
import torch as t
from torch.optim import Adam
from tqdm import tqdm as prettyprint
import matplotlib.pyplot as plt

# Transform for CIFAR-10
transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ]
)

# Load CIFAR-10 dataset
train_set = datasets.CIFAR10(root='../Lab_6_2/Datasets', train=True, transform=transform, download=True)
test_set = datasets.CIFAR10(root='../Lab_6_2/Datasets', train=False, transform=transform, download=True)

train_loader = DataLoader(dataset=train_set, batch_size=64, shuffle=True)
test_loader = DataLoader(dataset=test_set, batch_size=64, shuffle=False)

# Define the VAE Model with convolutional layers
class VAE(nn.Module):
    def __init__(self, latent_dims=20):
        super(VAE, self).__init__()
        
        # Encoder with Convolutions
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=2, stride=2),# (32 -2 )/2 -> 16
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=2, stride=2), #8
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=2, stride=2), #4
            nn.ReLU(),
            nn.Flatten()
        )
        
        # Linear layers to get mu and log_var
        self.fc_mu = nn.Linear(128 * 4 * 4, latent_dims)
        self.fc_log_var = nn.Linear(128 * 4 * 4, latent_dims)
        
        # Decoder with Deconvolutions
        self.decoder_fc = nn.Linear(latent_dims, 128 * 4 * 4)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2), #8
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2), #16
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=3, stride=2), #32
            nn.Sigmoid()
        )
        
    def encode(self, x):
        x = self.encoder(x)
        mu = self.fc_mu(x)
        log_var = self.fc_log_var(x)
        return mu, log_var
    
    def reparameterize(self, mu, log_var):
        std = t.exp(0.5 * log_var) # does e^( 1/2 log(sigma**2)) to get std
        eps = t.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        x = self.decoder_fc(z).view(-1, 128, 4, 4) # transforms x to batches back....
        return self.decoder(x)
    
    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        return self.decode(z), mu, log_var

# Instantiate the model
model = VAE(latent_dims=20)
# model.load_state_dict(t.load("vae_cifar10_model.pth"))
lr = 1e-3
epochs = 15
optimizer = Adam(model.parameters(), lr=lr)
device = t.device('cuda' if t.cuda.is_available() else 'cpu')
model.to(device)

# VAE loss function
def vae_loss_func(recon_x, x, mu, log_var):
    x = (x+1)/2 # brings x between 0 and 1 (originally between -.5 and .5)
    BCE = nn.functional.binary_cross_entropy(recon_x, x.view(-1, 3, 32, 32), reduction='sum')
    KLD = -0.5 * t.sum(1 + log_var - mu.pow(2) - log_var.exp()) # for forcing the probability distr of latent space given x to become normal....
    # essentially it is -1/2 sum( 1 + log(sigmaSquared) - mu^2 - sigma*2)
    return BCE + KLD

# Training the model
for epoch in range(epochs):
    lossval = 0
    for dataIndx, (image, labels) in prettyprint(enumerate(train_loader), total=len(train_loader), desc=f"Epoch [{epoch}/{epochs}]:"):
        
        image = image.to(device)
        optimizer.zero_grad()
        
        decoded, mu, logvar = model(image)
        loss = vae_loss_func(decoded, image, mu, logvar)
        
        lossval += loss.item()
        loss.backward()
        optimizer.step()
    
    print(f"Loss at epoch {epoch}: {lossval}")

# Saving the model
t.save(model.state_dict(), "vae_cifar10_model.pth")

# Testing the model
lossval = 0
with t.no_grad():
    for dataIndx, (image, labels) in prettyprint(enumerate(test_loader), total=len(test_loader)):
        image = image.to(device)
        
        decoded, mu, logvar = model(image)
        loss = vae_loss_func(decoded, image, mu, logvar)
        lossval += loss.item()

    print(f"Test Loss: {lossval}")

# Visualize some generated and original images
generated = []
actual = []
fig, axs = plt.subplots(3, 2, figsize=(10, 4))
for i in range(3):
    (images, labels) = next(iter(test_loader))
    decoded, mu, logvar = model(images[i].unsqueeze(0).to(device))
    generated.append(decoded.squeeze(0).detach().cpu())
    actual.append(images[i].squeeze(0).detach().cpu())
    
    axs[i, 0].imshow(actual[i].permute(1, 2, 0))
    axs[i, 1].imshow(generated[i].permute(1, 2, 0))

plt.show()

# Generate random images
with t.no_grad():
    random_z = t.randn(64, 20).to(device)
    random_images = model.decode(random_z)
    
    fig, axs = plt.subplots(8, 8, figsize=(10, 10))
    for i in range(8):
        for j in range(8):
            axs[i, j].imshow(random_images[i * 8 + j].cpu().permute(1, 2, 0))
            axs[i, j].axis('off')
    
    plt.show()
