import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

transform = transforms.Compose(
    [
        transforms.ToTensor(), 
        transforms.Normalize((0.5,), (0.5,))
    ]                           
)
'''
For MNIST, a simple 1-2 hidden layer network is usually enough. But for CIFAR-10, ImageNet, or medical imaging, you'll need much deeper architectures (like ResNets, VGG, or Transformers).

'''

train_set = datasets.MNIST('./Lab_2/Datasets',train=True,download=True,transform=transform)
test_set = datasets.MNIST('./Lab_2/Datasets',train=False,download=True,transform=transform)

train_set = torch.utils.data.Subset(dataset=train_set,indices=torch.arange(0,60000,step=1,dtype=torch.int))
test_set = torch.utils.data.Subset(dataset=test_set,indices=torch.arange(0,1000,step=1,dtype=torch.int))

train_loader = DataLoader(train_set,batch_size=128,shuffle=True)
test_loader = DataLoader(test_set,batch_size=64,shuffle=False)

class Model(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.W1 = torch.nn.Parameter(torch.randn((28*28,128),requires_grad=True))
        self.W2 = torch.nn.Parameter(torch.randn((128,10),requires_grad=True))

        self.b1 = torch.nn.Parameter(torch.randn((128),requires_grad=True))
        self.b2 = torch.nn.Parameter(torch.randn((10),requires_grad=True))

        torch.nn.init.xavier_uniform_(self.W1)
        torch.nn.init.xavier_uniform_(self.W2)

        self.relu = torch.nn.ReLU()
        
    def forward(self,x):
        x = self.relu(
            torch.matmul(x,self.W1) + self.b1
        )
        x = torch.matmul(x,self.W2) + self.b2
        return x

lr = 1e-4

model = Model()

optimizer = torch.optim.Adam(model.parameters(),lr=lr)

epochs = 20

cel = torch.nn.CrossEntropyLoss()


for epoch in range(epochs):
    lossval = 0
    
    for batchIdx, (images, labels) in tqdm(enumerate(train_loader),total=len(train_loader),desc=f"Epoch [{epoch}/{epochs}]:"):
        optimizer.zero_grad()

        images = torch.flatten(images.squeeze(), start_dim=1)
        y_pred = model(images)
        
        loss = cel(y_pred, labels.type(torch.LongTensor))

        loss.backward()

        optimizer.step()

        lossval = loss.item()
    print(f"Loss: {lossval}")

correctpred = 0
total_pred = 0
for datasetIndx , (images,labels) in tqdm(enumerate(test_loader),total=len(test_loader)):
    images = torch.flatten(torch.squeeze(images,dim=1),start_dim=1)
    y_pred = model(images)

    loss = cel(y_pred,labels.type(torch.LongTensor))
    correctpred += torch.sum(torch.argmax(y_pred,dim=1) == labels)
    total_pred += len(labels)
    
print(f"accuracy: {correctpred/total_pred}")

    


