import torch as t
import kagglehub as kh
from torch.utils.data import Dataset,DataLoader
from torchvision import datasets,transforms
import torchvision as tv
import tqdm as tq

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([.5,],[.5,])
])
path = kh.dataset_download("dimensi0n/imagenet-256")

datset = datasets.ImageFolder(root=path,transform=transform)

trainSize = 30000
testSize = 1000

assert trainSize < len(datset)
assert testSize <= trainSize

train_idx = t.randperm( len(datset) ) [:trainSize]

# train index -> tensor of length trainSize with random indices throughout datset...

test_indx = train_idx[ t.randperm(trainSize)[:1000] ] # this line aims to pic up few indices from train_index

train_idx = train_idx[~t.isin(train_idx,test_indx)]

train_subset = t.utils.data.Subset(
    datset,
    train_idx
)

test_subset = t.utils.data.Subset(
    datset,
    test_indx
)

device = t.device("cuda" if t.cuda.is_available() else "cpu")

train_loader = DataLoader(train_subset,batch_size=64,shuffle=True,num_workers=15)
test_loader = DataLoader(test_subset,batch_size=64,shuffle=False,num_workers=15)



class Model(t.nn.Module):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.Conv1   = t.nn.Conv2d(3,64,kernel_size=3,padding='same')
        self.Conv2   = t.nn.Conv2d(64,128,kernel_size=3,padding='same')
        self.pool    = t.nn.MaxPool2d(kernel_size=2)
        self.GAP     = t.nn.AdaptiveAvgPool2d(1)
        self.Lin1    = t.nn.Linear(128,1000)
        self.relu    = t.nn.ReLU()

    def forward(self,x):
        x = self.relu(self.Conv1(x))
        x = self.pool(x)
        x = self.relu(self.Conv2(x))
        x = self.pool(x)
        x = self.GAP(x)
        x = x.view(x.shape[0],-1)
        x = self.Lin1(x)
        return x


model = Model()
model.to(device=device)

lr = 1e-4

epochs = 10

optimizer = t.optim.Adam(model.parameters(),lr = lr)

cel = t.nn.CrossEntropyLoss()

for epoch in range(epochs):
    lossval = 0
    for dataIndx , (image,labels) in tq.tqdm(enumerate(train_loader),total=len(train_loader),desc=f"Epoch [{epoch}/{epochs}]:"):
        
        image,labels = image.to(device),labels.to(device)

        optimizer.zero_grad()

        y_pred = model(image)


        loss = cel(y_pred,labels.long())

        loss.backward() # computes gradients

        optimizer.step()
        lossval += loss.item()
    print(f"Loss: {lossval}")

t.save(model.state_dict(),"model.pth")

correctPred = 0
totalPred = 0
for dataIndx , (image,labels) in tq.tqdm(enumerate(test_loader),total=len(test_loader),desc=f"Epoch [{epoch}/{epochs}]:"):

    y_pred = model(image)

    loss = cel(y_pred,labels.long())

    lossval += loss.item()

    correctPred += t.sum( 
        t.argmax(y_pred,dim=1) == labels
    )

    totalPred += len(labels)
print(f"Accuracy : {correctPred/totalPred}")
