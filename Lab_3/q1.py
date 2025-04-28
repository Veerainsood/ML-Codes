import torch as t
import kagglehub as kh
from torch.utils.data import Dataset,DataLoader,Subset
from torchvision import datasets,transforms,models
import tqdm as tq

transform = transforms.Compose([
    models.ResNet50_Weights.IMAGENET1K_V1.transforms()
])
path = kh.dataset_download("dimensi0n/imagenet-256")

datset = datasets.ImageFolder(path,transform=transform)

trainSize = 30000
testSize = 1000

assert trainSize < len(datset)
assert testSize <= trainSize

train_idx = t.randperm( len(datset) ) [:trainSize]

# train index -> tensor of length trainSize with random indices throughout datset...

test_indx = train_idx[ t.randperm(trainSize)[:testSize] ] # this line aims to pic up few indices from train_index

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
        self.res = models.resnet50(weights= models.ResNet50_Weights.DEFAULT , progress = True)
        self.Lin1 = t.nn.Linear(1000,1000)
        self.Lin2 = t.nn.Linear(1000,1000)
        self.relu = t.nn.ReLU()
        for params in self.res.parameters():
            params.requires_grad = False

    def forward(self,x):
        x = self.res(x)
        # print(x.shape)
        # x = x.view(x.shape[0],-1)
        x = self.Lin1(x)
        x = self.Lin2(x)
        return x


model = Model()
model.to(device=device)

# model.load_state_dict(t.load("model.pth",map_location=t.device('cpu')))

lr = 1e-3

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
