import torch as t
import kagglehub as kh
from torch.utils.data import Dataset,DataLoader
from torchvision import datasets,transforms
import torchvision as tv
import torchvision.models as models
import tqdm as tq

transform = transforms.Compose([
    models.ResNet50_Weights.IMAGENET1K_V1.transforms()
])
path = kh.dataset_download("dimensi0n/imagenet-256")

datset = datasets.ImageFolder(root=path,transform=transform)

classes = [0,1,2,3,4,5,6,7,8,9]

subset = t.utils.data.Subset(datset,[i for i in range(len(datset)) if datset.imgs[i][1] in classes])

datset = subset

trainSize = 4000
testSize = len(datset.indices) - 4000

assert trainSize < len(datset)
assert testSize <= trainSize

train_idx = t.randperm( len(datset) ) [:trainSize]
indexes   = t.arange(len(datset))
test_idx  = indexes[~t.isin(indexes,train_idx)]

# exit(0)

train_subset = t.utils.data.Subset(
    datset,
    train_idx
)

test_subset = t.utils.data.Subset(
    datset,
    test_idx
)

device = t.device("cuda" if t.cuda.is_available() else "cpu")

train_loader = DataLoader(train_subset,batch_size=64,shuffle=True,num_workers=15)
test_loader = DataLoader(test_subset,batch_size=64,shuffle=False,num_workers=15)


class Model(t.nn.Module):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.Conv1 = t.nn.Conv2d(3,8,kernel_size=4,stride=2,padding=1)#114
        self.Conv2 = t.nn.Conv2d(8,64,kernel_size=4,stride=2,padding=1)#54
        self.Conv3 = t.nn.Conv2d(64,128,kernel_size=4,stride=2,padding=1)#28
        self.Lin1  = t.nn.Linear(128*28*28,64)
        self.Lin2  = t.nn.Linear(64,32)
        self.Lin3  = t.nn.Linear(32,10)
        self.relu  = t.nn.ReLU()
        self.bn1   = t.nn.BatchNorm2d(8)
        self.bn2   = t.nn.BatchNorm2d(64)
        self.bn3   = t.nn.BatchNorm2d(128)

    def forward(self,x):
        x = self.relu(self.bn1(self.Conv1(x)))
        x = self.relu(self.bn2(self.Conv2(x)))
        x = self.relu(self.bn3(self.Conv3(x)))
        x = x.view(x.shape[0],-1)
        x = self.Lin1(x)
        x = self.Lin2(x)
        x = self.Lin3(x)
        return x


model = Model()
# model.load_state_dict(t.load("q_3_model.pth",map_location='cpu'))
model.to(device=device)
lr = 1e-5

epochs = 10

optimizer = t.optim.Adam(model.parameters(),lr = lr)

cel = t.nn.CrossEntropyLoss()

for epoch in range(epochs):
    lossval = 0
    correctPred = 0
    totalPred = 0
    for dataIndx , (image,labels) in tq.tqdm(enumerate(train_loader),total=len(train_loader),desc=f"Epoch [{epoch}/{epochs}]:"):
        
        image,labels = image.to(device),labels.to(device)

        optimizer.zero_grad()

        y_pred = model(image)


        loss = cel(y_pred,labels.long())

        correctPred += t.sum( 
          t.argmax(y_pred,dim=1) == labels
        )

        totalPred += len(labels)

        loss.backward() # computes gradients

        optimizer.step()
        lossval += loss.item()
    print(f"Loss: {lossval}")
    print(f"Accuracy : {correctPred/totalPred}")

t.save(model.state_dict(),"q_3_model.pth")

correctPred = 0
totalPred = 0
lossval = 0
for dataIndx , (image,labels) in tq.tqdm(enumerate(test_loader),total=len(test_loader)):

    image , labels = image.to(device),labels.to(device)

    y_pred = model(image)

    loss = cel(y_pred,labels.long())

    lossval += loss.item()

    correctPred += t.sum( 
        t.argmax(y_pred,dim=1) == labels
    )

    totalPred += len(labels)
print(f"Accuracy : {correctPred/totalPred}")
