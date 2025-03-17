import torch as t
import kagglehub as kh
from torch.utils.data import Dataset,DataLoader
from torchvision import datasets,transforms
import torchvision as tv
import torchvision.models as models
import tqdm as tq

transform = transforms.Compose([
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
path = kh.dataset_download("dimensi0n/imagenet-256")

datset = datasets.ImageFolder(root=path,transform=transform)

classes = [0,1,2,11,4,5,19,7,8,9]

subset = t.utils.data.Subset(datset,[i for i in range(len(datset)) if datset.imgs[i][1] in classes])

class_to_index = {0: 0, 1: 1, 2: 2, 11: 3, 4: 4, 5: 5, 19: 6, 7: 7, 8: 8, 9: 9}

for i in range(len(subset)):
    img_path, label = subset.dataset.imgs[subset.indices[i]]
    new_label = class_to_index[label]  # Remap label
    subset.dataset.imgs[subset.indices[i]] = (img_path, new_label)
datset = subset

trainSize = 4000
testSize = len(datset.indices) - 4000
print(len(datset.indices))

assert trainSize < len(datset)
assert testSize <= trainSize

train_idx = t.randperm( len(datset) ) [:trainSize]
indexes = t.arange(len(datset))
test_idx = indexes[~t.isin(indexes,train_idx)]

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



class Model1(t.nn.Module):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.Conv1 = t.nn.Conv2d(3,128,kernel_size=3,padding=1) 
        self.Conv2 = t.nn.Conv2d(128,256,kernel_size=3,stride=2,padding=1) 
        self.Conv3 = t.nn.Conv2d(256,256,kernel_size=3,padding=1) 
        self.Conv4 = t.nn.Conv2d(256,256,kernel_size=3,stride=2,padding=1) 
        self.Conv5 = t.nn.Conv2d(256,256,kernel_size=3,padding=1) 
        self.Conv6 = t.nn.Conv2d(256,256,kernel_size=3,stride=2,padding=1) 
        self.bn1   = t.nn.BatchNorm2d(128)
        self.bn2   = t.nn.BatchNorm2d(256)
        self.bn3   = t.nn.BatchNorm2d(256)
        self.bn4   = t.nn.BatchNorm2d(256)
        self.bn5   = t.nn.BatchNorm2d(256)
        self.relu  = t.nn.ReLU()

    def forward(self,x):
        x = self.relu(self.bn1(self.Conv1(x)))
        x = self.relu(self.bn2(self.Conv2(x)))
        x = self.relu(self.bn3(self.Conv3(x)))
        x = self.relu(self.bn4(self.Conv4(x)))
        x = self.relu(self.bn5(self.Conv5(x)))
        x = self.relu(self.Conv6(x))

        return x




class Model(t.nn.Module):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model1 = Model1()
        self.model1.load_state_dict(t.load("q_4_1_model.pth",map_location='cpu'))
        for params in self.model1.parameters():
            params.requires_grad = False
        self.Lin1  = t.nn.Linear(256*8*8,128)
        self.Lin2  = t.nn.Linear(128,64)
        self.Lin3  = t.nn.Linear(64,32)
        self.Lin4  = t.nn.Linear(32,10)


    def forward(self,x):
        x = self.model1(x)
        x = x.view(x.shape[0],-1)
        x = self.Lin1(x)
        x = self.Lin2(x)
        x = self.Lin3(x)
        x = self.Lin4(x)

        return x


model = Model()


model.to(device=device)
lr = 1e-5

epochs = 1

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

t.save(model.state_dict(),"q_4_model.pth")
# t.save(model.model1.state_dict(),"q_4_1_model.pth")
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
