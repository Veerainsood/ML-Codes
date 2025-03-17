from sklearn.datasets import make_blobs
import torch as torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

scale = StandardScaler()

size = 10000
classes = 5
X,Y = make_blobs(n_samples=size,centers=classes,random_state=0,n_features=2)

x_train , x_test, y_train, y_test = train_test_split(X,Y,test_size=.2)

x_train = scale.fit_transform(x_train,y_train)

x_test = scale.transform(x_test)

x_train = torch.Tensor(x_train)
x_test = torch.Tensor(x_test)
y_test = torch.Tensor(y_test)
y_train = torch.Tensor(y_train)

epochs = 100

mse = torch.nn.CrossEntropyLoss()

def relu(x):
    return torch.nn.functional.relu(x)

class myModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.W1 = torch.randn((2,32),requires_grad=True)
        self.B1 = torch.randn(32,requires_grad=True)

        self.W2 = torch.randn((32,128),requires_grad=True)
        self.B2 = torch.randn(128,requires_grad=True)

        self.W3 = torch.randn((128,256),requires_grad=True)
        self.B3 = torch.randn(256,requires_grad=True)

        self.W4 = torch.randn((256,classes),requires_grad=True)
        self.B4 = torch.randn(classes,requires_grad=True)

    def forward(self,x):
        x1 = relu(torch.matmul(x,self.W1)  + self.B1)
        x2 = relu(torch.matmul(x1,self.W2) + self.B2)
        x3 = relu(torch.matmul(x2,self.W3) + self.B3)
        x4 = torch.matmul(x3,self.W4) + self.B4

        return x4

model = myModel()
optimizer = torch.optim.SGD([model.W4,model.W3,model.W2,model.W1,model.B1,model.B2,model.B3,model.B4],lr=.001)

model.load_state_dict(torch.load("./Lab_2/Lab_2_1_model_weights.pth"))

for epoch in range(epochs):

    optimizer.zero_grad()

    y_pred = model(x_train)

    loss = mse(y_pred,y_train.type(torch.LongTensor))

    print(f"Loss: {loss.item()}")

    loss.backward()

    optimizer.step()

y_test_pred = model(x_test)

print(f"Loss : {mse(y_test_pred,y_test.type(torch.LongTensor)).item()}")
print(f"Accuracy: {(torch.sum(torch.argmax(y_test_pred,dim=1) == y_test)/len(y_test) )*100}%")
torch.save(model.state_dict(), "./Lab_2/Lab_2_1_model_weights.pth")
