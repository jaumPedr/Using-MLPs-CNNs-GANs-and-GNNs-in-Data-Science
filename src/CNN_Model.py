import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets
from torchvision.transforms import ToTensor
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np

#RANDOM SEED
torch.manual_seed(42)

def getData():
    train_data = datasets.FashionMNIST(
        root="data",
        train=True,
        download=False,
        transform=ToTensor()
    )

    test_data = datasets.FashionMNIST(
        root="data",
        train=False,
        download=False,
        transform=ToTensor()
    )
    
    train_data_size = int( 0.8 * len(train_data.data) )
    valid_data_size = len(train_data) - train_data_size

    train_data, validation_data = random_split( dataset = train_data, lengths= [train_data_size, valid_data_size])


    #image size 28x28x1, 10 classes (0 - 9)
    train_dataloader = DataLoader(train_data, batch_size=64, shuffle=True)
    valid_dataloader = DataLoader(validation_data, batch_size=64, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)

    return train_dataloader, valid_dataloader, test_dataloader

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size = 2, stride = 2, padding = 0)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size = 2, stride = 2, padding = 0)

        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(64*7*7, 10)


    def forward(self, x):
        
        x = F.relu(self.conv1(x))
        x = self.pool1(x)

        x = F.relu(self.conv2(x))
        x = self.pool2(x)

        x = self.flatten(x)
        x = self.linear1(x)

        return x

def train_loop(dataloader, model, epochs = 40):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    model.train()
    for epoch in range(epochs):
        for batch, (X,y) in enumerate(tqdm(dataloader, desc = f"Epoch {epoch+1}/{epochs}")):
            pred = model(X)

            loss_fn = nn.CrossEntropyLoss()
            Loss = loss_fn(pred, y)

            optimizer.zero_grad()
            Loss.backward()
            optimizer.step()


def test_loop(dataloader, model):
    model.eval()
    correct, total = 0, 0
    y_true, y_pred = [], []
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            predicted_labels = torch.argmax(pred, dim=1)
            y_true.extend(y.cpu().numpy())
            y_pred.extend(predicted_labels.cpu().numpy())
            correct += (predicted_labels == y).sum().item()
            total += y.size(0)
    accuracy = correct / total if total > 0 else 0
    cm = confusion_matrix(y_true, y_pred)
    last_X = X [0,0]
    last_pred = y_pred [0]
    return accuracy, cm, last_X, last_pred

train_dataloader, valid_dataloader, test_dataloader = getData()
model = Model()
train_loop(train_dataloader, model)
accuracy, cm, last_X, last_pred = test_loop(test_dataloader, model)
print(f"Test Accuracy: {accuracy:.4f}")
print("Confusion Matrix:")
print(cm)

#plt.figure(figsize=(10, 7))
#plt.imshow(last_X, interpolation='nearest', cmap=plt.cm.Blues)
#plt.title(last_pred)
#plt.colorbar()
#plt.show()

