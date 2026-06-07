import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets
from torchvision.transforms import transforms
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import confusion_matrix

RANDOM_SEED = 42
BATCH_SIZE = 64
EPOCHS = 20
TRAIN_SPLIT  = 0.8

torch.manual_seed(RANDOM_SEED)

def getData():

    #Image normalization
    tranform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),  
            ]
        )

    #Training dataset
    train_data = datasets.FashionMNIST(
        root="data",
        train=True,
        download=False,
        transform=tranform
    )
    #Validation dataset
    validation_data = datasets.FashionMNIST(
        root="data",
        train=False,
        download=False,
        transform=tranform
    )
    
    #Split training dataset into train and test subsets
    train_data_size = int( TRAIN_SPLIT * len(train_data.data) )
    test_data_size = len(train_data) - train_data_size
    train_data, test_data = random_split( dataset = train_data, lengths= [train_data_size, test_data_size])

    #Create dataloaders
    train_dataloader = DataLoader(train_data, batch_size=64, shuffle=True)
    valid_dataloader = DataLoader(validation_data, batch_size=64, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)

    return train_dataloader, valid_dataloader, test_dataloader

#Model:
#Input Image -> Conv1 -> Pool1 -> Conv2 -> Pool2 -> Flatten -> Linear1 -> Linear2(Output)
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        #Convolutional Layer 1
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size = 2, stride = 2, padding = 0)

        #Convolutional Layer 2
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size = 2, stride = 2, padding = 0)

        #Flatten 
        self.flatten = nn.Flatten()

        #Fully Connected Hidden Layer 1
        self.linear1 = nn.Linear(128*7*7, 128)

        #Output Layer (10 classes)
        self.linear2 = nn.Linear(128, 10)


    def forward(self, x):
        
        x = F.relu(self.conv1(x))
        x = self.pool1(x)

        x = F.relu(self.conv2(x))
        x = self.pool2(x)

        x = self.flatten(x)
        x = F.relu(self.linear1(x))

        x = self.linear2(x)

        return x

def train_loop(dataloader, model, epochs = EPOCHS):
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
    return accuracy, cm


#Load datasets
train_dataloader, valid_dataloader, test_dataloader = getData()

#Create model
model = Model()

#Train model
train_loop(train_dataloader, model)

#Validation evaluation
accuracy, cm= test_loop(valid_dataloader, model)
print(f"Test Accuracy: {accuracy:.4f}")
print("Confusion Matrix:")
print(cm)

#Final test evaluation
accuracy, cm= test_loop(test_dataloader, model)
print(f"Test Accuracy: {accuracy:.4f}")
print("Confusion Matrix:")
print(cm)


