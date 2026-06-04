import torch
import torch.nn as nn
from ucimlrepo import fetch_ucirepo
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import ( accuracy_score, precision_score, recall_score, f1_score, confusion_matrix )
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader


#RANDOM SEED
torch.manual_seed(42)

def getData():

    #fetch dataset 
    breast_cancer_wisconsin_diagnostic = fetch_ucirepo(id=17) 
    
    #data
    X = breast_cancer_wisconsin_diagnostic.data.features 
    y = breast_cancer_wisconsin_diagnostic.data.targets 

    #data standardization
    scaler = StandardScaler()
    scaler.set_output(transform='pandas')
    scaler.fit(X)
    X = scaler.transform(X)

    y['Diagnosis'] = y['Diagnosis'].replace('M', 1)
    y['Diagnosis'] = y['Diagnosis'].replace('B', 0)
    y = y.values.ravel().astype(int)


    #Train, test, and validation data subsets (64% Train, 16% validation, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.20, random_state=42)

    #Transform subsets in tensors
    X_train = torch.tensor(X_train.values, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)

    X_valid = torch.tensor(X_valid.values, dtype=torch.float32)
    y_valid = torch.tensor(y_valid, dtype=torch.float32)

    X_test = torch.tensor(X_test.values, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)

    #Transform tensors in TensorDataset
    train_Dataset = TensorDataset(X_train, y_train)
    valid_Dataset = TensorDataset(X_valid, y_valid)
    test_Dataset = TensorDataset(X_test, y_test)

    #Create Dataloaders for train, validation and test Dataset
    train_Batches = DataLoader(
        dataset = train_Dataset,
        batch_size = 64,
        shuffle = True
    )

    valid_Batches = DataLoader(
        dataset = valid_Dataset,
        batch_size = 64,
        shuffle = True
    )

    test_Batches = DataLoader(
        dataset = test_Dataset,
        batch_size = 64,
        shuffle = True
    )

    return train_Batches, valid_Batches, test_Batches

#Model: 30 (imput) -> 16 -> 32 -> 16 -> 1 (output)
class MLP(nn.Module):

    def __init__(self):
        super().__init__()
        #Hidden Layer 1 
        self.fc1 = nn.Linear(30,16)
        #Hidden Layer 2
        self.fc2 = nn.Linear(16,32)
        #Hidden Layer 3
        self.fc3 = nn.Linear(32,16)
        #Output 
        self.fc4 = nn.Linear(16,1)
        

    def forward(self, X):
        
        x = torch.relu(self.fc1(X))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.sigmoid(self.fc4(x))
        return x


def train_model(model : MLP, train_Batches : DataLoader, max_epochs = 200):
    model.train()
    loss_Func = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)

    for epoch in range(max_epochs):
        for features , labels in train_Batches:
            
            labels = labels.unsqueeze(1)
            optimizer.zero_grad()
            
            y = model(features)
            loss = loss_Func(y, labels)

            loss.backward()
            optimizer.step()              

    return model


def evaluate(model, valid_Data : TensorDataset):

    model.eval()

    with torch.no_grad():
        logits = model(valid_Data.tensors[0])
        y_pred = torch.where(logits >= 0.5, 1, 0).numpy().ravel()
        y_true = valid_Data.tensors[1].numpy()

        print("Accuracy  :", accuracy_score(y_true, y_pred))
        print("Precision :", precision_score(y_true, y_pred))
        print("Recall    :", recall_score(y_true, y_pred))
        print("F1-Score :", f1_score(y_true, y_pred))

        print("\nConfusion Matrix:")
        print(confusion_matrix(y_true, y_pred))


train_Batches, valid_Dataset, test_Batches = getData()
model = MLP()
train_model(model,train_Batches)
evaluate(model, valid_Dataset)