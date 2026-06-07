
import torch
from torch import nn
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, accuracy_score
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from sklearn.metrics import ( accuracy_score, precision_score, recall_score, f1_score, confusion_matrix )


RANDOM_SEED = 42
EPOCHS = 200
torch.manual_seed(RANDOM_SEED)

def data_Module():

    dataset = Planetoid(
        root='./data', 
        name='Cora', 
        transform=T.NormalizeFeatures()
    )
    data = dataset[0] 

    return dataset, data

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.gcn1  = GCNConv(1433, 16)
        self.gcn2 = GCNConv(16, 7)

    
    def forward(self, x, edge_index):

        x = F.relu(self.gcn1(x, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)

        x = self.gcn2(x, edge_index)

        return x
    
def train_model(model, data, epochs=EPOCHS):

    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    loss_fn = nn.CrossEntropyLoss()

    
    for epoch in tqdm(range(epochs)):

        optimizer.zero_grad()

        output = model(data.x, data.edge_index)

        loss = loss_fn(output[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

    return model

def evaluate(model, data, mask):

    model.eval()

    with torch.no_grad():

        logits = model(
            data.x,
            data.edge_index
        )

        preds = logits[mask].argmax(dim=1)

        y_true = data.y[mask].cpu().numpy()
        y_pred = preds.cpu().numpy()

    return accuracy_score(y_true, y_pred), precision_score(y_true, y_pred, average="macro"), recall_score(y_true, y_pred, average="macro"), f1_score(y_true, y_pred, average="macro"), confusion_matrix(y_true, y_pred)
    

dataset, data = data_Module()

model = Model()
train_model(model, data)

accuracy, precision, recall, f1, confusion = evaluate(model, data, data.val_mask)
print("Validation")
print("Accuracy :", accuracy)
print("Precision:", precision)
print("Recall   :", recall)
print("F1       :", f1)
print(confusion)

print('\n')
accuracy, precision, recall, f1, confusion = evaluate(model, data, data.test_mask)
print("Validation")
print("Accuracy :", accuracy)
print("Precision:", precision)
print("Recall   :", recall)
print("F1       :", f1)
print(confusion)
