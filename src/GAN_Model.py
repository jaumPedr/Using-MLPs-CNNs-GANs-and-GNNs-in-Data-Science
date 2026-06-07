import torchvision.datasets as datasets
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import transforms
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt


RANDOM_SEED = 42
BATCH_SIZE = 64
EPOCHS = 100
DATASET_PATH = './data'

torch.manual_seed(RANDOM_SEED)

def data_Module():
    
    #Image normalization to [-1,1]
    tranform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),  
        ]
    )

    #MNIST training dataset
    train_data = datasets.MNIST(
        root=DATASET_PATH, 
        train=True, 
        transform=tranform, 
        download=True
    )

    train_dataloader = DataLoader(train_data, batch_size=64, shuffle=True)


    return train_dataloader

# Generator Network
# Input Noise (100) -> Linear -> Reshape -> ConvTranspose1 -> ConvTranspose2 -> Generated Image (28x28)
class Generator(nn.Module):
    def __init__(self, latent_dim = 100):
        super().__init__()

        self.linear1 = nn.Linear(latent_dim, 128*7*7)
        self.convTrans1 = nn.ConvTranspose2d(128, 64, kernel_size = 4, stride=2, padding=1)
        self.convTrans2 = nn.ConvTranspose2d(64, 1, kernel_size = 4, stride=2, padding=1)

    
    def forward(self, x):
        
        x = F.relu(self.linear1(x))
        x = x.view(-1, 128, 7, 7)

        x = F.relu(self.convTrans1(x))

        x = torch.tanh(self.convTrans2(x))

        return x

# Discriminator Network
# Input Image -> Conv1 -> Pool1 -> Conv2 -> Pool2 -> Flatten -> Linear -> Real/Fake Score
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size = 2, stride = 2, padding = 0)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size = 2, stride = 2, padding = 0)

        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(32*7*7, 1)


    def forward(self, x):
        
        x = F.relu(self.conv1(x))
        x = self.pool1(x)

        x = F.relu(self.conv2(x))
        x = self.pool2(x)

        x = self.flatten(x)
        x = self.linear1(x)

        return x



def loss_Generator(discriminator, generator, generator_optimizer, batch_size):
    noise_data = torch.randn(batch_size, 100)
    target = torch.ones(batch_size, 1)
    
    generated_images = generator(noise_data)

    discriminator_pred = discriminator(generated_images)

    loss_fn = nn.BCEWithLogitsLoss()
    Loss = loss_fn(discriminator_pred, target)

    generator_optimizer.zero_grad()
    Loss.backward()
    generator_optimizer.step()
    return Loss

def loss_Discriminator(discriminator, generator, discriminator_optimizer, image):
    batch_size = image.size(0)

    real_images_target = torch.ones(batch_size, 1)
    fake_images_target = torch.zeros(batch_size, 1)
    
    noise_data = torch.randn(batch_size, 100)
    generated_images = generator(noise_data).detach()

    pred_real_images = discriminator(image)
    pred_fake_images = discriminator(generated_images)

    loss_fn = nn.BCEWithLogitsLoss()
    loss_real_images = loss_fn(pred_real_images, real_images_target)
    loss_fake_images = loss_fn(pred_fake_images, fake_images_target)

    Loss = loss_real_images + loss_fake_images

    discriminator_optimizer.zero_grad()
    Loss.backward()
    discriminator_optimizer.step()
    return Loss

#Display generated images
def plot_generated_images(generator, num_images=16):
    generator.eval()

    with torch.no_grad():
        noise = torch.randn(num_images, 100)
        fake_images = generator(noise)

    fig, axes = plt.subplots(4, 4, figsize=(8, 8))

    for i, ax in enumerate(axes.flat):
        image = fake_images[i].squeeze().cpu().numpy()
        image = image * 0.5 + 0.5

        ax.imshow(image, cmap="gray", vmin=0, vmax=1)
        ax.axis("off")

    plt.tight_layout()
    plt.show()

    generator.train()

#GAN training loop
def train_loop(dataloader, discriminator, generator, epochs = EPOCHS):
    discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    generator_optimizer = torch.optim.Adam(generator.parameters(), lr=0.0002,betas=(0.5, 0.999))
    middle_epoch = epochs // 2
    discriminator.train()
    generator.train()

    for epoch in range(epochs):
        for batch, (X, y) in enumerate(tqdm(dataloader, desc = f"Epoch {epoch+1}/{epochs}")):
            
            #Update discriminator
            loss_disc = loss_Discriminator(discriminator, generator, discriminator_optimizer, X)
            
            #Update generator
            loss_gen = loss_Generator(discriminator, generator, generator_optimizer, X.size(0))
        
        #Visualize generator progress
        if epoch in [0, middle_epoch, epochs - 1]:
            plot_generated_images(generator)

#Evaluate discriminator against generated images
def test_loop(discriminator, generator):
    discriminator.eval()
    generator.eval()

    with torch.no_grad():
        noise = torch.randn(BATCH_SIZE, 100)

        fake_images = generator(noise)
        logits = discriminator(fake_images)

        probs = torch.sigmoid(logits)
        predictions = (probs >= 0.5).float()

        targets = torch.zeros_like(predictions)

        accuracy = ((predictions == targets).float().mean().item())
        cm = confusion_matrix(targets.cpu().numpy().ravel(),predictions.cpu().numpy().ravel())

    return accuracy, cm, fake_images 

#Load dataset
train_dataloader = data_Module()

#Create GAN models
gen = Generator()
dis = Discriminator()
train_loop(train_dataloader, dis, gen)

#Final evaluation
accuracy, cm, fake_images = test_loop(dis, gen)
print(f"Test Accuracy: {accuracy:.4f}")
print("Confusion Matrix:")
print(cm)

#Display generated samples
fig, axes = plt.subplots(4, 4, figsize=(8, 8))

for i, ax in enumerate(axes.flat):

    image = fake_images[i].squeeze().cpu().numpy()
    image = image * 0.5 + 0.5

    ax.imshow(image, cmap="gray", vmin=0, vmax=1)
    ax.axis("off")

plt.tight_layout()
plt.show()