# Task 2: 1-Layer Autoencoder
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load MNIST
transform = transforms.ToTensor()
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

# Task 2: Define 1-layer Autoencoder
print("\n=== Task 2: 1-Layer Autoencoder Training ===")
class Autoencoder1Layer(nn.Module):
    def __init__(self):
        super(Autoencoder1Layer, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(784, 32),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(32, 784),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

model = Autoencoder1Layer().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Train
epochs = 20
for epoch in range(epochs):
    for data, _ in train_loader:
        data = data.view(data.size(0), -1).to(device)
        output = model(data)
        loss = criterion(output, data)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    if (epoch+1) % 5 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.6f}")

# Visualize
print("\n=== Task 2c: 1-Layer AE Original vs Reconstructed Images ===")
test_data, _ = next(iter(test_loader))
test_data_flat = test_data.view(test_data.size(0), -1).to(device)
reconstructed = model(test_data_flat).cpu().detach().numpy()

def show_images(original, reconstructed, n=5):
    plt.figure(figsize=(10, 4))
    for i in range(n):
        plt.subplot(2, n, i + 1)
        plt.imshow(original[i].cpu().numpy().reshape(28, 28), cmap='gray')
        plt.axis('off')
        plt.subplot(2, n, i + 1 + n)
        plt.imshow(reconstructed[i].reshape(28, 28), cmap='gray')
        plt.axis('off')
    plt.show()

show_images(test_data_flat, reconstructed)
