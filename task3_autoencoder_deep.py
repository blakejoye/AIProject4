# Task 3: Deep Autoencoder
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

# --- Deep Autoencoder Architecture (for Task 3a) ---
# Encoder: 784 → 256 → 64 → 32
# Decoder: 32 → 64 → 256 → 784
# Activation: ReLU, Tanh, LeakyReLU

print("\n=== Task 3: Deep Autoencoder Training ===")


class DeepAutoencoder(nn.Module):
    def __init__(self, activation_fn):
        super(DeepAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(784, 256),
            activation_fn(),
            nn.Linear(256, 64),
            activation_fn(),
            nn.Linear(64, 32),
            activation_fn()
        )
        self.decoder = nn.Sequential(
            nn.Linear(32, 64),
            activation_fn(),
            nn.Linear(64, 256),
            activation_fn(),
            nn.Linear(256, 784),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


# Try different activations
activations = [nn.ReLU, nn.Tanh, nn.LeakyReLU]
best_loss = float('inf')
best_model = None
best_activation = None
criterion = nn.MSELoss()

for act_fn in activations:
    print(f"\nTraining Deep AE with {act_fn.__name__} activation...")
    model = DeepAutoencoder(act_fn).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    epochs = 20
    for epoch in range(epochs):
        for data, _ in train_loader:
            data = data.view(data.size(0), -1).to(device)
            output = model(data)
            loss = criterion(output, data)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    final_loss = loss.item()
    print(f"Final Loss with {act_fn.__name__}: {final_loss:.6f}")
    if final_loss < best_loss:
        best_loss = final_loss
        best_model = model
        best_activation = act_fn.__name__

# Best Deep AE evaluation
print(f"\n=== Task 3c: Best Deep Autoencoder Results ===")
print(f"Best Deep AE Model uses {best_activation} activation with Loss: {best_loss:.6f}")

# Visualize
print("\n=== Task 3c: Deep AE Original vs Reconstructed Images ===")
test_data, _ = next(iter(test_loader))
test_data_flat = test_data.view(test_data.size(0), -1).to(device)
reconstructed_deep = best_model(test_data_flat).cpu().detach().numpy()


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


show_images(test_data_flat, reconstructed_deep)
