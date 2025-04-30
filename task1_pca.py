# Task 1: PCA
import matplotlib.pyplot as plt
import numpy as np
from torchvision import datasets, transforms
from sklearn.decomposition import PCA

# Load MNIST
transform = transforms.ToTensor()
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
X_train = train_dataset.data.view(-1, 28*28).numpy() / 255.0

# Task 1a: PCA to 2D for visualization
print("\n=== Task 1a: PCA 2D Visualization ===")
pca_2d = PCA(n_components=2)
X_train_2d = pca_2d.fit_transform(X_train)

plt.figure(figsize=(8, 6))
plt.scatter(X_train_2d[:, 0], X_train_2d[:, 1], c=train_dataset.targets, cmap='tab10', alpha=0.7)
plt.colorbar()
plt.title('Task 1a: PCA 2D Visualization of MNIST')
plt.show()

# Task 1b: PCA to 32D
print("\n=== Task 1b: PCA 32D Reduction and Reconstruction ===")
pca_32d = PCA(n_components=32)
X_train_32d = pca_32d.fit_transform(X_train)
X_train_reconstructed = pca_32d.inverse_transform(X_train_32d)

# Task 1c: Compute MSE loss
print("\n=== Task 1c: PCA MSE Loss Calculation ===")
pca_mse_loss = np.mean((X_train - X_train_reconstructed) ** 2)
print(f"Task 1c: PCA Reconstruction MSE Loss: {pca_mse_loss:.6f}")

# Task 1d: Visualize some reconstructed images
print("\n=== Task 1d: PCA Original vs Reconstructed Images ===")
def show_images(original, reconstructed, n=5):
    plt.figure(figsize=(10, 4))
    for i in range(n):
        plt.subplot(2, n, i + 1)
        plt.imshow(original[i].reshape(28, 28), cmap='gray')
        plt.axis('off')
        plt.subplot(2, n, i + 1 + n)
        plt.imshow(reconstructed[i].reshape(28, 28), cmap='gray')
        plt.axis('off')
    plt.show()

show_images(X_train, X_train_reconstructed)
