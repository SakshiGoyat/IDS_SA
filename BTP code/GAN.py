import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

# ===========================
# 1. Load and Preprocess Data
# ===========================

def load_datasets():
    # Replace with actual file paths
    bot_iot = pd.read_csv("BoT_IoT.csv")
    iot_23 = pd.read_csv("IoT-23.csv")
    edge_iiot = pd.read_csv("Edge-IIoT.csv")
    
    # Combine datasets
    data = pd.concat([bot_iot, iot_23, edge_iiot], axis=0)

    # Drop non-numeric and categorical columns if necessary
    data = data.select_dtypes(include=[np.number])

    # Normalize features
    scaler = StandardScaler()
    data = scaler.fit_transform(data)

    return torch.tensor(data, dtype=torch.float32).unsqueeze(1)  # Add sequence dimension

# Load data
real_data = load_datasets()
dataset = TensorDataset(real_data)
dataloader = DataLoader(dataset, batch_size=128, shuffle=True)

# ===========================
# 2. Define LSTM-GAN Model
# ===========================

latent_dim = 128  # Size of the random noise vector
hidden_dim = 256  # LSTM hidden layer size
lambda_gp = 10    # Gradient penalty coefficient

# Generator with LSTM
class Generator(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Generator, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, z):
        lstm_out, _ = self.lstm(z)
        return self.fc(lstm_out)

# Discriminator with LSTM
class Discriminator(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Discriminator, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out).mean(dim=1)  # Mean pooling over time

# Get feature size from dataset
feature_size = real_data.shape[2]

# Initialize models
generator = Generator(latent_dim, hidden_dim, feature_size)
discriminator = Discriminator(feature_size, hidden_dim)

# Optimizers
optimizer_G = optim.Adam(generator.parameters(), lr=0.0001, betas=(0.5, 0.9))
optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0001, betas=(0.5, 0.9))

# Wasserstein loss
def wasserstein_loss(y_pred, y_true):
    return torch.mean(y_true * y_pred)

# Gradient Penalty Function
def gradient_penalty(discriminator, real_samples, fake_samples):
    alpha = torch.rand(real_samples.size(0), 1, 1)
    alpha = alpha.expand_as(real_samples)
    interpolates = alpha * real_samples + ((1 - alpha) * fake_samples)
    interpolates.requires_grad_(True)

    d_interpolates = discriminator(interpolates)
    grad_outputs = torch.ones_like(d_interpolates)

    gradients = torch.autograd.grad(outputs=d_interpolates,
                                    inputs=interpolates,
                                    grad_outputs=grad_outputs,
                                    create_graph=True,
                                    retain_graph=True)[0]

    gradients = gradients.view(gradients.size(0), -1)
    penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return lambda_gp * penalty

# ===========================
# 3. Training Loop
# ===========================

num_epochs = 200
critic_steps = 5  # Number of discriminator updates per generator update

for epoch in range(num_epochs):
    for real_samples in dataloader:
        real_samples = real_samples[0]  # Extract tensor from DataLoader
        batch_size = real_samples.size(0)

        # Train Discriminator (Multiple Steps)
        for _ in range(critic_steps):
            z = torch.randn(batch_size, latent_dim, 1)  # Noise input
            fake_samples = generator(z).detach()

            optimizer_D.zero_grad()
            real_loss = wasserstein_loss(discriminator(real_samples), torch.ones(batch_size, 1))
            fake_loss = wasserstein_loss(discriminator(fake_samples), -torch.ones(batch_size, 1))
            gp = gradient_penalty(discriminator, real_samples, fake_samples)

            d_loss = real_loss + fake_loss + gp  # Wasserstein loss + GP
            d_loss.backward()
            optimizer_D.step()

        # Train Generator (Once Per Discriminator Updates)
        optimizer_G.zero_grad()
        fake_samples = generator(z)
        g_loss = -wasserstein_loss(discriminator(fake_samples), torch.ones(batch_size, 1))  # Maximize Discriminator Loss

        g_loss.backward()
        optimizer_G.step()

    print(f"Epoch [{epoch+1}/{num_epochs}] | D Loss: {d_loss.item():.4f} | G Loss: {g_loss.item():.4f}")

# ===========================
# 4. Generate Synthetic Intrusion Data
# ===========================
z = torch.randn(5000, latent_dim, 1)
synthetic_data = generator(z).detach().numpy()

# Save synthetic data
pd.DataFrame(synthetic_data.squeeze()).to_csv("synthetic_intrusion_data.csv", index=False)
print("Synthetic intrusion data saved!")
