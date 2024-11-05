import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

# Define basic settings
latent_dim = 100  # Dimension of latent space
image_size = 64   # Output image size (64x64 for simplicity)
batch_size = 64
num_epochs = 100

# Define the Transformation Generator Network
class TGenerator(nn.Module):
    def __init__(self):
        super(TGenerator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, image_size * image_size * 3),
            nn.Tanh()  # Output transformation parameters for image manipulation
        )

    def forward(self, z):
        transformation = self.model(z)
        transformation = transformation.view(-1, 3, image_size, image_size)  # Reshape to image dimensions
        return transformation

# Define the Transformation Discriminator Network
class TDiscriminator(nn.Module):
    def __init__(self):
        super(TDiscriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(image_size * image_size * 3, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid()  # Outputs probability of being a real transformed image
        )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)  # Flatten image
        validity = self.model(img_flat)
        return validity

# Initialize models and set up loss function and optimizer
generator = TGenerator()
discriminator = TDiscriminator()
adversarial_loss = nn.BCELoss()

optimizer_G = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

# Load dataset (example with CIFAR-10)
transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.ToTensor(),
    transforms.Normalize([0.5] * 3, [0.5] * 3)
])
dataloader = torch.utils.data.DataLoader(datasets.CIFAR10('.', download=True, transform=transform), batch_size=batch_size, shuffle=True)

# Training loop
for epoch in range(num_epochs):
    for i, (imgs, _) in enumerate(dataloader):
        # Adversarial ground truths
        valid = torch.ones((imgs.size(0), 1), requires_grad=False)
        fake = torch.zeros((imgs.size(0), 1), requires_grad=False)

        # Train Generator
        optimizer_G.zero_grad()
        
        # Sample noise and generate transformed images
        z = torch.randn(imgs.size(0), latent_dim)
        transformed_imgs = generator(z)

        # Loss measures generator's ability to fool the discriminator
        g_loss = adversarial_loss(discriminator(transformed_imgs), valid)
        g_loss.backward()
        optimizer_G.step()

        # Train Discriminator
        optimizer_D.zero_grad()

        # Real images
        real_loss = adversarial_loss(discriminator(imgs), valid)
        # Fake images
        fake_loss = adversarial_loss(discriminator(transformed_imgs.detach()), fake)
        # Total loss
        d_loss = (real_loss + fake_loss) / 2

        d_loss.backward()
        optimizer_D.step()

    print(f"[Epoch {epoch}/{num_epochs}] [D loss: {d_loss.item()}] [G loss: {g_loss.item()}]")
