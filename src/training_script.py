# train_gan.py

import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from gan_model import Generator, Discriminator
import matplotlib.pyplot as plt
from utils import set_seed, check_device

def train_gan(data_path, epochs=1000, batch_size=16, latent_dim=128, img_shape=128):
    device = check_device()

    # Initialize Generator and Discriminator
    generator = Generator(latent_dim, img_shape).to(device)
    discriminator = Discriminator(img_shape).to(device)

    # Loss and optimizers
    adversarial_loss = torch.nn.BCELoss()
    optimizer_G = optim.Adam(generator.parameters(), lr=0.00005, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=0.00005, betas=(0.5, 0.999))

    # Load dataset
    dataloader = DataLoader(
        datasets.ImageFolder(data_path, transform=transforms.Compose([
            transforms.Resize(img_shape),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])),
        batch_size=batch_size, shuffle=True
    )

    for epoch in range(epochs):
        for i, (imgs, _) in enumerate(dataloader):
            real_imgs = imgs.to(device)
            valid = torch.ones(imgs.size(0), 1, device=device)
            fake = torch.zeros(imgs.size(0), 1, device=device)

            # Train Generator
            optimizer_G.zero_grad()
            z = torch.randn(imgs.size(0), latent_dim, device=device)
            gen_imgs = generator(z)
            g_loss = adversarial_loss(discriminator(gen_imgs), valid)
            g_loss.backward()
            optimizer_G.step()

            # Train Discriminator
            optimizer_D.zero_grad()
            real_loss = adversarial_loss(discriminator(real_imgs), valid)
            fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)
            d_loss = (real_loss + fake_loss) / 2
            d_loss.backward()
            optimizer_D.step()

            print(f"[Epoch {epoch}/{epochs}] [Batch {i}/{len(dataloader)}] [D loss: {d_loss.item()}] [G loss: {g_loss.item()}]")

        # Display and save generated images every 10 epochs
        if epoch % 100 == 0:
            show_generated_images(gen_imgs)
            save_generated_image(gen_imgs, epoch)

    os.makedirs("C:/Users/Alwaysgame/text_to_image_gan_project/models", exist_ok=True)
    torch.save(generator.state_dict(), "C:/Users/Alwaysgame/text_to_image_gan_project/models/gan_trained.pth")
    print("Model training complete and saved at models/gan_trained.pth")

def show_generated_images(images, num_images=5):
    images = (images[:num_images].cpu().detach() + 1) / 2.0
    images = images.permute(0, 2, 3, 1).numpy()

    plt.figure(figsize=(10, 2))    
    for i in range(num_images):
        plt.subplot(1, num_images, i + 1)
        plt.imshow(images[i])
        plt.axis('off')
    plt.show()

def save_generated_image(images, epoch, num_images=5):
    os.makedirs("generated_samples", exist_ok=True)
    images = (images[:num_images].cpu().detach() + 1) / 2.0
    images = images.permute(0, 2, 3, 1).numpy()

    fig, axs = plt.subplots(1, num_images, figsize=(15, 3))
    for i in range(num_images):
        axs[i].imshow(images[i])
        axs[i].axis('off')
    plt.savefig(f"generated_samples/sample_epoch_{epoch}.png")
    plt.close()

if __name__ == "__main__":
    data_path = "C:/Users/Alwaysgame/text_to_image_gan_project/data/processed/train_data"
    train_gan(data_path)
