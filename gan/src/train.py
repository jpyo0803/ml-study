import torch
import torch.nn as nn

from config import Config
from model import Generator, Discriminator
from datasets import load_train_dataset
from torch.utils.data import DataLoader

cfg = Config()
device = cfg.device

def train():
    train_dataset = load_train_dataset()

    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=2)

    G = Generator(cfg.latent_dim, cfg.hidden_dim, cfg.img_dim).to(device)
    D = Discriminator(cfg.img_dim, cfg.hidden_dim).to(device)

    criterion = nn.BCELoss()

    optimizer_G = torch.optim.Adam(G.parameters(), lr=cfg.lr) # DCGAN says this beta settings reduces unstability
    optimizer_D = torch.optim.Adam(D.parameters(), lr=cfg.lr)

    for epoch in range(cfg.num_epochs):
        G.train()
        D.train()

        for x, y in train_loader:
            x = x.to(device)

            real_imgs = x.view(-1, cfg.img_dim) # (B, 28, 28) to (B, 28*28)
            batch_size = real_imgs.shape[0]

            real_labels = torch.ones(batch_size, 1, device=device)
            fake_labels = torch.zeros_like(real_labels)
            '''
                First train discriminator
            '''
            z = torch.randn(batch_size, cfg.latent_dim, device=device) # Sample latent space vector z from N(0, 1)
            fake_imgs = G(z).detach() # Detach fake images from Generator so that G is not updated when D is updated

            real_loss = criterion(D(real_imgs), real_labels)
            fake_loss = criterion(D(fake_imgs), fake_labels)

            D_loss = (real_loss + fake_loss) / 2

            optimizer_D.zero_grad()
            D_loss.backward()
            optimizer_D.step()
            
            '''
                Second train generator
            '''
            z = torch.randn(batch_size, cfg.latent_dim, device=device)
            generated_imgs = G(z)
            G_loss = criterion(D(generated_imgs), real_labels)

            optimizer_G.zero_grad()
            G_loss.backward()
            optimizer_G.step()

        print(f"Epoch [{epoch+1}/{cfg.num_epochs}] | D_loss: {D_loss:.4f} | G_loss: {G_loss:.4f}")

    torch.save(G.state_dict(), 'outputs/checkpoints/generator.pth')
    torch.save(D.state_dict(), 'outputs/checkpoints/discriminator.pth')
    print("Training complete")

if __name__ == '__main__':
    train()