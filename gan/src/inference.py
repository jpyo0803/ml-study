import torch

from model import Generator
from config import Config
import matplotlib.pyplot as plt

cfg = Config()

@torch.no_grad()
def inference(z):
    G = Generator(cfg.latent_dim, cfg.hidden_dim, cfg.img_dim)
    G.load_state_dict(torch.load('outputs/checkpoints/generator.pth'))
    G.eval()

    generated = G(z).view(-1, 1, 28, 28).detach()
    return generated

if __name__ == '__main__':
    z = torch.randn(10, cfg.latent_dim)
    generated = inference(z)

    grid = torch.cat([img for img in generated], dim=2)
    plt.imshow(grid.squeeze().numpy(), cmap="gray")
    plt.axis("off")
    plt.show()
