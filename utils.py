import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML
import torchvision.utils as vutils
import torch
import cv2

def plot_loss(G_losses, D_losses):
    plt.figure(figsize=(10, 5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(G_losses, label="G")
    plt.plot(D_losses, label="D")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()


def plot_animation(img_list):
    fig = plt.figure(figsize=(8, 8))
    plt.axis("off")
    ims = [[plt.imshow(np.transpose(i, (1, 2, 0)), animated=True)] for i in img_list]
    ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)
    HTML(ani.to_jshtml())


def plot_comparison(dataloader, img_list):
    # Grab a batch of real images from the dataloader
    real_batch = next(iter(dataloader))

    # Plot the real images
    plt.figure(figsize=(15, 15))
    plt.subplot(1, 2, 1)
    plt.axis("off")
    plt.title("Real Images")
    plt.imshow(
        np.transpose(vutils.make_grid(real_batch[0][:64], padding=5, normalize=True).cpu(), (1, 2, 0)))

    # Plot the fake images from the last epoch
    plt.subplot(1, 2, 2)
    plt.axis("off")
    plt.title("Fake Images")
    plt.imshow(np.transpose(img_list[-1], (1, 2, 0)))
    plt.show()


def inference(generator, noise):
    with torch.no_grad():
        # Generate a batch of images
        fake_images = generator(noise)
        plt.figure(figsize=(15, 15))
        plt.axis("off")
        plt.title("inference Images")
        plt.imshow(
            np.transpose(vutils.make_grid(fake_images, padding=5, normalize=True).cpu(), (1, 2, 0)))
        plt.show()
def inference_scaled(generator, noise):
    with torch.no_grad():
        img = generator(noise).detach().cpu().numpy()
        img = np.squeeze((img + 1) / 2)
        img = np.transpose(img, (1, 2, 0))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img = cv2.resize(img, (1024, 1024),interpolation=cv2.INTER_CUBIC)
        cv2.imshow("img", img)
        cv2.waitKey()
        cv2.destroyAllWindows()