import nets
import datasets
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils
import utils

# Config
config = {
    'batch_size': 128,
    'num_workers': 2,  # Number of workers for dataloader
    'image_size': 64,
    'nz': 100,  # Size of z latent vector (i.e. size of generator input)
    'ngf': 64,  # Size of feature maps in generator
    'ndf': 64,  # Size of feature maps in discriminator
    'lr': 0.0002,  # Learning rate for optimizers
    'ngpu': 1,  # Number of GPUs available. Use 0 for CPU mode.
    'beta1': 0.5,  # Beta1 hyperparam for Adam optimizers
    'epochs': 100,
    'is_save_model': True,  # Save the trained model or not
}
device = torch.device("cuda:0" if (torch.cuda.is_available() and config['ngpu'] > 0) else "cpu")


# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


# Create the generator and discriminator
netG = nets.Generator().to(device)
netD = nets.Discriminator().to(device)
if (device.type == 'cuda') and (config['ngpu'] > 1):
    netG = nn.DataParallel(netG, list(range(config['ngpu'])))
if (device.type == 'cuda') and (config['ngpu'] > 1):
    netD = nn.DataParallel(netD, list(range(config['ngpu'])))
netG.apply(weights_init)
netD.apply(weights_init)

# Initialize BCELoss function
criterion = nn.BCELoss()

# Create batch of latent vectors that we will use to visualize
#  the progression of the generator
fixed_noise = torch.randn(64, config['nz'], 1, 1, device=device)

# Establish convention for real and fake labels during training
real_label = 1.
fake_label = 0.

# Setup Adam optimizers for both G and D
optimizerD = optim.Adam(netD.parameters(), lr=config['lr'], betas=(config['beta1'], 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=config['lr'], betas=(config['beta1'], 0.999))

# Create the dataloader
dataloader = datasets.load_data('./data', config['image_size'], config['batch_size'])
if __name__ == '__main__':
    # Training Loop

    # Lists to keep track of progress
    img_list = []
    G_losses = []
    D_losses = []
    iters = 0

    print("Starting Training Loop...")
    # For each epoch
    for epoch in range(config['epochs']):
        # For each batch in the dataloader
        for i, data in enumerate(dataloader, 0):

            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            ## Train with all-real batch
            netD.zero_grad()
            # Format batch
            real_cpu = data[0].to(device)
            b_size = real_cpu.size(0)
            label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
            # Forward pass real batch through D
            output = netD(real_cpu).view(-1)
            # Calculate loss on all-real batch
            errD_real = criterion(output, label)
            # Calculate gradients for D in backward pass
            errD_real.backward()
            D_x = output.mean().item()

            ## Train with all-fake batch
            # Generate batch of latent vectors
            noise = torch.randn(b_size, config['nz'], 1, 1, device=device)
            # Generate fake image batch with G
            fake = netG(noise)
            label.fill_(fake_label)
            # Classify all fake batch with D
            output = netD(fake.detach()).view(-1)
            # Calculate D's loss on the all-fake batch
            errD_fake = criterion(output, label)
            # Calculate the gradients for this batch
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            # Add the gradients from the all-real and all-fake batches
            errD = errD_real + errD_fake
            # Update D
            optimizerD.step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            netG.zero_grad()
            label.fill_(real_label)  # fake labels are real for generator cost
            # Since we just updated D, perform another forward pass of all-fake batch through D
            output = netD(fake).view(-1)
            # Calculate G's loss based on this output
            errG = criterion(output, label)
            # Calculate gradients for G
            errG.backward()
            D_G_z2 = output.mean().item()
            # Update G
            optimizerG.step()

            # Output training stats
            if i % 50 == 0:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                      % (epoch, config['epochs'], i, len(dataloader),
                         errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

            # Save Losses for plotting later
            G_losses.append(errG.item())
            D_losses.append(errD.item())

            # Check how the generator is doing by saving G's output on fixed_noise
            if (iters % 500 == 0) or ((epoch == config['epochs'] - 1) and (i == len(dataloader) - 1)):
                with torch.no_grad():
                    fake = netG(fixed_noise).detach().cpu()
                img_list.append(vutils.make_grid(fake, padding=2, normalize=True))
                utils.plot_comparison(dataloader, img_list)

            iters += 1
    print("Training Complete")
    utils.plot_loss(G_losses, D_losses)
    utils.plot_animation(img_list)
    utils.plot_comparison(dataloader, img_list)
    # random test
    random_noise = torch.randn(16, config['nz'], 1, 1, device=device)
    utils.inference(netG, random_noise)
    if config['is_save_model']:
        print('Saving Model...')
        torch.save(netG.state_dict(), "netG.pth")
        torch.save(netD.state_dict(), "netD.pth")
