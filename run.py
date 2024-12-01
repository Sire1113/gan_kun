import nets
import datasets
import torch
import torch.nn as nn
import torch.optim as optim
import train


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
    'epochs': 500,
    'is_save_model': True,
    'is_train_from_scratch':True# Save the trained model or not
}
device = torch.device("cuda:0" if (torch.cuda.is_available() and config['ngpu'] > 0) else "cpu")
# print(f'using device: {device}')

# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


# Create the generator and discriminator
if config['is_train_from_scratch']:
    netG = nets.Generator().to(device)
    netD = nets.Discriminator().to(device)
    if (device.type == 'cuda') and (config['ngpu'] > 1):
        netG = nn.DataParallel(netG, list(range(config['ngpu'])))
    if (device.type == 'cuda') and (config['ngpu'] > 1):
        netD = nn.DataParallel(netD, list(range(config['ngpu'])))
    netG.apply(weights_init)
    netD.apply(weights_init)
else:
    netG = nets.Generator().to(device)
    netD = nets.Discriminator().to(device)
    netD.load_state_dict(torch.load('netD.pth'))
    netG.load_state_dict(torch.load('netG.pth'))

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
  train.train(config, dataloader, device, criterion, optimizerD, optimizerG, netG, netD, fixed_noise)