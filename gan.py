from __future__ import print_function
import argparse
import os
import random
from pathlib import Path

import yaml
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset import ArtImageDataset
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--dataroot', required=False, default='/Users/m_vys/PycharmProjects/ArtGAN', help='path to dataset')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=1)
parser.add_argument('--batch_size', type=int, default=64, help='input batch size')
parser.add_argument('--image_size', type=int, default=1024, help='the height / width of the input image to network')
parser.add_argument('--latent_size', type=int, default=100, help='size of the latent z vector')
parser.add_argument('--g_depth', type=int, default=64)
parser.add_argument('--d_depth', type=int, default=64)
parser.add_argument('--epochs', type=int, default=10000, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--dry-run', action='store_true', help='check a single training cycle works')
parser.add_argument('--n_gpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--output_file', default='./models', help='folder to output images and model checkpoints')
parser.add_argument('--manualSeed', type=int, help='manual seed')


class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, d_depth, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(d_depth, d_depth * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(d_depth * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(d_depth * 2, d_depth * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(d_depth * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(d_depth * 4, d_depth * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(d_depth * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(d_depth * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        if cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)

        return output.view(-1, 1).squeeze(1)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        torch.nn.init.normal_(m.weight, 1.0, 0.02)
        torch.nn.init.zeros_(m.bias)


class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(latent_size, g_depth * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(g_depth * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(g_depth * 8, g_depth * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(g_depth * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(g_depth * 4, g_depth * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(g_depth * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(g_depth * 2, g_depth, 4, 2, 1, bias=False),
            nn.BatchNorm2d(g_depth),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(g_depth, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        if cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output


if __name__ == '__main__':
    opt = parser.parse_args()
    print(opt)
    cuda = torch.cuda.is_available()
    try:
        os.makedirs(opt.outf)
    except OSError:
        pass

    if opt.manualSeed is None:
        opt.manualSeed = random.randint(1, 10000)
    print("Random Seed: ", opt.manualSeed)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)

    cudnn.benchmark = True

    dataset = ArtImageDataset(f'/{opt.dataroot}/data/augumented_gallery/',
            transform=transforms.Compose([transforms.Resize((opt.image_size, opt.image_size)),
                                                              transforms.ToTensor(),
                                                              transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                                              ]))
    nc = 3

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size,
                                             shuffle=True)

    device = torch.device("cuda:0" if cuda else "cpu")
    n_gpu = int(opt.n_gpu)
    latent_size = int(opt.latent_size)
    g_depth = int(opt.g_depth)
    d_depth = int(opt.d_depth)

    netG = Generator(n_gpu).to(device)
    netG.apply(weights_init)
    if opt.netG != '':
        netG.load_state_dict(torch.load(opt.netG))
    print(netG)

    netD = Discriminator(n_gpu).to(device)
    netD.apply(weights_init)
    if opt.netD != '':
        netD.load_state_dict(torch.load(opt.netD))
    print(netD)

    criterion = nn.BCELoss()

    fixed_noise = torch.randn(opt.batch_size, latent_size, 1, 1, device=device)
    real_label = 1
    fake_label = 0

    # setup optimizer
    optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

    if opt.dry_run:
        opt.epochs = 1

    model_dir = f'./models/gd_{g_depth}_dd_{d_depth}_lr_{opt.lr}_ls_{latent_size}_bs_{opt.batch_size}_beta_{opt.beta1}/models/'
    img_dir = f'./models/gd_{g_depth}_dd_{d_depth}_lr_{opt.lr}_ls_{latent_size}_bs_{opt.batch_size}_beta_{opt.beta1}/created_imgs/'
    stats_dir = f'./models/gd_{g_depth}_dd_{d_depth}_lr_{opt.lr}_ls_{latent_size}_bs_{opt.batch_size}_beta_{opt.beta1}/stats/'
    config_dir = f'./models/gd_{g_depth}_dd_{d_depth}_lr_{opt.lr}_ls_{latent_size}_bs_{opt.batch_size}_beta_{opt.beta1}/config/'

    Path(model_dir).mkdir(exist_ok=True, parents=True)
    Path(img_dir).mkdir(exist_ok=True, parents=True)
    Path(stats_dir).mkdir(exist_ok=True, parents=True)
    Path(config_dir).mkdir(exist_ok=True, parents=True)

    d_config = dict(nc=nc, d_depth=d_depth, optimizer='adam')
    g_config = dict(nc=nc, g_depth=g_depth, optimizer='adam')
    d_optim_config = dict(lr=opt.lr, betas=(opt.beta1, 0.999))
    g_optim_config = dict(lr=opt.lr, betas=(opt.beta1, 0.999))
    dataset_config = dict(batch_size=opt.batch_size, image_size=opt.image_size,
                          transforms=['resize', 'normalize', 'to_tensor'])

    config_dict = dict(g_config=g_config, d_config=d_config,
                  g_optim_config=g_optim_config, d_optim_config=d_optim_config,
                  dataset_config=dataset_config)

    with open(f'{config_dir}/config.yaml', 'w') as file:
        yaml.dump(config_dict, file)
    for epoch in range(100, opt.epochs+100):
        d_loss, g_loss, d_of_x, d_of_g_z1, d_of_g_z2 = [], [], [], [], []
        for i, data in enumerate(dataloader, 0):
            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            # train with real
            netD.zero_grad()
            real_cpu = data.to(device)
            batch_size = real_cpu.size(0)
            label = torch.full((batch_size,), real_label,
                               dtype=real_cpu.dtype, device=device)

            output = netD(real_cpu)
            errD_real = criterion(output, label)
            errD_real.backward()
            D_x = output.mean().item()

            # train with fake
            noise = torch.randn(batch_size, latent_size, 1, 1, device=device)
            fake = netG(noise)
            label.fill_(fake_label)
            output = netD(fake.detach())
            errD_fake = criterion(output, label)
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            errD = errD_real + errD_fake
            optimizerD.step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            netG.zero_grad()
            label.fill_(real_label)  # fake labels are real for generator cost
            output = netD(fake)
            errG = criterion(output, label)
            errG.backward()
            D_G_z2 = output.mean().item()
            optimizerG.step()

            print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
                  % (epoch, opt.niter, i, len(dataloader),
                     errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
            g_loss.append(errG)
            d_loss.append(errD)
            d_of_x.append(D_x)
            d_of_g_z1.append(D_G_z1)
            d_of_g_z2.append(D_G_z2)
            if i % 40 == 0:
                vutils.save_image(real_cpu,
                                  '%s/real_samples.png' % img_dir,
                                  normalize=True)
                fake = netG(fixed_noise)
                vutils.save_image(fake.detach(),
                                  '%s/fake_samples_epoch_%03d.png' % (img_dir, epoch),
                                  normalize=True)

            if opt.dry_run:
                break
        # do checkpointing
        g_loss = np.array(g_loss)
        d_loss = np.array(d_loss)
        d_of_x = np.array(d_of_x)
        d_of_g_z1 = np.array(d_of_g_z1)
        d_of_g_z2 = np.array(d_of_g_z2)

        np.save(f'{stats_dir}/g_losses_{epoch}.npy', g_loss)
        np.save(f'{stats_dir}/d_losses_{epoch}.npy', d_loss)
        np.save(f'{stats_dir}/d_of_x_{epoch}.npy', d_of_x)
        np.save(f'{stats_dir}/d_of_g_z1_{epoch}.npy', d_of_g_z1)
        np.save(f'{stats_dir}/d_of_g_z2_{epoch}.npy', d_of_g_z2)

        torch.save(netG.state_dict(), '%s/netG_epoch_%d.pth' % (model_dir, epoch))
        torch.save(netD.state_dict(), '%s/netD_epoch_%d.pth' % (model_dir, epoch))