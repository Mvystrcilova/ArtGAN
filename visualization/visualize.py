import matplotlib.pyplot as plt
import torch
from gan import Generator, Discriminator
import numpy as np
import seaborn as sns

sns.color_palette("Spectral", as_cmap=True)

def visualize_generator(model):
    print(model)
    convs = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.ConvTranspose2d):
            convs.append(module)
    for conv in convs:
        weight = conv.weight
        visualize_weight(weight)


def visualize_weight(weight):
    in_ch, out_ch, w, h = weight.shape
    in_size = in_ch
    out_size = out_ch
    if in_size < in_ch:
        random_input = np.random.randint(0, in_ch, size=in_size)
    else:
        random_input = np.arange(0, in_ch)
    if out_size < in_size:
        random_output = np.random.randint(0, out_ch, size=out_size)
    else:
        random_output = np.arange(0, out_ch)
    all_weights = np.zeros((in_ch*w, out_ch*h))
    for i, input in enumerate(random_input):
        for j, output in enumerate(random_output):
            all_weights[(i*w):(i*w)+w, j*w:(j*h)+h] = weight[i,j].detach().numpy()
    # weight = torch.flatten(weight, start_dim=2)
    max_value = np.max(np.abs(all_weights))
    ratio = in_ch/out_ch

    fig, ax = plt.subplots(figsize=(max(5, int(15/ratio)), 15))
    plt.imshow(all_weights,cmap='magma', vmin=-max_value*0.1, vmax=max_value*0.1)
    plt.axis('off')

    plt.show()
    # fig, ax = plt.subplots(in_size, out_size)
    # for i, input in enumerate(random_input):
    #     for j, output in enumerate(random_output):
    #         ax[i, j].imshow(weight[input, output].detach().numpy(), cmap='icefire', vmin=-0.02, vmax=0.02)
    #         ax[i, j].axis('off')
    #         ax[i, j].set_aspect('equal')
    # plt.subplots_adjust(wspace=0, hspace=0)
    # plt.show()


def load_generator(model_dir, laten_size, g_depth, epoch, nc):
    netG = Generator(0, latent_size=laten_size, g_depth=g_depth, nc=nc)
    netG.load_state_dict(torch.load(f'{model_dir}/netG_epoch_{epoch}.pth'))
    return netG


def load_discriminator(model_dir, epoch, d_depth, nc):
    netD = Discriminator(0, d_depth=d_depth, nc=nc)
    netD.load_state_dict(torch.load(f'{model_dir}/netD_epoch_{epoch}.pth'))
    return netD


if __name__ == '__main__':
    netG = load_generator('/Users/m_vys/PycharmProjects/ArtGAN/models', epoch=170, laten_size=100,
                          g_depth=64, nc=3)
    visualize_generator(netG)