import torch.nn as nn
import torch
import os
import numpy as np
from torchvision.utils import save_image, make_grid 
from torch.utils.data import DataLoader
from torch.autograd import Variable
from model_qr import *
from dataset_qr import *
from torchsummary import summary

os.makedirs("images", exist_ok=True)
os.makedirs("saved_models_qr", exist_ok=True)
cuda = torch.cuda.is_available()
Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor
dataloader = DataLoader(ImageDataset("../../data/qr_dataset"), batch_size = 1, shuffle=False, num_workers = 8)

if cuda:
    G = Generator().cuda()
    D = Discriminator((1,256,256)).cuda()
    criterion_GAN = torch.nn.MSELoss().cuda()
    criterion_content = torch.nn.L1Loss().cuda()

summary(D,(1,256,256))
summary(G,(1,64,64))

optimizer_G = torch.optim.Adam(G.parameters(), 0.0002, betas=(0.5, 0.999))
optimizer_D = torch.optim.Adam(D.parameters(), 0.0002, betas=(0.5, 0.999))

for i, imgs in enumerate(dataloader):

    imgs_lr = Variable(imgs["lr"].type(Tensor))
    imgs_hr = Variable(imgs["hr"].type(Tensor))
    valid = Variable(Tensor(np.ones((imgs_lr.size(0), *D.output_shape))), requires_grad=False)
    fake = Variable(Tensor(np.zeros((imgs_lr.size(0), *D.output_shape))), requires_grad=False)

    #--Train Generators--#

    optimizer_G.zero_grad()
    gen_hr = G(imgs_lr)

    loss_GAN = criterion_GAN(D(gen_hr), valid) + criterion_GAN(D(imgs_hr), fake)
    loss_content = criterion_content(gen_hr, imgs_hr)
    # Total loss
    loss_G = 1e-3 * (loss_GAN) + loss_content
    loss_G.backward()
    optimizer_G.step()

    #--Train Discriminator--#

    optimizer_D.zero_grad()
    loss_real = criterion_GAN(D(imgs_hr), valid)
    loss_fake = criterion_GAN(D(gen_hr.detach()), fake)
    # Total loss
    loss_D = (loss_real + loss_fake)
    loss_D.backward()
    optimizer_D.step()

    if (i+1) % 50 == 0:
        
        imgs_lr = nn.functional.interpolate(imgs_lr, scale_factor=4)
        gen_hr = make_grid(gen_hr, nrow=1, normalize=True)
        imgs_lr = make_grid(imgs_lr, nrow=1, normalize=True)
        imgs_hr = make_grid(imgs_hr, nrow=1, normalize=True)
        img_grid = torch.cat((imgs_hr,gen_hr,imgs_lr), -1)
        save_image(img_grid, "images/%d.png" % (i+1), normalize=False)

torch.save(G.state_dict(), "saved_models_qr/generator.pth")
torch.save(D.state_dict(), "saved_models_qr/discriminator.pth")
