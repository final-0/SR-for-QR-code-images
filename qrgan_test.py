import torch.nn as nn
import torch
import os
from torchvision.utils import save_image, make_grid 
from torch.utils.data import DataLoader
from torch.autograd import Variable
from model_qr import *
from dataset_qr import *

os.makedirs("test_img_qr", exist_ok=True)
cuda = torch.cuda.is_available()
Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor
dataloader = DataLoader(ImageDataset("../../data/qr_test/qr_test1"), batch_size = 1, shuffle=False, num_workers = 8)

if cuda:
    G = Generator().cuda()

G.load_state_dict(torch.load("saved_models_qr/generator.pth"))

for i, imgs in enumerate(dataloader):
    
    imgs_lr = Variable(imgs["lr"].type(Tensor))
    imgs_hr = Variable(imgs["hr"].type(Tensor))
  
    gen_hr = G(imgs_lr)
        
    imgs_lr = nn.functional.interpolate(imgs_lr, scale_factor=4)
    
    gen_hr = make_grid(gen_hr, nrow=1, normalize=True)
    imgs_lr = make_grid(imgs_lr, nrow=1, normalize=True)
    imgs_hr = make_grid(imgs_hr, nrow=1, normalize=True)
    img_grid = torch.cat((imgs_hr,imgs_lr, gen_hr), -1)
    save_image(img_grid, "test_img_qr/%d.png" % (i), normalize=False)
