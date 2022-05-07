import glob
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as tr

#--Variable--#
K = 7    #kernel size
S = 1.2  #sigma

class ImageDataset(Dataset):
    def __init__(self, root):
        self.trans = tr.Compose([tr.ToTensor(), tr.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225]), tr.Grayscale()])
        self.lr_tr = tr.Compose([tr.Resize((256,256), Image.BICUBIC), tr.Resize((64,64), Image.BICUBIC), tr.GaussianBlur(kernel_size=K, sigma=S)])
        self.hr_tr = tr.Compose([tr.Resize((256,256), Image.BICUBIC)])
        self.files = sorted(glob.glob(root + "/*.*"))
        
    def __getitem__(self, index):
        img = Image.open(self.files[index % len(self.files)])
        img_lr = self.trans(self.lr_tr(img))
        img_hr = self.trans(self.hr_tr(img))
        return {"hr": img_hr, "lr": img_lr}

    def __len__(self):
        return len(self.files)