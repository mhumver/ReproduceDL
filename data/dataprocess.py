import random
import torch
import torch.utils.data
from PIL import Image
from glob import glob
import numpy as np
import torchvision.transforms as transforms

class DataProcess(torch.utils.data.Dataset):
    def __init__(self, de_root, st_root, mask_root, opt, train=True):
        super(DataProcess, self).__init__()
        self.img_transform = transforms.Compose([
            transforms.Resize(opt.fineSize),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))
        ])
        # mask should not normalize, is just have 0 or 1
        self.mask_transform = transforms.Compose([
            transforms.Resize(opt.fineSize),
            transforms.ToTensor()
        ])
        self.Train = False
        self.opt = opt
        print(train)
        trian=True
        if train:
            
            
            self.de_paths = sorted(glob('/content/drive/My Drive/ReproductionDL/celeba_256_1000/*.jpg'))
            self.st_paths = sorted(glob(st_root+'/*', recursive=False))
            self.mask_paths = sorted(glob(mask_root+'*', recursive=True))
            
           # self.de_paths = sorted(glob('{:s}/*'.format(de_root), recursive=True))
          #  self.st_paths = sorted(glob('{:s}/*'.format(st_root), recursive=True))
         #   self.mask_paths = sorted(glob('{:s}/*'.format(mask_root), recursive=True))
            print(self.de_paths)
            print(self.st_paths)
            print(self.mask_paths)
            self.Train=True
        print('length paths')
        self.N_mask = len(self.mask_paths)
        print(len(self.de_paths))
        print(len(self.st_paths))
        print(self.N_mask)
    def __getitem__(self, index):

        de_img = Image.open(self.de_paths[index])
        st_img = Image.open(self.st_paths[index])
        mask_img = Image.open(self.mask_paths[random.randint(0, self.N_mask - 1)])
        de_img = self.img_transform(de_img.convert('RGB'))
        st_img = self.img_transform(st_img .convert('RGB'))
        mask_img = self.mask_transform(mask_img.convert('RGB'))
        return de_img, st_img, mask_img

    def __len__(self):
        return len(self.de_paths)
