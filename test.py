import time
import pdb
from options.test_options import TestOptions
from data.dataprocess import DataProcess
from models.models import create_model
import torchvision
from torch.utils import data
#from torch.utils.tensorboard import SummaryWriter
import os
import torch
from PIL import Image
import numpy as np
from glob import glob
from tqdm import tqdm
import torchvision.transforms as transforms
import torchvision
from torch.utils import data
from torch.utils.tensorboard import SummaryWriter


if __name__ == "__main__":

    img_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    mask_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])


    opt = TestOptions().parse()
    model = create_model(opt)
    
    
    
    param_paths = sorted(glob('/content/drive/My Drive/ReproductionDL/checkpoints/Mutual Encoder-Decoder/*.pth'), reverse = True)
    plen = len(param_paths) 
    print('len param')
    print(plen)           
    if plen>0:
            print(param_paths[2])
            print(param_paths[3])
            print(param_paths[0])
            model.netEN.load_state_dict(torch.load(param_paths[2]))
            model.netDE.load_state_dict(torch.load(param_paths[3]))
            model.netMEDFE.load_state_dict(torch.load(param_paths[0]))
           # model.netF.load_state_dict(torch.load(param_paths[1]))
           # model.netD.load_state_dict(torch.load(param_paths[4]))

            model.netEN.eval()
            model.netDE.eval()
            model.netMEDFE.eval()
           # model.netF.eval()
           # model.netD.eval()
            
                
    results_dir = '/content/drive/My Drive/ReproductionDL/checkpoints/Results'
    #if not os.path.exists( results_dir):
    #    os.mkdir(results_dir)
    de = opt.de_root+'*.jpg' #os.path.join(opt.de_root, '*.jpg')
    st = opt.st_root + '*.jpg' #os.path.join(opt.st_root, '*.jpg')
    ma = opt.mask_root+ '*.png' #os.path.join(opt.mask_root, '*.png')
    print(de)
    print(st)
    print(ma)
    de_paths = sorted(glob(de))
    st_paths = sorted(glob(st))
    mask_paths = sorted(glob(ma))
            
   # de_paths = sorted(glob('/content/drive/My Drive/ReproductionDL/celeba_256_1000/*.jpg'))
   # st_paths = sorted(glob('/content/drive/My Drive/ReproductionDL/celebastruct_256_1000/*.jpg'))
   # mask_paths = sorted(glob('/content/drive/My Drive/ReproductionDL/mask_dataset28/*.png'))
    print(len(de_paths))
    print(len(st_paths))
    print(len(mask_paths))

    writer = SummaryWriter(log_dir=results_dir, comment=opt.name)

    #mask_paths = glob('{:s}/*'.format(opt.mask_root))
    #de_paths = glob('{:s}/*'.format(opt.de_root))
    #st_path = glob('{:s}/*'.format(opt.st_root))
    image_len = len(de_paths )
    #for i in tqdm(range(image_len)):
    for i in range(1000):
         # only use one mask for all image
        path_m = mask_paths[0]
        path_d = de_paths[i]
        path_s = de_paths[i]

        mask = Image.open(path_m).convert("RGB")
        detail = Image.open(path_d).convert("RGB")
        structure = Image.open(path_s).convert("RGB")


        mask = mask_transform(mask)
        detail = img_transform(detail)
        structure = img_transform(structure)
        mask = torch.unsqueeze(mask, 0)
        detail = torch.unsqueeze(detail, 0)
        structure = torch.unsqueeze(structure,0)

        with torch.no_grad():
            model.set_input(detail, structure, mask)
            model.forward()
            fake_out = model.fake_out
            fake_out = fake_out.detach().cpu() * mask + detail*(1-mask)
            fake_image = (fake_out+1)/2.0
        #output = fake_image.detach().numpy()[0].transpose((1, 2, 0))*255
        #output = Image.fromarray(output.astype(np.uint8))
        #output.save("/content/drive/My Drive/ReproductionDL/checkpoints/Results/"+str(i)+".png")
        
        if i%10 == 0:
            input, output, GT = model.get_current_visuals()
            image_out = torch.cat([input, output, GT], 0)
            grid = torchvision.utils.make_grid(image_out)
            writer.add_image('img_(%d)' % (i), grid, i + 1)

    writer.close()
        
        
        
        
        
        
        
