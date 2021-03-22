import time
from options.train_options import TrainOptions
from data.dataprocess import DataProcess
from models.models import create_model
import torchvision
from torch.utils import data
from torch.utils.tensorboard import SummaryWriter
import os
import torch

import matplotlib.pyplot as plt

print('in if')
if __name__ == "__main__":
    print('read args')
    opt = TrainOptions().parse()
    # define the dataset
    print('define dataset')
    print(opt.de_root)
    print(opt.st_root)
    print(opt.mask_root)
    dataset = DataProcess(opt.de_root,opt.st_root,opt.mask_root,opt,opt.isTrain)
    iterator_train = (data.DataLoader(dataset, batch_size=opt.batchSize, shuffle=True, num_workers=opt.num_workers))
    # Create model
    print('create model')
    model = create_model(opt)
    total_steps=0
    # Create the logs
    dir = opt.log_dir
    dir = os.path.join(opt.log_dir, opt.name).replace('\\', '/')
    #if not os.path.exists(dir):
       # os.mkdir('/checkpoints/Mutual Encoder-Decoder.test11.txt')
     #  os.mkdir(dir)
    writer = SummaryWriter(log_dir=dir, comment=opt.name)
    # Start Training
    # for epoch in range (opt.epoch_count, opt.niter + opt.niter_decay + 1):
    amount_epochs = 5
    len_dataset = 1000
    starttimetotal = time.time()
    for epoch in range(amount_epochs):
        epoch_start_time = time.time()
        epoch_iter = 0
        for detail, structure, mask in iterator_train:
            iter_start_time = time.time()
            total_steps += opt.batchSize
            epoch_iter += opt.batchSize
            model.set_input(detail, structure, mask)
            model.optimize_parameters()
            # display the training processing
            if total_steps % 200 == 0: #dispfreq 10
                #print('display')
                input, output, GT = model.get_current_visuals()
                image_out = torch.cat([input, output, GT], 0)
                grid = torchvision.utils.make_grid(image_out)
#                plt.imshow(grid)

                #image = image.reshape(80,80)
                """
                for i in range(3):
                    plt.imshow(grid[i])

                    for j in range(3):
                        plt.matshow(image_out[i][j])
                        plt.show() 
                """
                
                writer.add_image('Epoch_(%d)_(%d)' % (epoch, total_steps + 1), grid, total_steps + 1)
            # display the training loss
            if total_steps % opt.print_freq == 0:  #printfreq 50
                
                errors = model.get_current_errors()
                t = (time.time() - iter_start_time)
                #t = (time.time() - iter_start_time) / opt.batchSize
                writer.add_scalar('G_GAN', errors['G_GAN'], total_steps + 1)
                writer.add_scalar('G_L1', errors['G_L1'], total_steps + 1)
                writer.add_scalar('G_stde', errors['G_stde'], total_steps + 1)
                writer.add_scalar('D_loss', errors['D'], total_steps + 1)
                writer.add_scalar('F_loss', errors['F'], total_steps + 1)
                print('iteration time: %g; step: %d / %d' %( t, total_steps, len_dataset*amount_epochs ))
                print('time left : %g seconds' %(t*(len_dataset*amount_epochs-total_steps)))
                
        if (epoch % 1) == 0: #epsavefreq 2
            print('saving the model at the end of epoch %d, iters %d' %
                  (epoch, total_steps))
            model.save_networks(epoch)
        print('End of epoch %d / %d \t Time Taken: %d sec' %
              (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
        model.update_learning_rate()
    writer.close()
    print('total time: %g' %(time.time()-starttimetotal))
    os.system('python -m tensorflow.tensorboard --logdir=' + dir)

