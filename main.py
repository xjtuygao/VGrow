from __future__ import print_function

# basic functions
import argparse
import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.switch_backend('agg')

# torch functions
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils

# local functions
from network import *
from resnet import * 
from utils import poolSet, inceptionScore

#--------------------------------------------------------------------
# input arguments
parser = argparse.ArgumentParser(description='VGrow')
parser.add_argument('--divergence', '-div', type=str, default='KL', help='KL | logd | JS | Jeffrey')
parser.add_argument('--dataset', required=True, help='mnist | fashionmnist | cifar10')
parser.add_argument('--dataroot', required=True, help='path to dataset')

parser.add_argument('--gpuDevice', type=str, default='1', help='CUDA_VISIBLE_DEVICES')
parser.add_argument('--workers', type=int, default=0, help='number of data loading workers')
parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
parser.add_argument('--imageSize', type=int, default=32, help='input image size')

parser.add_argument('--nz', type=int, default=128, help='size of the latent vector')
parser.add_argument('--ngf', type=int, default=128)
parser.add_argument('--ndf', type=int, default=128)

parser.add_argument('--nEpoch', type=int, default=10000, help='maximum Outer Loops')
parser.add_argument('--nDiter', type=int, default=1, help='number of D update')
parser.add_argument('--nPiter', type=int, default=20, help='number of particle update')
parser.add_argument('--nProj', type=int, default=20, help='number of G projection')
parser.add_argument('--nPool', type=int, default=20, help='times of batch size for particle pool')
parser.add_argument('--period', type=int, default=50, help='period of saving ckpts') 

parser.add_argument('--eta', type=float, default=0.5, help='learning rate for particle update')
parser.add_argument('--lrg', type=float, default=0.0001, help='learning rate for G, default=0.0001')
parser.add_argument('--lrd', type=float, default=0.0001, help='learning rate for D, default=0.0001')
parser.add_argument('--decay_g', type=bool, default=True, help='lr_g decay')
parser.add_argument('--decay_d', type=bool, default=True, help='lr_d decay')

parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--netG', default='', help='path to netG (to continue training)')
parser.add_argument('--netD', default='', help='path to netD (to continue training)')
parser.add_argument('--outf', default='./results', help='folder to output images and model checkpoints')
parser.add_argument('--resume', type=bool, default=False, help='resume from checkpoint')
parser.add_argument('--resume_epoch', type=int, default=0)
parser.add_argument('--start_save', type=int, default=800)
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--increase_nProj', type=bool, default=True, help='increase the projection times')

opt = parser.parse_args()
print(opt)

os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpuDevice

try:
    os.makedirs(opt.outf)
except OSError:
    pass

try:
    os.mkdir('./projection_loss')
except:
    pass

if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print('Random Seed: ', opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

cudnn.benchmark = True

train_transforms = transforms.Compose([
                   transforms.Resize(opt.imageSize),
                   transforms.ToTensor(),
                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                     ])
if opt.dataset == 'mnist':
    dataset = dset.MNIST(root=opt.dataroot, download=True,
                           transform=train_transforms)
    nc = 1
    nclass = 10

elif opt.dataset == 'fashionmnist':
    dataset = dset.FashionMNIST(root=opt.dataroot, download=True,
                           transform=train_transforms)
    nc = 1
    nclass = 10

elif opt.dataset == 'cifar10':
    dataset = dset.CIFAR10(root=opt.dataroot, download=True,
                           transform=train_transforms)
    nc = 3
    nclass = 10

else:
    raise NameError

assert dataset
dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
                                         shuffle=True, num_workers=int(opt.workers))

device = torch.device('cuda:0' if torch.cuda.is_available() and not opt.cuda else 'cpu')
ngpu = int(opt.ngpu)
nz = int(opt.nz)
ngf = int(opt.ngf)
ndf = int(opt.ndf)
eta = float(opt.eta)

# nets
netG = G_resnet(nc, ngf, nz)
netD = D_resnet(nc, ndf)

netG.apply(weights_init)
netG.to(device)
netD.apply(weights_init)
netD.to(device)
print('#-----------GAN initializd-----------#')

if opt.resume:
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    state = torch.load('./checkpoint/GBGAN-%s-%s-%s-ckpt.t7' % (opt.divergence, opt.dataset, str(opt.resume_epoch)))
    netG.load_state_dict(state['netG'])
    netD.load_state_dict(state['netD'])
    start_epoch = state['epoch'] + 1
    is_score = state['is_score']
    best_is = state['best_is']
    loss_G = state['loss_G']
    print('#-----------Resumed from checkpoint-----------#')

else:
    start_epoch = 0
    is_score = []
    best_is = 0.0

netIncept = PreActResNet18(nc)
netIncept.to(device)
netIncept = torch.nn.DataParallel(netIncept)

if torch.cuda.is_available() and not opt.cuda:
    checkpoint = torch.load('./checkpoint/resnet18-%s-ckpt.t7' % opt.dataset)
    netIncept.load_state_dict(checkpoint['net'])

else:
    checkpoint = torch.load('./checkpoint/resnet18-%s-ckpt.t7' % opt.dataset, map_location=lambda storage, loc: storage)
    netIncept.load_state_dict(checkpoint['net'])

print('#------------Classifier load finished------------#')


poolSize = opt.batchSize * opt.nPool

z_b = torch.FloatTensor(opt.batchSize, nz).to(device)
img_b = torch.FloatTensor(opt.batchSize, nc, opt.imageSize, opt.imageSize).to(device)
p_z = torch.FloatTensor(poolSize, nz).to(device)
p_img = torch.FloatTensor(poolSize, nc, opt.imageSize, opt.imageSize).to(device)

show_z_b = torch.FloatTensor(64, nz).to(device)
eval_z_b = torch.FloatTensor(250, nz).to(device)

# set optimizer
optim_D = optim.RMSprop(netD.parameters(), lr=opt.lrd)
optim_G = optim.RMSprop(netG.parameters(), lr=opt.lrg)

if opt.dataset == 'mnist':
    scheduler_D = MultiStepLR(optim_D, milestones=[400, 800], gamma=0.5)
    scheduler_G = MultiStepLR(optim_G, milestones=[400, 800], gamma=0.5)

elif opt.dataset == 'fashionmnist':
    scheduler_D = MultiStepLR(optim_D, milestones=[400, 800, 1200], gamma=0.5)
    scheduler_G = MultiStepLR(optim_G, milestones=[400, 800, 1200], gamma=0.5)

elif opt.dataset == 'cifar10':
    scheduler_D = MultiStepLR(optim_D, milestones=[800, 1600, 2400], gamma=0.5)
    scheduler_G = MultiStepLR(optim_G, milestones=[800, 1600, 2400], gamma=0.5)

# set criterion
criterion_G = nn.MSELoss()

def get_nProj_mfm(epoch):
    if epoch < 200:
        nProj_t = 5
    elif 199 < epoch < 1000:
        nProj_t = 10
    elif 999 < epoch < 1500:
        nProj_t = 15
    elif 1499 < epoch:
        nProj_t = 20

    return nProj_t

def get_nProj_cf(epoch):
    if epoch < 600:
        nProj_t = 5
    elif 599 < epoch < 2000:
        nProj_t = 10
    elif 1999 < epoch < 3000:
        nProj_t = 15
    elif 2999 < epoch:
        nProj_t = 20

    return nProj_t

def get_nProj_t(epoch):
    if opt.dataset == 'mnist' or 'fashionmnist':
        nProj_t = get_nProj_mfm(epoch)
    elif opt.dataset == 'cifar10':
        nProj_t = get_nProj_cf(epoch)
    else:
        raise NameError

    return nProj_t
    
#--------------------------- main function ---------------------------#
real_show, _ = next(iter(dataloader))
vutils.save_image(real_show / 2 + 0.5, './results/real-%s.png' % opt.dataset, padding=0)

for epoch in range(start_epoch, start_epoch + opt.nEpoch):    
    # decay lr
    if opt.decay_d:
        scheduler_D.step()
    if opt.decay_g:
        scheduler_G.step()

    # input_pool
    netD.train()
    netG.eval()
    p_z.normal_()
    p_img.copy_(netG(p_z).detach())

    for t in range(opt.nPiter): 

        for _ in range(opt.nDiter):
            
            # Update D
            netD.zero_grad()
            # real
            real_img, _ = next(iter(dataloader))
            img_b.copy_(real_img.to(device))
            real_D_err = torch.log(1 + torch.exp(-netD(img_b))).mean()
            real_D_err.backward()

            # fake
            z_b_idx = random.sample(range(poolSize), opt.batchSize)
            img_b.copy_(p_img[z_b_idx])
            fake_D_err = torch.log(1 + torch.exp(netD(img_b))).mean()
            fake_D_err.backward()

            optim_D.step()

        # update particle pool            
        p_img_t = p_img.clone().to(device)

        p_img_t.requires_grad_(True)
        if p_img_t.grad is not None:
            p_img_t.grad.zero_()
        fake_D_score = netD(p_img_t)

        # set s(x)
        if opt.divergence == 'KL':
            s = torch.ones_like(fake_D_score.detach())

        elif opt.divergence == 'logd':
            s = 1 / (1 + fake_D_score.detach().exp())
            
        elif opt.divergence == 'JS':
            s = 1 / (1 + 1 / fake_D_score.detach().exp())

        elif opt.divergence == 'Jeffrey':
            s = 1 + fake_D_score.detach().exp()

        else:
            raise NameError

        s.unsqueeze_(1).unsqueeze_(2).unsqueeze_(3).expand_as(p_img_t)
        fake_D_score.backward(torch.ones(len(p_img_t)).to(device))
        p_img = torch.clamp(p_img + eta * s * p_img_t.grad, -1, 1)

    # update G
    netG.train()
    netD.eval()
    poolset = poolSet(p_z.cpu(), p_img.cpu())
    poolloader = torch.utils.data.DataLoader(poolset, batch_size=opt.batchSize, shuffle=True, num_workers=opt.workers)

    loss_G = []

    # set nProj_t
    if opt.increase_nProj:
        nProj_t = get_nProj_t(epoch)
    else:
        nProj_t = opt.nProj

    for _ in range(nProj_t):

        loss_G_t = []
        for _, data_ in enumerate(poolloader, 0):
            netG.zero_grad()

            input_, target_ = data_
            pred_ = netG(input_.to(device))
            loss = criterion_G(pred_, target_.to(device))
            loss.backward()

            optim_G.step()
            loss_G_t.append(loss.detach().cpu().item())

        loss_G.append(np.mean(loss_G_t))
        
    vutils.save_image(target_ / 2 + 0.5, './results/particle-%s-%s-%s-%s.png' 
                          % (str(epoch).zfill(4), opt.divergence, opt.dataset, str(opt.eta)), padding=0)
    print('Epoch(%s/%s)%d: %.4fe-4 | %.4fe-4 | %.4f' 
          % (opt.divergence, opt.dataset, epoch, real_D_err*10000,fake_D_err*10000, p_img_t.grad.norm(p=2)))
    
    #-----------------------------------------------------------------
    if epoch % opt.period == 0:
        fig = plt.figure()
        plt.style.use('ggplot')
        plt.plot(loss_G, label=opt.divergence)
        plt.xlabel('Loop')
        plt.ylabel('Projection Loss')
        plt.legend()
        fig.savefig('./projection_loss/projection' + str(epoch).zfill(4) + '.png')
        plt.close()

        # show image
        netG.eval()
        show_z_b.normal_()
        fake_img = netG(show_z_b)
        vutils.save_image(fake_img.detach().cpu() / 2 + 0.5, './results/fake-%s-%s-%s-%s.png' 
                          % (str(epoch).zfill(4), opt.divergence, opt.dataset, str(opt.eta)), padding=0)

        # inception score
        is_score.append(inceptionScore(netIncept, netG, device, nz, nclass))
        print('[%d] Inception Score is: %.4f' % (epoch, is_score[-1]))
        best_is = max(is_score[-1], best_is)

        fig = plt.figure()
        plt.style.use('ggplot')
        plt.plot(opt.period * (np.arange(epoch//opt.period + 1)), is_score, label=opt.divergence)
        plt.xlabel('Loop')
        plt.ylabel('Inception Score')
        plt.legend()
        fig.savefig('IS-%s-%s.png' % (opt.divergence, opt.dataset))
        plt.close()

        if best_is == is_score[-1]:
            print('Save the best Inception Score: %.4f' % is_score[-1])
        else:
            pass

    if epoch > opt.start_save and epoch % 50 == 0:
        state = {
            'netG': netG.state_dict(),
            'netD': netD.state_dict(),
            'is_score': is_score,
            'loss_G': loss_G,
            'epoch': epoch,
            'best_is': best_is
            }
        torch.save(state, './checkpoint/GBGAN-%s-%s-%s-ckpt.t7' % (opt.divergence, opt.dataset, str(epoch)))

    # save IS
    if epoch % 500 == 0:
        dataframe = pd.DataFrame({'IS-%s' % opt.divergence: is_score})
        dataframe.to_csv('is-%s-%s.csv' % (opt.divergence, opt.dataset), sep=',')



