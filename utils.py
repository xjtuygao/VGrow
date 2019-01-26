import torch
from torch.utils.data.dataset import Dataset
import torch.nn.functional as F
import numpy as np

class poolSet(Dataset):
    
    def __init__(self, p_z, p_img):
        self.len = len(p_z)
        self.z_data = p_z
        self.img_data = p_img
    
    def __getitem__(self, index):
        return self.z_data[index], self.img_data[index]
    
    def __len__(self):
        return self.len

def inceptionScore(net, netG, device, nz, nclass, batchSize=250, eps=1e-6):
    
    net.to(device)
    netG.to(device)
    net.eval()
    netG.eval()

    pyx = np.zeros((batchSize*200, nclass))

    for i in range(200):

        eval_z_b = torch.randn(batchSize, nz).to(device)
        fake_img_b = netG(eval_z_b)
        pyx[i*batchSize: (i+1)*batchSize] = F.softmax(net(fake_img_b).detach(), dim=1).cpu().numpy()

    py = np.mean(pyx, axis=0)
    
    kl = np.sum(pyx * (np.log(pyx+eps) - np.log(py+eps)), axis=1)
    kl = kl.mean()
    
    return np.exp(kl)
