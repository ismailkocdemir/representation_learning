import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import _LRScheduler

device = "cuda:0" if torch.cuda.is_available() else "cpu"

def fix_random_seeds(seed=31):
    """
    Fix random seeds.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

def calculate_new_shape(shape, factors=[1, 1/4, 1/4]):
    assert len(shape) == len(factors), 'shape and factos shoud have the same dimensionality'
    return [int(s*f) for s,f in zip(shape, factors)]

def affine_decompose(A):
    sx = (A[:,0,0].pow(2) + A[:,1,0].pow(2)).sqrt()
    sy = (A[:,1,1] * A[:,0,0] - A[:,0,1] * A[:,1,0]) / sx
    m = (A[:,0,1] * A[:,0,0] + A[:,1,0] * A[:,1,1]) / (A[:,1,1] * A[:,0,0] - A[:,0,1] * A[:,1,0])
    theta = torch.atan2(A[:,1,0] / sx, A[:,0,0] / sx)
    tx = A[:, 0, 2]
    ty = A[:, 1, 2]
    return sx, sy, m, theta, tx, ty

def construct_affine(params):
    device = params.device
    n = params.shape[0]
    sx, sy, angle = params[:,0], params[:,1], params[:,2]
    m, tx, ty = params[:,3], params[:,4], params[:,5]
    zeros = torch.zeros(n, 2, 2, device=device)
    rot = torch.stack((torch.stack((angle.cos(), -angle.sin()), dim=1), 
                       torch.stack((angle.sin(), angle.cos()), dim=1)), dim=1)
    shear = zeros.clone()
    shear[:,0,0] = 1; shear[:,1,1] = 1; shear[:,0,1] = m
    scale = zeros.clone()
    scale[:,0,0] = sx; scale[:,1,1] = sy
    A = torch.matmul(torch.matmul(rot, shear), scale)
    b = torch.stack((tx, ty), dim=1)
    theta = torch.cat((A,b[:,:,None]), dim=2)
    return theta.reshape(n, 6)

def torch_expm(A):
    """ """
    n_A = A.shape[0]
    A_fro = torch.sqrt(A.abs().pow(2).sum(dim=(1,2), keepdim=True))
    
    # Scaling step
    maxnorm = torch.Tensor([5.371920351148152]).type(A.dtype).to(A.device)
    zero = torch.Tensor([0.0]).type(A.dtype).to(A.device)
    n_squarings = torch.max(zero, torch.ceil(torch_log2(A_fro / maxnorm)))
    Ascaled = A / 2.0**n_squarings    
    n_squarings = n_squarings.flatten().type(torch.int64)
    
    # Pade 13 approximation
    U, V = torch_pade13(Ascaled)
    P = U + V
    Q = -U + V
    R, _ = torch.gesv(P, Q) # solve P = Q*R
    
    # Unsquaring step    
    n = n_squarings.max()
    res = [R]
    for i in range(n):
        res.append(res[-1].matmul(res[-1]))
    R = torch.stack(res)
    expmA = R[n_squarings, torch.arange(n_A)]
    return expmA

class WarmUpLR(_LRScheduler):
    """warmup_training learning rate scheduler
    Args:
        optimizer: optimzier(e.g. SGD)
        total_iters: totoal_iters of warmup phase
    """
    def __init__(self, optimizer, total_iters, last_epoch=-1):

        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """we will use the first m batches, and set the learning
        rate to base_lr * m / total_iters
        """
        return [base_lr * self.last_epoch / (self.total_iters + 1e-8) for base_lr in self.base_lrs]


class CenterCrop(nn.Module):
    def __init__(self, h, w):
        super(CenterCrop, self).__init__()
        self.h = h
        self.w = w
        
    def forward(self, x):
        h, w = x.shape[2:]
        x1 = int(round((h - self.h) / 2.))
        y1 = int(round((w - self.w) / 2.))
        out = x[:,:,x1:x1+self.h,y1:y1+self.w]
        return out


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()
    
    def forward(self, input):
        return input.view(input.size(0), -1)   


class BatchReshape(nn.Module):
    def __init__(self, shape):
        super(BatchReshape, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(-1, *self.shape)

