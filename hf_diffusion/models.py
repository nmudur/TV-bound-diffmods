import torch
import numpy as np

from inspect import isfunction
from functools import partial
from torch import nn, einsum
from einops import rearrange
#import torchvision.transforms as transforms
#from torchvision.transforms import Compose

### All NN models and helper functions that are used in hf_diffusion and main

#helper functions and transforms
def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

def extract(a, t, x_shape):
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

class SinusoidalPositionEmbeddings(nn.Module):
    #embeds time in the phase
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = np.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None].float() * embeddings[None, :] #t1: [40, 1], t2: [1, 32]. Works on cpu, not on mps
        #^ is matmul: torch.allclose(res, torch.matmul(t1.float(), t2)): True when cpu
        #NM: added float for mps
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings #Bx64

class noise_model(nn.Module):
    def __init__(self, input_dim=2, hidden=64,  dim=64, time_embed_dim=256):
        super(noise_model, self).__init__()
        #time_embedding common to all blocks
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(dim),
            nn.Linear(dim, time_embed_dim),
            nn.GELU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

        # input: input_dim ---------------> output: hidden (64)
        self.C01 = nn.Linear(input_dim, hidden)
        self.C02 = nn.Linear(hidden, hidden)
        self.time_dep0 = nn.Sequential(nn.SiLU(), nn.Linear(time_embed_dim, hidden)) #TODO: ADD einsum here to go from b c to b c 1 1 
        self.B01 = nn.BatchNorm1d(hidden)

        # input: hidden ---------------> output: width2 (32)
        width2 = int(hidden/2)
        self.C11 = nn.Linear(hidden, width2)
        self.C12 = nn.Linear(width2, width2)
        self.time_dep1 = nn.Sequential(nn.SiLU(), nn.Linear(time_embed_dim, width2)) #TODO: ADD einsum here to go from b c to b c 1 1 
        self.B11 = nn.BatchNorm1d(width2)
        
        
        # input: width2 ---------------> output: width3 (16)
        width3 = int(hidden/4)
        self.C21 = nn.Linear(width2, width3)
        self.C22 = nn.Linear(width3, width3)
        self.time_dep2 = nn.Sequential(nn.SiLU(), nn.Linear(time_embed_dim, width3)) #TODO: ADD einsum here to go from b c to b c 1 1 
        self.B21 = nn.BatchNorm1d(width3)
        self.C23 = nn.Linear(width3, width3)
        
        self.final01 = nn.Linear(width3, 2)
        
        
    def forward(self, inputdata, time, labels):
        time_embedded = self.time_mlp(time)
        out = self.C01(inputdata)
        #print(inputdata.shape, len(out), out.shape, out[0].shape)
        x = torch.tanh(out)
        x = torch.tanh(self.B01(self.C02(x)))
        x = x + self.time_dep0(time_embedded)

        x = torch.tanh(self.C11(x))
        x = torch.tanh(self.B11(self.C12(x)))
        x = x + self.time_dep1(time_embedded)

        x = torch.tanh(self.C21(x))
        x = torch.tanh(self.B21(self.C22(x)))
        x = x + self.time_dep2(time_embedded)
        x = torch.tanh(self.C23(x))
        
        out = self.final01(x)
        return out
