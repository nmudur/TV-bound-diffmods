
#https://huggingface.co/blog/annotated-diffusion

import os
import sys
import wandb
import yaml
import torch
import datetime
import numpy as np
import astropy
import shutil

import torch.nn.functional as F
from torch.optim import Adam, SGD
from torch.utils.data import Dataset, DataLoader
from torch import nn
from torchvision.transforms import Compose, ToTensor, Lambda, CenterCrop, Resize
from astropy.io import fits
import matplotlib.pyplot as plt
from functools import partial
import pickle

import hf_diffusion
from hf_diffusion import *


with open(sys.argv[1], 'r') as stream:
    config_dict = yaml.safe_load(stream)

if 'seed' in config_dict.keys():
    SEED = int(config_dict['seed'])
else:
    SEED = 23 #5 for all older runs

torch.manual_seed(SEED)
np.random.seed(SEED)

DEBUG= False

dt = datetime.datetime.now()
name = f'Run_{dt.month}-{dt.day}_{dt.hour}-{dt.minute}'

print(name)

timesteps = int(config_dict['diffusion']['timesteps'])
epochs = int(config_dict['train']['epochs'])
beta_schedule_key = config_dict['diffusion']['beta_schedule']

BATCH_SIZE = int(config_dict['train']['batch_size'])
LR = float(config_dict['train']['learning_rate'])

if torch.cuda.is_available(): 
    device = 'cuda'
else: 
    device='cpu'
print(device)

#moving to diffusion
beta_func = getattr(hf_diffusion, beta_schedule_key)
beta_args = config_dict['diffusion']['schedule_args']
beta_schedule = partial(beta_func, **beta_args)
betas = beta_schedule(timesteps=timesteps)
diffusion = Diffusion(betas)

transforms, inverse_transforms = nn.Identity(), nn.Identity()


def train(model, dataloader, optimizer, epochs, loss_type="huber", sampler=None,  resdir=None,
          misc_save_params=None, inverse_transforms=None, start_itn=0, start_epoch=0):

    '''
    #alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0) #needed where?
    #sqrt_recip_alphas = torch.sqrt(1.0 / alphas)

    #posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
    '''
    itn = start_itn
    epoch = start_epoch
    loss_spike_flg = 0
    while epoch<epochs:  # Epochs: number of full passes over the dataset
        print('Epoch: ', epoch)
        for step, batch in enumerate(dataloader):  # Step: each pass over a batch
            optimizer.zero_grad() #prevents gradient accumulation

            batch_size = batch.shape[0]
            batch = batch.to(device)

            # Algorithm 1 line 3: sample t uniformly for every example in the batch
            t = sampler.get_timesteps(batch_size, itn) #[0, T-1]
            loss = diffusion.p_losses(model, batch, t, loss_type=loss_type, labels=None)
            if sampler.type=='loss_aware':
                with torch.no_grad():
                    loss_timewise = diffusion.timewise_loss(model, batch, t, loss_type=loss_type, labels=None)
                    sampler.update_history(t, loss_timewise)
            if step % 100 == 0:
                print("Loss:", loss.item())
            if not DEBUG:
                wandb.log({"loss": loss.item(), "iter": itn, "epoch": epoch})

            loss.backward()

            optimizer.step()
            itn+=1
            if itn%4000 == 0:
                misc_save_params.update({'epoch': epoch, 'itn': itn, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()})
                torch.save(misc_save_params, resdir+f'checkpoint_{itn}.pt')
                #save samples
                if (itn%4000 == 0) and (itn>10000): #added >10k for the Nx=256 case
                    NSAMP = 5
                    samplabels   = labels[:NSAMP, :]
                    samples = diffusion.sample(model, image_size=image_size, batch_size=NSAMP,
                                               channels=misc_save_params["model_kwargs"]["channels"], labels=samplabels)
                    sampimg = np.stack([samples[-1][i].reshape(image_size, image_size) for i in range(NSAMP)])
                    invtsamples = inverse_transforms(torch.tensor(sampimg)).numpy()
                    np.save(resdir+f'samples_{itn}.npy', invtsamples)
        epoch+=1
    return itn, epoch





if __name__ == '__main__':

    weights = np.array([0.7, 0.3], dtype=np.float32)
    means = [np.array([-1, 2], dtype=np.float32), np.array([2, -1], dtype=np.float32)]
    covs = [np.diag(np.array([0.8, 0.8], dtype=np.float32)), np.diag(np.array([0.6, 0.6], dtype=np.float32))]
    print(len(weights), len(means), len(covs))
    config_dict.update({'data': {}}) 
    config_dict['data'].update({'weights': weights, 'means': means, 'covs': covs})

    if not DEBUG:
        wandb.init(project='infoth_toy', job_type='unconditional',
                       config=config_dict, name=name)

    traindata = GaussianMixtureDataset(weights, means, covs)
    dataloader = DataLoader(traindata, batch_size=BATCH_SIZE, shuffle=True)


    ### get model
    input_dim = len(means[0])
    model = noise_model(input_dim=input_dim)
    model.to(device)
    
    if config_dict['train']['optimizer']=='Adam':
        print('Reinitializing optimizer')
        optimizer = Adam(model.parameters(), lr=LR)
    else:
        raise NotImplementedError()

    #get sampler type
    sampler = TimestepSampler(timesteps=timesteps, device=device, **config_dict['diffusion']['sampler_args'])

    #sample from trained model
    resdir = f'results/samples_exps/{name}/'
    os.mkdir(resdir)
    shutil.copy(sys.argv[1], resdir+sys.argv[1][sys.argv[1].rindex('/')+1:])
    misc_save_params = {'model_type': 'noise_model',
                "schedule": beta_schedule_key, "schedule_args": beta_args, "betas": betas}
    start_itn = 0     
    start_epoch = 0 
    end_itn, end_epoch = train(model, dataloader, optimizer, epochs=epochs, loss_type=config_dict['train']['loss_type'], sampler=sampler,
          resdir=resdir, misc_save_params=misc_save_params, inverse_transforms = inverse_transforms, start_itn=start_itn, start_epoch=start_epoch)
    
    misc_save_params.update({'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'itn': end_itn, 'epoch': end_epoch})
    torch.save(misc_save_params, resdir+'model.pt')


    NSAMP=10
    samples = diffusion.sample(model, inpt_dim=input_dim, batch_size=NSAMP)

