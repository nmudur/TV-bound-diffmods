import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from .models import extract

class Diffusion():
    def __init__(self, betas):
        #NOTE: you're choosing to make loss-type an argument of p_losses and not the diffusion model class
        #you should then save it separately when saving a run
        # calculations for diffusion q(x_t | x_0) and others
        self.betas = betas
        self.alphas = 1. - betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)  # alpha_bar
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)
        # x_t = sqrt_alphas_cumprod* x_0 + sqrt_one_minus_alphas_cumprod * eps_t
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        self.timesteps = len(self.betas)

    # forward diffusion (using the nice property)
    def q_sample(self, x_start, t, noise=None):
        '''
        Here, t=0 corresponds to x1
        t=t correponds to xt+1
        t=[0, T-1] (T possible values).
        t is the index of betas
        '''
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_start.shape
        )

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
    
    def q_sample_incl_t0(self, x_start, t, noise=None):
        '''
        Here, t=0 corresponds to x0
        t=t correponds to xt
        t=[0, T] (T+1 possible values)
        t-1 is the index of betas
        '''
        if noise is None:
            noise = torch.randn_like(x_start)
        
        output = x_start.detach().clone()
        tnzmask = t!=0
        t_beta_index = t[tnzmask]-1
        
        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t_beta_index, x_start[tnzmask].shape)
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t_beta_index, x_start[tnzmask].shape)
        output[tnzmask] = sqrt_alphas_cumprod_t * x_start[tnzmask] + sqrt_one_minus_alphas_cumprod_t * noise[tnzmask]
        return output

    def p_losses(self, denoise_model, x_start, t, noise=None, loss_type="l1", labels=None):
        # L_CE <= L_VLB ~ Sum[eps_t - MODEL(x_t(x_0, eps_t), t) ]
        if noise is None:
            noise = torch.randn_like(x_start)

        x_t = self.q_sample(x_start=x_start, t=t, noise=noise)
        predicted_noise = denoise_model(x_t, t, labels)

        if loss_type == 'l1':
            loss = F.l1_loss(noise, predicted_noise)
        elif loss_type == 'l2':
            loss = F.mse_loss(noise, predicted_noise)
        elif loss_type == "huber":
            loss = F.smooth_l1_loss(noise, predicted_noise)
        else:
            raise NotImplementedError()
        return loss

    @torch.no_grad()
    def timewise_loss(self, denoise_model, x_start, t, noise=None, loss_type="l1", labels=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        x_t = self.q_sample(x_start=x_start, t=t, noise=noise)
        predicted_noise = denoise_model(x_t, t, labels)
        if loss_type == 'l1':
            loss = F.l1_loss(noise, predicted_noise, reduction='none')
        elif loss_type == 'l2':
            loss = F.mse_loss(noise, predicted_noise, reduction='none')
        elif loss_type == "huber":
            loss = F.smooth_l1_loss(noise, predicted_noise, reduction='none')
        else:
            raise NotImplementedError()
        loss = torch.mean(loss, dim=[-3, -2, -1]) #mean over all spatial dims
        return loss

    @torch.no_grad()
    def p_sample(self, model, x, t, t_index, label=None):
        betas_t = extract(self.betas, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(
            self.sqrt_one_minus_alphas_cumprod, t, x.shape
        )
        sqrt_recip_alphas_t = extract(self.sqrt_recip_alphas, t, x.shape)

        # Equation 11 in the paper
        # Use our model (noise predictor) to predict the mean
        model_output =  model(x, t, label)
        model_mean = sqrt_recip_alphas_t * (
                x - betas_t * model_output / sqrt_one_minus_alphas_cumprod_t
        )

        if t_index == 0:
            return model_mean
        else:
            posterior_variance_t = extract(self.posterior_variance, t, x.shape)
            noise = torch.randn_like(x)
            # Algorithm 2 line 4:
            return model_mean + torch.sqrt(posterior_variance_t) * noise

    # Algorithm 2 but save all images:
    @torch.no_grad()
    def p_sample_loop(self, model, shape, labels=None, return_all_timesteps=True):
        device = next(model.parameters()).device
        print('sample device', device)
        b = shape[0]
        # start from pure noise (for each example in the batch)
        img = torch.randn(shape, device=device)
        imgs = []
        if labels is not None:
            assert labels.shape[0] == shape[0]

        for i in tqdm(reversed(range(0, self.timesteps)), desc='sampling loop time step', total=self.timesteps):
            img = self.p_sample(model, img, torch.full((b,), i, device=device, dtype=torch.long), i, labels)
            if return_all_timesteps:
                imgs.append(img.cpu().numpy())
        if return_all_timesteps:
            return imgs
        else:
            return img.cpu().numpy()
    

    @torch.no_grad()
    def sample(self, model, inpt_dim, batch_size=16, labels=None, return_all_timesteps=True):
        return self.p_sample_loop(model, shape=(batch_size, inpt_dim), labels=labels, return_all_timesteps=return_all_timesteps)
