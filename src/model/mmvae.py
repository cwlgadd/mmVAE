import os
import numpy as np
import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from torch.distributions.bernoulli import Bernoulli
from torch.distributions.relaxed_bernoulli import (LogitRelaxedBernoulli, RelaxedBernoulli)
from torch.distributions.relaxed_categorical import RelaxedOneHotCategorical
from torch.utils.data.dataloader import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt

from helpers import summarise_binary_profiles, post_process, sim_mat
dir_path = os.path.dirname(os.path.realpath(__file__))

from .Layers import L0Dense, LinearPositive


class RelaxedCategoricalAutoEncoder(nn.Module):
    """
    Autoencoder with a binary latent space
    fitted via continuous relaxations of binary random variables
    """
#     # Proper scoring losses
#     @staticmethod
#     def log_loss(x, p, eps=1e-10):
#         return torch.sum(- x*torch.log(p + eps) - (1-x)*torch.log(1-p + eps ), axis=1)
        
#     @staticmethod
#     def boosting_loss(x, p, eps=1e-10):
#         t1 = x * torch.exp(0.5 * (torch.log(eps+1-p) - torch.log(p+eps)))
#         t2 = (1-x) * torch.exp(0.5 * (torch.log(p) - torch.log(1-p+eps)))
#         return torch.sum(t1 + t2, axis=1)
    
    @staticmethod
    def logit(x):
        return torch.log(x / (1-x))
    
    def FC(self, in_dim, out_dim, constrain, last=False):
        if constrain is True and last is not True:
            return LinearPositive(in_dim, out_dim, init='kaiming', bias=True)
        elif constrain is 'L0':
            return L0Dense(in_dim, out_dim, **self.L0_kwargs)
        else:
            return nn.Linear(in_dim, out_dim, bias=True)
    
    
    def __str__(self):
        s = f"Encoder network:\n {self.enc_layers}"
        s += f"\nDecoder network:\n {self.dec_layers}"
        return s
    
    
    def __init__(self, enc_h, dec_h, norm_beta=1, constrain=[False, False],  L0_kwargs={}, l1=0):
        """
        
        """
        assert (len(enc_h) > 1) and len(dec_h) > 1, "enc_h and dec_h must include both start and end dimension" 

        super().__init__()

        self.enc_con = constrain[0]
        self.dec_con = constrain[1]
        self.beta = norm_beta * enc_h[0] / enc_h[-1]
        self.L0_kwargs = L0_kwargs
        self.l1 = l1

        # Define encoder
        ################
        self.enc_layers = nn.ModuleList()
        self.enc_layers.append(self.FC(enc_h[0], enc_h[1], self.enc_con))
        for idx in range(1, len(enc_h) - 1): 
            self.enc_layers.append(nn.BatchNorm1d(enc_h[idx]))
            self.enc_layers.append(nn.ReLU())
            last = (idx == len(enc_h) - 2)
            self.enc_layers.append(self.FC(enc_h[idx], enc_h[idx + 1], self.enc_con, last=last))

        # Define decoder
        ################
        self.dec_layers = nn.ModuleList()
        self.dec_layers.append(self.FC(dec_h[0], dec_h[1], self.dec_con))
        for idx in range(1, len(dec_h) - 1):
            self.dec_layers.append(nn.BatchNorm1d(dec_h[idx]))
            self.dec_layers.append(nn.ReLU())
            last = (idx == len(dec_h) - 2)
            self.dec_layers.append(self.FC(dec_h[idx], dec_h[idx + 1], self.dec_con, last=last))
        self.dec_layers.append(nn.Sigmoid())
        
        self.mse_loss = torch.nn.MSELoss(reduction='none')
        
    def forward(self, x, temperature):
    
        # encode
        for layer in self.enc_layers:
            x = layer(x)

        # Re-parameterise
        q_z = RelaxedBernoulli(temperature, logits=x)
        logit_z = q_z.rsample()       # re-parameterised sample 
            
        # Could use relaxed samples due to underflow (See https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/RelaxedBernoulli)
        mu = q_z.probs
        
        # decoding
        for layer in self.dec_layers:
            logit_z = layer(logit_z)
        
        return logit_z, mu
            
    
    def loss_function(self, p, x, mu, eps=1e-10, kl_anneal=1):

        # Compute p(x|z) = 
        # recon_loss = torch.sum(self.log_loss(x, p))
        recon_loss = torch.sum(self.mse_loss(x, p))
        # print(f"log loss: {torch.sum(self.log_loss(x, p))} || Boosting_loss: {torch.sum(self.boosting_loss(x, p))} || MSE: {torch.sum(self.mse_loss(p, x))}") 
        
        # see Appendix B from VAE paper: Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014 ====> https://arxiv.org/abs/1312.6114
        # If the prior is the uniform distribution, the KL is the entropy 
        H = torch.sum(- mu*torch.log(mu + eps) - (1-mu)*torch.log((1-mu) + eps))
        kld_loss = -H

        # Regularisation term
        if self.l1 > 0:
            lasso_loss = self.l1 * torch.sum(torch.sum(torch.abs(mu), dim=-1))
        else:
            lasso_loss = 0
                
        loss = recon_loss + (kl_anneal * self.beta * kld_loss) + lasso_loss
        
        return loss, recon_loss, H   

    
def decode_mmVAE(z, m=None, **kwargs):
    """
    Procedure for decoding latent class labels in the Rarity clustering model.
    :param Z: The latent class labels
    :return:
    """
    z = torch.Tensor(z)
    if m is None:
        m = RelaxedCategoricalAutoEncoder(**kwargs)
        m.load_state_dict(torch.load("../scripts/"+dir_path+"/cached_model.pt"))
    m.eval()

    # decoding
    for layer in m.dec_layers:
        z = layer(z)

    
    return z.detach().numpy()


def encode_mmVAE(x, m=None, **kwargs):
    """
    Procedure for decoding latent class labels in the Rarity clustering model.
    :param Z: The latent class labels
    :return:
    """
    x = torch.Tensor(x)

    if m is None:
        m = RelaxedCategoricalAutoEncoder(**kwargs)
        m.load_state_dict(torch.load("../scripts/"+dir_path+"/cached_model.pt"))
    m.eval()

    # decoding
    for layer in m.enc_layers:
        x = layer(x)

    
    return x.detach().numpy()