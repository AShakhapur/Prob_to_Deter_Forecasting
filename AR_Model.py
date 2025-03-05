''' AR Model and Cost Function Definition based off of: https://arxiv.org/pdf/1811.12692 '''
''' Uses AR_Cost_Function to generate probabilistic forecasts using deterministic inputs and residuals '''

import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
import pandas as pd

from scipy.special import erf


import math


class ARNetwork(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        #self.bn1 = nn.BatchNorm1d(64)
        self.fc2 = nn.Linear(128, 64)
        #self.bn2 = nn.BatchNorm1d(16)
        self.fc3 = nn.Linear(64, 1)

        self.dropout = nn.Dropout(0.1)
        self.activation = nn.LeakyReLU(0.1)  # or nn.LeakyReLU(0.1)
        self.softplus = nn.Softplus()

        # Initialize weights
        nn.init.kaiming_normal_(self.fc1.weight, mode='fan_in', nonlinearity='leaky_relu')
        nn.init.kaiming_normal_(self.fc2.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.dropout(x)
        x = self.activation(self.fc2(x))
        x = self.fc3(x)
        #x = self.softplus(x)

        return x
    
def est_beta(d):
    
    # Calculates Scale_CRPS, CRPS_min, and RS_min given residuals

    N = d.shape[0]
    #i = np.arange(1, N+1)
    # RS_min_1 = 1/(np.sqrt(np.pi)*N)
    # Convert to PyTorch and add clamping
    quantiles = (2*torch.arange(1,N+1).float()-1)/N - 1
    quantiles = quantiles.clamp(min=-0.99, max=0.99)  # Prevent extreme values

    RS_min_2 = -1*torch.erfinv(quantiles)**2
    RS_min = torch.exp(RS_min_2).sum()/(torch.sqrt(torch.tensor(math.pi))*N)
    # CRPS from matlab code
    sigma = np.abs(d)/np.sqrt(np.log(2))+1e-6
    dum = d/sigma/np.sqrt(2)
    CRPS_min = np.nanmean(sigma*(np.sqrt(2)*dum*erf(dum)
                            + np.sqrt(2/np.pi)*np.exp(-1*dum**2) 
                            - 1/np.sqrt(np.pi)))
    CRPS_min = torch.tensor(CRPS_min)
    
    return RS_min/(RS_min+CRPS_min), CRPS_min, RS_min
    
class AR_Cost(nn.Module):
    def __init__(self, CRPS_min, RS_min):
        super(AR_Cost, self).__init__()
        self.CRPS_min = CRPS_min
        self.RS_min = RS_min

        self.sqrt_2 = torch.sqrt(torch.tensor(2.0))
        self.sqrt_pi = torch.sqrt(torch.tensor(math.pi))
        self.sqrt_2_over_pi = torch.sqrt(torch.tensor(2.0 / torch.pi))
        self.one = torch.tensor(1.0)


    def forward(self, d: torch.Tensor, curr: torch.Tensor, N: int):
      
      sigma = torch.exp(curr).squeeze() 

      x = (d / sigma) / self.sqrt_2

      erf_x = torch.erf(x)
      exp_neg_x2 = torch.exp(-x**2)

      ind = torch.argsort(x)
      ind_orig = torch.argsort(ind)+1

      CRPS_1 = self.sqrt_2*x*torch.erf(x)
      CRPS_2 = self.sqrt_2_over_pi*torch.exp(-x**2) 
      CRPS_3 = torch.tensor(1)/self.sqrt_pi                
      CRPS = sigma*(CRPS_1 + CRPS_2 - CRPS_3)

      term1 = (x / N) * (erf_x + self.one)
      term2 = x * (2.0 * ind_orig - self.one) / (N ** 2)
      term3 = exp_neg_x2 / (self.sqrt_pi * N)

      RS = N*(x/N*(torch.erf(x)+1) - x*(2*ind_orig-1)/N**2 + torch.exp(-x**2)/self.sqrt_pi/N)

      loss = (CRPS / self.CRPS_min) + (RS / self.RS_min)
      loss = torch.mean(loss)
        
      return loss + 1e-6
    