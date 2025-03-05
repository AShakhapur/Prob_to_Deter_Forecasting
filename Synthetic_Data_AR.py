import torch 
import numpy as np

def Generate_Uni_Data(N):

    x = torch.tensor(np.random.rand(N))  # Uniformly distributed random numbers between 0 and 1
    x[0] = 0  # Ensure the first element is 0
    x[-1] = 1  # Ensure the last element is 1

    # Define the nonlinear function
    nonlin_f = lambda x: 2 * torch.sin(2 * torch.pi * x)
    y = nonlin_f(x)

    # True standard deviation as a function of x, TRUE SIGMA - function of the inputs x
    std_true = x + 0.5

    # Generate random noise with standard deviation dependent on x
    d = torch.Tensor(torch.randn(N) * std_true)  # ERROR

    # Compute the true output with noise
    t_true = torch.Tensor(nonlin_f(x) + d)

    return x, t_true, nonlin_f, d, std_true, y


def oscil(xx, u=None, a=None):
    xx = torch.tensor(xx, dtype=torch.float32)
    d = xx.shape[0]
    
    if u is None:
        u = torch.full((d,), 0.5, dtype=torch.float32)
    else:
        u = torch.tensor(u, dtype=torch.float32)
    
    if a is None:
        a = torch.full((d,), 5.0, dtype=torch.float32)
    else:
        a = torch.tensor(a, dtype=torch.float32)
    
    term1 = 2 * np.pi * u[0]
    sum_term = torch.sum(a * xx)
    
    y = 0.45 * (torch.cos(term1 + sum_term) + 1.2)
    return y


def Generate_Multi_Data(N, D):
    x = torch.rand(N, D)  # Batch-first convention

    # Compute std_true: shape [N]
    # (Vectorized version of oscil)
    a = torch.full((D,), 5.0)
    term1 = 2 * np.pi * 0.5  # Fixed u[0] = 0.5
    sum_term = x @ a  # Matrix multiply [N, D] @ [D] → [N]
    std_true = 0.45 * (torch.cos(term1 + sum_term) + 1.2)

    # Generate residuals d: shape [N]
    d = torch.randn(N) * std_true

    return x, d, d, std_true
