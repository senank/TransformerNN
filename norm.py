import torch

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Norm Constants
EPS = 1e-5
MOMENTUM = 0.1

# Normalization classes
class Norm:
    def __init__(self, dim, eps):
        self.eps = eps
        #parameters
        self.gamma = torch.ones(dim, device=DEVICE)
        self.beta = torch.zeros(dim, device=DEVICE)

    def parameters(self):
        return [self.gamma, self.beta]

class LayerNorm(Norm):
    def __init__(self, dim, eps=EPS):
        super().__init__(dim, eps)
        

    def __call__(self, x):
        # calculate the forward pass 
        xmean = x.mean(1, keepdim=True) # batch mean across T dim instead of B dim for batch 
        xvar = x.var(1, keepdim=True) # batch variance
        xhat = (x - xmean) / torch.sqrt(xvar + self.eps) # normalize to unit variance
        self.out = self.gamma * xhat + self.beta
        return self.out

class BatchNorm(Norm):
    def __init__(self, dim, eps=EPS, momentum=MOMENTUM):
        super().__init__(dim, eps)
        self.momentum = momentum
        self.training = True

        # buffers
        self.running_mean = torch.zeros(dim, device=DEVICE)
        self.running_var = torch.ones(dim, device=DEVICE)

    def __call__(self, x):
        # calculate the forward pass
        if self.training:
            xmean = x.mean(0, keepdim=True) # batch mean
            xvar = x.var(0, keepdim=True) # batch variance
        else:
            xmean = self.running_mean
            xvar = self.running_var
        xhat = (x - xmean) / torch.sqrt(xvar + self.eps) # normalize to unit variance
        self.out = self.gamma * xhat + self.beta
        
        if self.training:
            with torch.no_grad():
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * xmean
                self.running_var = (1 - self.momentum) * self.running_var + self.momentum * xvar
        return self.out