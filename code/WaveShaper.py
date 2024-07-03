import torch
import torch.nn.functional as F
from torch import nn

class WaveShaper(nn.Module):
    def __init__(self, n):
        super(WaveShaper, self).__init__()
        self.n = n
        self.init_weights()
    
    def forward_symetric(self, x): 
        # for a signal where -1 < x < 1
        return self.forward(x) - self.forward(-x)

    def forward(self, x):
        batch = x.shape[0]
        x = torch.clamp(x, 0, 1)
        
        # Ensure stable boundary conditions
        self.params_X.data[0]=0
        self.params_X.data[-1]=1
        
        # Compute distances
        dist = torch.stack([self.params_X]*batch,dim=0)      
        dist = dist - x

        # Compute the distance to the 2 closest points
        var = torch.stack([self.params_var]*batch,dim=0)
        values, indices = torch.topk(-torch.abs(dist), 2, dim=-1)
        closest_vars = torch.gather(var, 1, indices)  # Shape [batch, 2]
        
        # Comute variance
        var_weights = 1.0 / (-values + 1e-6)  # Shape [batch, 2]
        var_weights = var_weights / var_weights.sum(dim=-1, keepdim=True)
        var = (closest_vars * var_weights).sum(dim=1,keepdim=True)
        var = 100 * F.sigmoid(var)

        # Compute rectified distance (using variance)
        dist = torch.exp(-0.5 *torch.abs(dist)*var)
        dist = dist/torch.sum(dist,dim=-1,keepdim=True)
        
        # Final Interpolation
        weights = torch.stack([self.params]*batch,dim=0)
        interpolated_x = torch.sum(dist*weights,dim=-1,keepdim=True)

        return interpolated_x

    def init_weights(self):
        # Initialize parameters with a linspace between 0 and 1
        self.params = nn.Parameter(torch.linspace(0, 1, steps=self.n))
        self.params_var = nn.Parameter(torch.ones(self.n))
        self.params_X = nn.Parameter(torch.linspace(0, 1, steps=self.n)) #[1:-1]
    
    def plot(self, axe, num_points=100):
        
        # Generate inputs and predictions
        self.eval()
        inputs = torch.linspace(0, 1, num_points).unsqueeze(1)
        predicted_output = self(inputs).detach()

        axe.plot(inputs.squeeze().numpy(), 
                 predicted_output.squeeze().numpy(), 
                 label='Predicted Output', 
                 color='red',alpha = 0.6)
        
        for indices, param, var, param_x in zip(torch.linspace(0, 1, steps=self.n),self.params,self.params_var,self.params_X):
            axe.vlines(param_x.item(), ymin=0, ymax= param.item(), color='blue', linestyle='dotted')
            axe.scatter(param_x.item(), param.item(),s=F.sigmoid(var).item()*200 , color='blue')