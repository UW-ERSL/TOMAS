
"""
__author__ = "Rahul Kumar Padhy, Aaditya Chandrasekhar, and Krishnan Suresh"
__email__ = "rkpadhy@wisc.edu, cs.aaditya@gmail.com and ksuresh@wisc.edu"
"""
from dataclasses import dataclass
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

@dataclass
class NeuralNetworkParameters:
  input_dim: int
  output_dim: int
  num_layers: int
  num_neurons_per_layer: int

def set_seed(seed: int):
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  np.random.seed(seed)
  
  
class TopOptNet(nn.Module):
  def __init__(self,
              nn_params: NeuralNetworkParameters,
              seed = 77):
    self.nn_params = nn_params
    super().__init__()
    self.layers = nn.ModuleList()
    set_seed(seed)

    current_dim = self.nn_params.input_dim
    for lyr in range(self.nn_params.num_layers):
      l = nn.Linear(current_dim, self.nn_params.num_neurons_per_layer,
                    dtype = torch.float64)
      nn.init.xavier_normal_(l.weight)
      nn.init.zeros_(l.bias)
      self.layers.append(l)
      current_dim = self.nn_params.num_neurons_per_layer
    self.layers.append(nn.Linear(current_dim, self.nn_params.output_dim,
                                 dtype = torch.float64))
    
    self.bn_layer = nn.ModuleList()
    for lyr in range(self.nn_params.num_layers): 
        self.bn_layer.append(nn.BatchNorm1d(self.nn_params.num_neurons_per_layer,
                                            dtype = torch.float64))

  def forward(self, x: torch.tensor):
    """ Forward prop of the NN
    
    Args:
      x: Array of shape (num_elems, input_dim). In this case, the input_dim
        is set to the number of spatial dimensions if Fourier projection is
        off or the number of Fourier dim X 2 if the projection is on.
    Returns: A tuple containing the mstr_type, size and orientations.
      mstr_type is an array of size (num_elems, num_mstrs). The size and
      orientation are each of size (num_elems,). User ensures that the
      output_dim of the NN is set to num_mstrs+2
    """
    m = nn.LeakyReLU() 
    for ctr, layer in enumerate(self.layers[:-1]):
      x = m(self.bn_layer[ctr](layer(x)))

    nn_out = (self.layers[-1](x))
    z, theta = nn_out[:, :-1], nn_out[:, -1]
    z = -3. + 6.*torch.sigmoid(z)
    theta = torch.pi*torch.sigmoid(theta)   
    
    return z, theta


