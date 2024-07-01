"""
__author__ = "Rahul Kumar Padhy, Aaditya Chandrasekhar, and Krishnan Suresh"
__email__ = "rkpadhy@wisc.edu, cs.aaditya@gmail.com and ksuresh@wisc.edu"
"""
""" Defines input and output projection functions

"""
from enum import Enum, auto
from dataclasses import dataclass
import numpy as np
import torch
from typing import Tuple

def set_seed(seed: int = 27):
  """
  Set seed for reproducibility.

  Args:
      seed (int): Seed value for random number generators.
  """
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  np.random.seed(seed)

class FourierActivation(Enum):
  FOURIER_ON = auto()
  FOURIER_OFF = auto()
  

class SymmetryActivation(Enum):
  SYM_X_AXIS_ON = auto()
  SYM_X_AXIS_OFF = auto()
  SYM_Y_AXIS_ON = auto() 
  SYM_Y_AXIS_OFF = auto()



@dataclass
class SymParams:
  sym_x_axis_mid_pt: float
  sym_y_axis_mid_pt: float

class FourierMap:
  def __init__(self, fluid_mesh,
              fourier_map_activation: SymmetryActivation,
              num_fourier_terms: int,
              max_radius: int,
              min_radius: int,
              ):
    self.fluid_mesh = fluid_mesh
    self.fourier_map_activation = fourier_map_activation
    self.num_fourier_terms = num_fourier_terms
    self.max_radius = max_radius
    self.min_radius = min_radius
    self.fourier_map = self.compute_fourier_map()
    
  def compute_fourier_map(self, spatial_dim = 2):
    """ Compute the map for Fourier projection for 2D domain.
    Fourier projection helps in faster convergence and enables
    neural network to capture finer details.For more details see:
      Chandrasekhar, Aaditya, and Krishnan Suresh.
      "Approximate Length Scale Filter in Topology Optimization
      using Fourier Enhanced Neural Networks."
      Computer-Aided Design 150 (2022): 103277.
    Args: -
    Returns: Array of size (spatial_dim, num_fourier_terms) that
      can then be used to project coordinates from the mesh to the
      frequency space. spatial_dim is 2
    """
    set_seed(1234)
    
    coordn_map_size = (spatial_dim, self.num_fourier_terms)
    freq_sign = np.random.choice([-1.,1.], coordn_map_size)
    std_uniform = np.random.uniform(0.,1., coordn_map_size)
    wmin = 1./(2*self.max_radius*self.fluid_mesh.elem_dx)
    wmax = 1./(2*self.min_radius*self.fluid_mesh.elem_dy) # w~1/R
    wu = wmin +  (wmax - wmin)*std_uniform
    coordn_map = np.einsum('ij,ij->ij', freq_sign, wu)
    coordn_map = torch.tensor(coordn_map)
    return coordn_map

  def apply_fourier_map(self, xy):
    """ Projects the coordinates from Euclidean space to
      Fourier space.
    Args:
      xy: Array of size (num_elems, num_spatial_dim) which contain
        the Euclidean coordinates of the mesh elements
      fourier_map: Array of size (spatial_dim, num_fourier_terms) that
        is used to project coordinates from the mesh to the frequency space.
    Returns: Array of size (num_elems, 2Xnum_freq_terms) which contain
      the cos and sin terms of the Fourier terms
    """

    if(self.fourier_map_activation == FourierActivation.FOURIER_ON):
      c = torch.cos(2*np.pi*torch.einsum('ed,df->ef', xy, self.fourier_map))
      s = torch.sin(2*np.pi*torch.einsum('ed,df->ef', xy, self.fourier_map))
      xy = torch.cat((c,s), axis = 1)
    return xy
  



  

def apply_symmetry(xy: np.ndarray, sym_activation_y_axis: SymmetryActivation, 
                   sym_activation_x_axis: SymmetryActivation, symm_params: SymParams) -> np.ndarray:
  """
  Apply symmetry transformation to input coordinates.

  Args:
      xy: Input coordinates.
      sym_activation_y_axis: Activation for symmetry along y-axis.
      sym_activation_x_axis: Activation for symmetry along x-axis.
      symm_params: Symmetry parameters.

  Returns:
      np.ndarray: Transformed coordinates.
  """
  if(sym_activation_y_axis == SymmetryActivation.SYM_Y_AXIS_ON):
    xv =( symm_params.sym_y_axis_mid_pt + torch.abs( xy[:,0] - symm_params.sym_y_axis_mid_pt))
  else:
    xv = xy[:,0]
  if(sym_activation_x_axis == SymmetryActivation.SYM_X_AXIS_ON):
    yv = (symm_params.sym_x_axis_mid_pt +
              torch.abs( xy[:,1] - symm_params.sym_x_axis_mid_pt))
  else:
    yv = xy[:,1]

  xy = torch.transpose(torch.stack((xv,yv)),0,1)
  return xy

def apply_reflection(xy: np.ndarray, sym_activation_y_axis: SymmetryActivation, 
                   sym_activation_x_axis: SymmetryActivation, symm_params: SymParams)-> Tuple[torch.Tensor, dict]:
  """
  Apply reflection transformation on the given points.

  Args:
      xy: Input coordinates.
      sym_activation_y_axis: Activation for symmetry along y-axis.
      sym_activation_x_axis: Activation for symmetry along x-axis.
      symm_params: Symmetry parameters.

  Returns:
      Tuple[torch.Tensor, dict]: Transformed points after applying reflection and signs dictionary.
  """
  signs = {}
  if(sym_activation_y_axis == SymmetryActivation.SYM_Y_AXIS_ON):
    signs['Y'] =  -torch.sign(xy[:,0] - symm_params.sym_y_axis_mid_pt + 1e-6)
    xv =( symm_params.sym_y_axis_mid_pt + torch.abs( xy[:,0] - symm_params.sym_y_axis_mid_pt));
  else:
    signs['Y'] = torch.ones((xy.shape[0]))
    xv = xy[:, 0]
  if(sym_activation_x_axis == SymmetryActivation.SYM_X_AXIS_ON):
    signs['X'] = -torch.sign(xy[:,1] - symm_params.sym_x_axis_mid_pt + 1e-6)
    yv = (symm_params.sym_x_axis_mid_pt + torch.abs( xy[:,1] - symm_params.sym_x_axis_mid_pt)) ;
  else:
    signs['X'] = torch.ones((xy.shape[0]))
    yv = xy[:, 1]
  
  xy = torch.transpose(torch.stack((xv,yv)), 0, 1);
  return xy, signs