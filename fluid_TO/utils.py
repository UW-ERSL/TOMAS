"""
__author__ = "Rahul Kumar Padhy, Aaditya Chandrasekhar, and Krishnan Suresh"
__email__ = "rkpadhy@wisc.edu, cs.aaditya@gmail.com and ksuresh@wisc.edu"
"""

"""Utility functions for VAE - FLuTO."""

import numpy as np
import torch
from enum import Enum, auto


class Gpu(Enum):
  OVERRIDE = auto()
  

def to_torch(x):
  return torch.tensor(x).float()


def to_np(x):
  return x.detach().cpu().numpy()


def decompose_inv_permeability(vec: torch.tensor):
  """ Decomposes the constitutive property in 2D dimension to x, y, xy and yx components
  Args:
    vec: A vector of size (N,) whose components are to be raised to a power
  
  Returns: Four tensors of size (N,) which are components of the input tensor.
  """
  I  = torch.eye(2)
  C_00 = I[0, 0]*vec
  C_01 = I[0, 1]*vec
  C_10 = I[1, 0]*vec
  C_11 = I[1, 1]*vec
  return C_00, C_01, C_10, C_11

def rotate_2d_diagonal_matrix(P_00, P_11, theta):
  """ Rotate a 2D diagonal matrix
  
  Args:
    inv_C_00: Array of size (num_elems,) which contain the 00th elements
    inv_C_11: Array of size (num_elems,) which contain the 11th elements
    theta: Array of size (num_elems,) which are the angles (radians)by
      which the matrices are to be rotated by.
  Returns:  A tuple (rot_P_00, rot_P_11, rot_P_01) each of size (num_elems,) that
    contain the rotated components. Observe that the rotated matrix
    is symmetric and hence inv_C_01 = inv_C_10
  """
  c = torch.cos(theta)
  s = torch.sin(theta)

  rot_P_00 = (P_00)*c**2 + (P_11)*s**2
  rot_P_11 = (P_11)*c**2 + (P_00)*s**2
  rot_P_01 = (P_00)*c*s - (P_11)*c*s
  return rot_P_00, rot_P_11, rot_P_01


def generate_points_in_domain(
        nelx: int,
        nely: int,
        elem_dx: float,
        elem_dy: float,
        num_dim: int,
        resolution: int=1):
    """ Generate points inside a rectangular domain with corners at 
     [0,0] and [geometry.nelx*elem_dx, geometry.nely*elem_dy]
    Args:
      geometry.nelx: number of elements along X
      geometry.nely: number of elements along Y
      elem_dx: size of element along X
      elem_dy: size of element along Y
      resolution: number of points per element per axis

    Returns: Array of size (geometry.nelx*geometry.nely*resolution*resolution, 2) that
      contains the xy coordinates of the points within the domain
    """
    lx = elem_dx*nelx
    ly = elem_dy*nely
    [x_grid, y_grid] = np.meshgrid(np.linspace(elem_dx/2., lx-elem_dx/2., resolution*nelx),
                                   np.linspace(elem_dy/2., ly-elem_dy/2., resolution*nely))
    elem_centers = np.stack((x_grid, y_grid)).T.reshape(-1, num_dim)

    return elem_centers


def set_device(gpu_status)->torch.device:
  """
  Set the device for computation based on GPU availability and status.

  Args:
    gpu_status: Status of GPU override.

  Returns:
    torch.device: Selected device (GPU if available and status is overridden, otherwise CPU).
  """
  if(torch.cuda.is_available() and (gpu_status == Gpu.OVERRIDE) ):
    device = torch.device("cuda:0")
    print("GPU enabled")
  else:
    device = torch.device("cpu")
    print("Running on CPU")
  return device

