"""
__author__ = "Rahul Kumar Padhy, Aaditya Chandrasekhar, and Krishnan Suresh"
__email__ = "rkpadhy@wisc.edu, cs.aaditya@gmail.com and ksuresh@wisc.edu"
"""

import torch
import numpy as np
from fluid_stiffness_template import FluidStiffnessTemplates


class FluidMaterial():
  def __init__(self, elem_dx, elem_dy, mat_constants):
    self.elem_dx, self.elem_dy = elem_dx, elem_dy
    self.viscosity = mat_constants.kinematic_viscosity
    self.stiffness_templates = FluidStiffnessTemplates(elem_dx, elem_dy, mat_constants)
  
  def compute_elem_stiffness_matrix(self,
                                    C_00: torch.Tensor,
                                    C_01: torch.Tensor,
                                    C_10: torch.Tensor,
                                    C_11: torch.Tensor):
    """
    Args:
      mstr_type: Array of size (num_elems, num_mstr) in range [0,1]. value [i,j]
        corresponds to the density of microstructure j of element i
      size: Array of size (num_elems,) which are the sizes of the microstructures.
        values are in range [0,1]
      orientation: Array of size (num_elems,) which are the orientation of the
        microstructures. Values are in radians [0, pi]
    Returns:
      Aelem : Array of size (num_elems, 23, 23) which are the element
        stiffness matrices corresponding velocity  dofs
      elem_stiffness : Array of size (num_elems, 23, 23) which are the element
        stiffness matrices
    """
    Aelem = (torch.einsum('e,jk->ejk', C_00, self.stiffness_templates.Aalpha_X) +
             torch.einsum('e,jk->ejk', C_11, self.stiffness_templates.Aalpha_Y) +
              torch.einsum('e,jk->ejk', C_01, self.stiffness_templates.Aalpha_XY) + 
             self.stiffness_templates.A_mu[np.newaxis,:,:])

    elem_stiffness = (Aelem +
                      self.stiffness_templates.B[np.newaxis,:,:] +
                      self.stiffness_templates.mat_area[np.newaxis,:,:])
   
    return Aelem, elem_stiffness