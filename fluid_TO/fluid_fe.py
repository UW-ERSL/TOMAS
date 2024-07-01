"""
__author__ = "Rahul Kumar Padhy, Aaditya Chandrasekhar, and Krishnan Suresh"
__email__ = "rkpadhy@wisc.edu, cs.aaditya@gmail.com and ksuresh@wisc.edu"
"""

import numpy as np
import utils
import torch
import matplotlib.pyplot as plt
import fluid_material
from torch_sparse_solve import solve


from fluid_mesher import Q2Q1Mesh

class FluidSolver:
  def __init__(self, mesh: Q2Q1Mesh, bc:np.ndarray, fixture_const = 1e15):

    self.mesh = mesh
    self.bc=bc
    
    self.velocity_pressure_field=np.zeros((self.mesh.num_total_dofs))
    V = np.zeros((self.mesh.num_total_dofs, self.mesh.num_total_dofs));
    V[self.bc.fixed_dofs, self.bc.fixed_dofs] = 1.
    V = torch.tensor(V[np.newaxis])
    indices = torch.nonzero(V).t()
    values = V[indices[0], indices[1], indices[2]] # modify this based on dimensionality    
    self.fixed_BC_penalty_matrix = \
        fixture_const*torch.sparse_coo_tensor(indices, values, V.size())
    self.bc.dir_BC = self.bc.dir_BC*fixture_const
    self.f = torch.tensor(self.bc.dir_BC).unsqueeze(0).unsqueeze(2)
    
    
  def assemble_stiffness_matrix(self,
  elem_stiffness_matrix: torch.DoubleTensor, Aelem: torch.DoubleTensor):
  
    """
    *--*--*
    |     |
    *  *  *       
    |     |
    *--*--*
    velocity : * denotes dofs (18)
    *-----*
    |     |
    |     |
    *-----*
    pressure : * denotes dofs (4)
    
    Args:
      elem_stiffness_matrix: Matrix of size (num_elems, 23, 23) where 23 corresponds
        to the number of dofs per element of quadratic velocity, 
        dofs per element of bilinear pressure and one dof for lagrange multiplier.
    Returns: assembled_stiffness_matrix: A torch sparse coo tensor that contains 
       the assembled global stiffness matrix
      assembled_A_matrix : A torch sparse coo tensor that contains 
         the assembled velocity stiffness matrix
    """

    num_velocity_dofs = self.mesh.num_velocity_dofs
    total_dofs = self.mesh.num_total_dofs

    assembled_A_matrix =  torch.sparse_coo_tensor(
                  self.mesh.node_idx_a,
                  Aelem[:,0:18,0:18].flatten(),
                  (num_velocity_dofs, num_velocity_dofs))  

    assembled_stiffness_matrix = torch.sparse_coo_tensor(
                  self.mesh.node_idx,
                  elem_stiffness_matrix.flatten(), \
                  (1, total_dofs, total_dofs))
    
    assembled_stiffness_matrix = (assembled_stiffness_matrix +
                                  self.fixed_BC_penalty_matrix).coalesce()

    return assembled_stiffness_matrix, assembled_A_matrix
  
  

  def simulate_fluid_system(self, assembled_stiffness_matrix):
    """
    Args:
      assembled_stiffness_matrix: A torch sparse coo tensor that contains 
         the assembled global stiffness matrix
    Returns: 
      velocity_field: An array velocity_field of size 
      (2(2nelx+1)(2nelx+1)) which contain the velocity. 
      Arrangement of dofs: u1 v1 u2 v2 u3 v3 ... un vn 
      An array velocity_pressure_field of size 
      pressure_field: An array velocity_field of size 
      ((nelx+1)(nelx+1)) which contain the pressure. 
      Arrangement of dofs: p1 p2 p3 ...pn
      An array velocity_pressure_field of size 
    (2(2nelx+1)(2nelx+1)+(nelx+1)(nelx+1) +1) which contain the
      velocity, pressure and lagrage multiplier. 
      Arrangement of dofs: u1 v1 u2 v2 u3 v3 ... un vn p1 p2 p3 ...pn lambda
      
    """
    
    
    velocity_pressure_field = solve(assembled_stiffness_matrix, self.f).flatten()
    
    num_vel_dofs = self.mesh.num_velocity_dofs
    num_press_dofs = self.mesh.num_pressure_dofs
    
    velocity = velocity_pressure_field[0:num_vel_dofs]
    pressure = velocity_pressure_field[num_vel_dofs: num_vel_dofs+num_press_dofs]
    
    return velocity, velocity_pressure_field
  
  
  def compute_dissipated_power(self,
                              velocity_field: np.ndarray,
                              assembled_A_matrix):
    """
    Args:
      velocity_field: An array velocity_field of size 
      (2*(2*nelx+1)*(2*nelx+1)) which contain the velocity. 
      Arrangement of dofs: u1 v1 u2 v2 u3 v3 ... un vn 
      assembled_A_matrix : A torch sparse coo tensor that contains 
         the assembled velocity stiffness matrix
    Returns: dissipated_power : single float value which is 
            the optimization objective
    
    """
    velocity_field = velocity_field.view((-1,1))
    force = torch.sparse.mm(assembled_A_matrix, velocity_field)
    dissipated_power = 0.5*torch.einsum('i,i->', force.view((-1)), velocity_field.view((-1)))
    return dissipated_power
        

  def fluid_objective_function(self,
                    fluid_material: fluid_material.FluidMaterial,
                    C_00: torch.DoubleTensor,
                    C_11: torch.DoubleTensor,
                    theta: torch.DoubleTensor)->torch.Tensor:
  
    """
    Calculates the fluid objective or the dissipated power.
    
    Args:
      fluid_material: Object representing the fluid material properties.
      C_00: Tensor of size (num_elems,) and represents the C_00 component of
            the permeability tensor.
      C_11: Tensor of size (num_elems,) and represents the C_11 component of
            the permeability tensor.
      theta: Tensor of size (num_elems,) representing the orientation value of the microstructures.
    
    Returns:
    dissipated_power: Tensor of size (1,)representing the dissipated power.
    """
    
    def objective_wrapper(P_00: torch.DoubleTensor, P_11: torch.DoubleTensor,
                          theta: torch.DoubleTensor)->torch.Tensor:
      """
      Calculates the fluid objective by calling the fluid functions required
      to solve fluid FEA.
      
      Args:
        
        P_00: Tensor of size (num_elems,) and represents the P_00 component of
              the inverse permeability tensor.
        P_11: Tensor of size (num_elems,) and represents the P_11 component of
              the inverse permeability tensor.
        theta: Tensor of size (num_elems,) representing the orientation value of the microstructures.
      
      Returns:
      dissipated_power: Tensor of size (1,) representing the dissipated power.
      """
      
      rot_P_00, rot_P_11, rot_P_01 = utils.rotate_2d_diagonal_matrix(P_00, P_11, theta)
    
      Aelem, elem_stiffness = fluid_material.compute_elem_stiffness_matrix(rot_P_00,
                                                                           rot_P_01,
                                                                           rot_P_01,
                                                                           rot_P_11)
      
      assembled_stiffness, assembled_A = self.assemble_stiffness_matrix(
                                                  elem_stiffness, Aelem)
      
      fluid_velocity, velocity_pressure_field = self.simulate_fluid_system(assembled_stiffness)
      
      
      dissipated_power = self.compute_dissipated_power(fluid_velocity,
                                                         assembled_A)
      return dissipated_power, velocity_pressure_field
  
    dissipated_power, velocity_pressure_field = objective_wrapper(1/C_00, 1/C_11, theta)
    return dissipated_power, velocity_pressure_field


        
        
  def get_element_velocity_pressure(self,
                                    velocity_pressure_field: np.ndarray):
    """ ...
    Args:
      velocity_pressure_field: An array of size 
          (2(2nelx+1)(2nelx+1)+(nelx+1)(nelx+1) +1) which contain the
            velocity, pressure and lagrage multiplier. 
            Arrangement of dofs: u1 v1 u2 v2 u3 v3 ... un vn p1 p2 p3 ...pn lambda
    Returns: u_velocity, v_velocity, pressure : Arrays of size (nelx*nely)
            containg element average u_velocity, v_velocity, pressure.
    """
    num_vel_dofs = self.mesh.num_velocity_dofs
    num_press_dofs = self.mesh.num_pressure_dofs
    
    velocity_nodal = velocity_pressure_field[0:num_vel_dofs]
    pressure_nodal = velocity_pressure_field[num_vel_dofs: num_vel_dofs+num_press_dofs]
    
    elem_velocity = (velocity_nodal[self.mesh.edof_mat_velocity].
                      reshape( (self.mesh.num_elems,
                               self.mesh.num_velocity_dofs_per_elem) ))
    
    elem_pressure = (pressure_nodal[self.mesh.edof_mat_pressure].
                      reshape( (self.mesh.num_elems,
                               self.mesh.num_pressure_dofs_per_elem) ))
    
    u_velocity = torch.mean(elem_velocity[:,0::2], axis=1)
    v_velocity = torch.mean(elem_velocity[:,1::2], axis=1)
    pressure = torch.mean(elem_pressure, axis=1)
    return u_velocity, v_velocity, pressure