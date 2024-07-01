"""
__author__ = "Rahul Kumar Padhy, Aaditya Chandrasekhar, and Krishnan Suresh"
__email__ = "rkpadhy@wisc.edu, cs.aaditya@gmail.com and ksuresh@wisc.edu"
"""

"""Simple rectangular geometry mesher."""
import numpy as np
from enum import Enum, auto
import matplotlib.pyplot as plt
from dataclasses import dataclass 
from typing import List
import utils


@dataclass
class BoundingBox:
  x_min: float
  y_min: float
  x_max: float
  y_max: float

  
class Meshes(Enum):
  """
      Q2Q1: A mesh type which has num_dim*9 velocity dofs 
      and 4 pressure dofs per element.
    """
  Q2Q1 = auto()
  
class Field(Enum):
  """
      Attributes:
        U_VEL: Velocity field in x-direction.
        V_VEL: Velocity field in y-direction.
  """      
  U_VEL = auto()
  V_VEL = auto()
  

@dataclass
class Q2Q1Mesh:
  """  Attributes:    
      num_dim: number of dimensions of the mesh. Currently only handles 2D    
      nelx: number of elements along X axis    
      nely: number of elements along Y axis    
      num_elems: number of elements in the mesh   
      num_nodes: number of nodes in the mesh. Assume a bilinear quad element
      elem_area: Area of each element
      elem_dx: Array which contains the size of the element along X 
      elem_dy: Array which contains the size of the element along Y 
      elem_centers: Array of size (num_elems, 2) which are the coordinates
      of the enters of the element  
      length_x: length of domain along X axis    
      length_y: length of domain along Y axis  
      num_velocity_dofs_per_elem: number of velocity degrees of freedom per element
      num_pressure_dofs_per_elem: number of pressure degrees of freedom per element
      num_total_dofs_per_elem: number of total degrees of freedom per element
      num_velocity_nodes: number of velocity nodes in the mesh
      num_pressure_nodes: number of pressure nodes in the mesh
      num_velocity_dofs: number of velocity degrees of freedom in the mesh
      num_pressure_dofs: number of pressure degrees of freedom in the mesh
      num_total_dofs: total number of degrees of freedom in the mesh
      bounding_box: Contains the max and min coordinates of the mesh    
      edof_mat_pressure: element degree of freedom matrix for pressure
      edof_mat_velocity: element degree of freedom matrix for velocity
      edof_mat_global: global element degree of freedom matrix
      node_to_vel_dof: mapping between velocity node indices and degrees of freedom indices
      node_idx: node IDs for each element for the stiffness matrix assembly
      node_idx_a: node IDs for each element for velocity stiffness assembly
      node_xy: node coordinates of the mesh
      u_velocity_dofs: velocity degrees of freedom in the x-direction
      v_velocity_dofs: velocity degrees of freedom in the y-direction
      pressure_dofs: pressure degrees of freedom
      total_dofs: total degrees of freedom
      """
  num_dim: int
  nelx: int
  nely: int
  num_elems: int
  elem_dx: float
  elem_dy: float
  num_nodes: int
  elem_area: float
  elem_centers: np.ndarray
  length_x: float
  length_y: float
  num_velocity_dofs_per_elem: int
  num_pressure_dofs_per_elem: int
  num_total_dofs_per_elem: int
  num_velocity_nodes: int
  num_pressure_nodes: int
  num_velocity_dofs: int
  num_pressure_dofs: int
  num_total_dofs: int
  bounding_box: BoundingBox
  edof_mat_pressure: np.ndarray
  edof_mat_velocity: np.ndarray
  edof_mat_global: np.ndarray
  node_to_vel_dof: np.ndarray
  node_idx:  List[np.ndarray]
  node_idx_a:  List[np.ndarray]
  node_xy: np.ndarray
  u_velocity_dofs: np.ndarray
  v_velocity_dofs: np.ndarray
  pressure_dofs: np.ndarray
  total_dofs: np.ndarray
  



def q2q1_mesher(nelx: int, nely: int, bounding_box: BoundingBox):
  # return a Q2Q1 mesh object
  """ Constructs a simple 2D uniform grid Q2-Q1 quad mesh with  
      quadratic interpolation for velocity and linear for pressure for
      fluid simulation.
    Global node numbering velocity:
    3--6--9
    |     |
    2  5  8
    |     |
    1--4--7
    
    Global node numbering pressure:
    2----4
    |    |
    |    |
    |    |
    1----3
    
    Elemental node numbering velocity:
    4--8--3
    |     |
    5  9  7
    |     |
    1--6--2
    
    Elemental node numbering pressure:
    4----3
    |    |
    |    |
    |    |
    1----2
  Args:
    nelx: number of elements along X axis
    nely: number of elements along Y axis
    bounding_box: TODO
    
  Returns: A dataclass of Q2Q1Mesh that contain all the
    information pertinent to the mesh.
  """
  
  num_dim = 2
  num_velocity_dofs_per_elem = 18
  num_pressure_dofs_per_elem = 4
  length_x = np.abs(bounding_box.x_max - bounding_box.x_min)
  length_y = np.abs(bounding_box.y_max - bounding_box.y_min)
  elem_dx = length_x/nelx
  elem_dy = length_y/nely
  elem_area = elem_dx*elem_dy
  num_elems = nelx*nely
  
  num_nodes = (num_dim*nelx+1)*(num_dim*nely+1)
  num_velocity_nodes = num_nodes
  num_pressure_nodes = (nelx+1)*(nely+1)
          
  num_velocity_dofs = num_dim*num_velocity_nodes
  lagrangian_dof = 1
  num_pressure_dofs = num_pressure_nodes
  num_total_dofs = (num_velocity_dofs + num_pressure_dofs +lagrangian_dof)
  
  
  u_velocity_dofs = np.arange(0, num_velocity_dofs, num_dim)
  v_velocity_dofs = np.arange(1, num_velocity_dofs, num_dim)
  pressure_dofs = np.arange(num_velocity_dofs,
                            num_total_dofs)
  total_dofs = np.hstack((u_velocity_dofs, v_velocity_dofs,
                            pressure_dofs))
  
  node_to_vel_dof = np.vstack((u_velocity_dofs,
                             v_velocity_dofs)).T

  bounding_box = BoundingBox(x_min=0., y_min=0.,
                             x_max=length_x, y_max=length_y)
  
  elem_centers = utils.generate_points_in_domain(nelx, nely,
                                             elem_dx, elem_dy,
                                             num_dim)
  
  
    
  x = np.linspace(0, length_x, 2*nelx+1)
  y = np.linspace(0, length_y, 2*nely+1)
  Y,X = np.meshgrid(y,x)
  node_xy = np.vstack((X.reshape(-1), Y.reshape(-1))).T
  
  num_total_dofs_per_elem = (num_velocity_dofs_per_elem 
                             + num_pressure_dofs_per_elem+
                             lagrangian_dof)
  
  edof_mat_velocity = np.zeros((num_elems, num_velocity_dofs_per_elem))
  
  for elx in range(nelx):
    for ely in range(nely):
      el = ely+elx*nely
      n1 = (2*nely+1)*2*elx + 2*ely
      n2 = (2*nely+1)*(2*elx+1) + 2*ely
      n3 = (2*nely+1)*(2*elx+2) + 2*ely
  
      edof_mat_velocity[el,:] = np.array([2*n1, 2*n1+1, 2*(n3), 2*(n3)+1,\
                                          2*(n3+2), 2*(n3+2)+1, 2*(n1+2), 2*(n1+2)+1,\
                                          2*(n2), 2*(n2)+1, 2*(n3+1), 2*(n3+1)+1,\
                                          2*(n2+2), 2*(n2+2)+1, 2*(n1+1), 2*(n1+1)+1,\
                                          2*(n2+1), 2*(n2+1)+1])
  

  edof_mat_velocity = edof_mat_velocity.astype(int)
  
  edof_mat_pressure = np.zeros((num_elems, num_pressure_dofs_per_elem))
  ctr = 0
  for elx in range(nelx):
    for ely in range(nely):
      el = ely + elx*nely
      n1 = (nely+1)*elx+ely
      n2 = (nely+1)*(elx+1)+ely
      edof_mat_pressure[el,:] = ctr + np.array([n1, n2, n2+1, n1+1])
  

  edof_mat_pressure = edof_mat_pressure.astype(int)
  
  
  edof_mat_global = np.zeros((num_elems,
                                num_velocity_dofs_per_elem
                                + num_pressure_dofs_per_elem))
  
  one_matrix = np.ones((num_elems,1))
  edof_mat_global = np.hstack((edof_mat_velocity, \
                  num_velocity_dofs+ edof_mat_pressure,\
                  num_velocity_dofs+\
                  num_pressure_dofs+one_matrix-1))
  iK= np.kron(edof_mat_global,\
                      np.ones((num_total_dofs_per_elem ,1))).flatten().astype(int)
  jK = np.kron(edof_mat_global,\
                      np.ones((1,num_total_dofs_per_elem))).flatten().astype(int)
  bK = tuple(np.zeros((len(iK))).astype(int)) #batch values
  node_idx = [bK, iK, jK] # TODO: Getback batch dim!!!!
  
  
  iA= np.kron(edof_mat_velocity,\
                      np.ones((num_velocity_dofs_per_elem ,1))).flatten().astype(int)
  jA = np.kron(edof_mat_velocity,\
                      np.ones((1,num_velocity_dofs_per_elem))).flatten().astype(int)
  bA = tuple(np.zeros((len(iA))).astype(int)) #batch values
  node_idx_a = [iA, jA] 
  
  
  mesh = Q2Q1Mesh(num_dim, nelx, nely, num_elems, elem_dx, elem_dy,
                    num_nodes, elem_area, elem_centers, length_x, length_y,
                    num_velocity_dofs_per_elem, num_pressure_dofs_per_elem,
                    num_total_dofs_per_elem, num_velocity_nodes, num_pressure_nodes,
                    num_velocity_dofs, num_pressure_dofs, num_total_dofs, bounding_box,
                    edof_mat_pressure, edof_mat_velocity, edof_mat_global,
                    node_to_vel_dof, node_idx, node_idx_a, node_xy, u_velocity_dofs,
                    v_velocity_dofs, pressure_dofs, total_dofs)
  
  
  return mesh



def fluid_mesher(nelx: int, nely: int, bounding_box: BoundingBox,
                 mesh_type: Meshes = Meshes.Q2Q1):
    """Meshes the given geometry according to the specified mesh type.

      Args:
          mesh_type (Meshes): the type of mesh to generate
          geometry: the geometry to mesh

      Returns:
          The meshed geometry.
      """
    if(mesh_type == Meshes.Q2Q1):
        return q2q1_mesher(nelx, nely, bounding_box)
  
    