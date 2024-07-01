"""
__author__ = "Rahul Kumar Padhy, Aaditya Chandrasekhar, and Krishnan Suresh"
__email__ = "rkpadhy@wisc.edu, cs.aaditya@gmail.com and ksuresh@wisc.edu"
"""

import numpy as np
import fluid_mesher
from dataclasses import dataclass
from enum import Enum, auto


@dataclass
class FluidBoundaryCondition:
  dir_BC: np.ndarray
  fixed_dofs: np.ndarray
  free_dofs: np.ndarray
  
def get_boundary_box(mesh, 
                     boundary_tol: float = 0.5,
                     tol: float = 1e-8) -> (tuple[dict[str, float],
                                                  dict[str, float],
                                                  dict[str, float],
                                                  dict[str, float]]):
  """Returns the boundary boxes for the given mesh.

  Args:
      mesh: the type of mesh for which to compute the boundary boxes
      boundary_tol (float): the tolerance to use when computing the boundary boxes (default 0.5)
      tol (float): the tolerance to use for numerical comparisons (default 1e-8)

  Returns:
      A tuple of four dictionaries, each representing a boundary box. Each dictionary contains the following keys:
      - x_start (np.array of size (no. of inlets)): the x-coordinates of the start point of the boundary
      - x_end (np.array of size (no. of inlets)): the x-coordinate of the end point of the boundary
      - y_start (np.array of size (no. of inlets)): the y-coordinate of the start point of the boundary
      - y_end (np.array of size (no. of inlets)): the y-coordinate of the end point of the boundary
      
  """
  lx, ly = mesh.length_x, mesh.length_y
  dx, dy = mesh.elem_dx, mesh.elem_dy
  
    
  inlet = {'x_start':0., 'x_end':0., 'y_start': 0., 'y_end': 0. + ly, 'length_x': 0., 'length_y': ly}
  outlet = {'x_start':lx, 'x_end':lx, 'y_start': 0., 'y_end': 0. + ly, 'length_x': 0., 'length_y': ly}
  top = ({'x_start':0.+boundary_tol*dx, 'x_end':lx-boundary_tol*dx,
         'y_start': ly, 'y_end': ly, 'length_x': lx, 'length_y': 0.})
  bottom = ({'x_start':0.+boundary_tol*dx, 'x_end':lx-boundary_tol*dx,
             'y_start': 0., 'y_end': 0., 'length_x': lx, 'length_y': 0.})
  return inlet, outlet, top, bottom



def compute_BC(bb, u_vel_profile, v_vel_profile,
               mesh):
  """
  Args:
    bb (np.array of size (no. of inlets)) : geometric extent of bcs imposition
    profile(np.array of size (no. of inlets)) : values of velocity imposed as a mathematical function
    mesh : spec of the mesh of dataclass of type Q2Q1Mesh
  Returns:
    bc_nodes: (np.array of size (bc nodes, no. of inlets)) contains node numbers
      for the geometric  coordinates where bcs is applied and bc_values contain the
      force values for the corresponding nodes
    u_vel_bc_values: (np.array of size (bc nodes, no. of inlets)) contains dirichlet
      velocity bc values in x-direction for the corresponding bc nodes nodes
    v_vel_bc_values: (np.array of size (bc nodes, no. of inlets)) contains dirichlet
      velocity bc values in y-direction for the bc nodes corresponding nodes
    press_bc_values: (np.array of size (bc nodes, no. of inlets)) contains dirichlet
      pressure bc values for the corresponding bc nodes nodes
  """

  # Define the bounding box
  node_bounds = (mesh.node_xy[:, 0][:, np.newaxis] >= bb['xmin']) & \
              (mesh.node_xy[:, 0][:, np.newaxis] < bb['xmax']) & \
              (mesh.node_xy[:, 1][:, np.newaxis] >= bb['ymin']) & \
              (mesh.node_xy[:, 1][:, np.newaxis] < bb['ymax'])
  n = node_bounds.shape[1]
  # print('node_bounds', node_bounds.shape)
  bc_nodes = np.column_stack([np.where(node_bounds[:, i])[0] for i in range(n)])
  # Evaluate the profile function at the nodes inside the bounding box
  bc_x = mesh.node_xy[bc_nodes, 0]
  bc_y = mesh.node_xy[bc_nodes, 1]
  
  u_vel_bc_values = np.array([u_vel_profile[i](bc_x[:, i], bc_y[:, i]) for i in range(n)]).T
  v_vel_bc_values = np.array([v_vel_profile[i](bc_x[:, i], bc_y[:, i]) for i in range(n)]).T

  return (bc_nodes, np.array(u_vel_bc_values),
          np.array(v_vel_bc_values))




def getBC(mesh, edge: dict,
          start: np.ndarray, end: np.ndarray, u_vel_profile, v_vel_profile,
          default_dir =0., tol = 1e-8):
  '''
  Args:
    edge dictionary: containing start and end coordinates in x and y direction, and length in x and y directions
    start: bcs start point. Ranges from 0 to 1 indicating fraction of edge. 
    end: bcs end point. Ranges from 0 to 1 indicating fraction of edge .
    velocity_profile: lambda function scope is from start to end of the edge.

  Returns:
    edge_nodes: total node numbers of the edge
    force_nodes: node numbers of the edge where force is applied
    force_bc: force values applied on force nodes
  '''
  n = start.shape[0]
  default_profile = np.full(n, lambda x, y: default_dir*x*y)
  dir_domain = {
    'xmin': np.array([edge['x_start'] + start[i]*edge['length_x'] - tol for i in range(n)]),
    'xmax': np.array([edge['x_start'] + end[i]*edge['length_x'] + tol for i in range(n)]),
    'ymin': np.array([edge['y_start'] + start[i]*edge['length_y'] - tol for i in range(n)]),
    'ymax': np.array([edge['y_start'] + end[i]*edge['length_y'] + tol for i in range(n)])
}

  edge_domain = {'xmin':edge['x_start']- tol, 'xmax':edge['x_end']+ tol, \
                 'ymin': edge['y_start']- tol, 'ymax':edge['y_end']+ tol}
    
  dir_nodes, u_vel_bc_values, v_vel_bc_values = compute_BC(
    dir_domain, u_vel_profile, v_vel_profile, mesh)
  
  edge_nodes, total_u_vel_dir_bc, total_v_vel_dir_bc = compute_BC(edge_domain, default_profile, 
                                        default_profile, 
                                        mesh)
  
  return edge_nodes, dir_nodes, u_vel_bc_values, v_vel_bc_values

def get_y_parabolic_profile(start: np.ndarray, end: np.ndarray, char_velocity) -> callable:
    """
    Returns a lambda function that takes in two arguments (x, y) and returns 
    a float value representing the y-coordinate
    of a point on a parabolic curve that starts at `start` 
    and ends at `end`.

    Args:
      start: a float representing the y-coordinate of the starting point
        of the parabolic curve.
      end: a float representing the y-coordinate of the ending point of 
        the parabolic curve.

    Returns: A callable lambda function that takes in two arguments (x, y) and
    returns a float value representing the y-coordinate
    of a point on a parabolic curve that starts at `start` and ends at `end`.
    """
    return (lambda x, y: char_velocity*(y - start) * (end - y)/(0.5*(end-start))**2)


def get_x_parabolic_profile(start: np.ndarray, end: np.ndarray) -> callable:
    """
    Returns a lambda function that takes in two arguments (x, y) and returns 
    a float value representing the y-coordinate
    of a point on a parabolic curve that starts at `start` 
    and ends at `end`.

    Args:
      start: a float representing the y-coordinate of the starting point of
        the parabolic curve.
      end: a float representing the y-coordinate of the ending point of the
        parabolic curve.

    Returns: A callable lambda function that takes in two arguments (x, y) and
      returns a float value representing the y-coordinate
      of a point on a parabolic curve that starts at `start` and ends at `end`.
    """
    return (lambda x, y: (x - start) * (end - x)/(0.5*(end-start))**2)
  
def get_linear_profile(start: np.ndarray, end: np.ndarray, char_vel,
                         default_dir = 0.) -> callable:
    """
    Returns a lambda function that takes in two arguments (x, y) and returns 
      a float value representing the y-coordinate
      of a point on a parabolic curve that starts at `start` and ends at `end`.
    Args:
      start: a float representing the y-coordinate of the starting point of
        the parabolic curve.
      end: a float representing the y-coordinate of the ending point of the
        parabolic curve.

    Returns: A callable lambda function that takes in two arguments (x, y) and
    returns a float value representing the y-coordinate
    of a point on a parabolic curve that starts at `start` and ends at `end`.
    """
    return (lambda x, y: char_vel +  default_dir*x*y)
  
  

def get_bc_dofs(mesh, nodes, field):
  """
    Get boundary condition degrees of freedom.

    Args:
        mesh: The mesh object.
        nodes: Array of node indices.
        field: The field type u or v velocity.

    Returns:
        np.ndarray: Array containing boundary condition degrees of freedom.
    """
  if (field == fluid_mesher.Field.U_VEL):
    dofs = mesh.node_to_vel_dof[nodes, 0]
  elif (field == fluid_mesher.Field.V_VEL):
    dofs = mesh.node_to_vel_dof[nodes, 1]
  return dofs


def double_pipe(mesh, char_vel, default_dir =0.):
  
  """
    Identify boundary condition degrees of freedom for a double pipe configuration.

    Args:
        mesh: The mesh object.
        char_vel: Characteristic velocity.
        default_dir: Default dirichlet value. Default is 0.

    Returns:
        Tuple:
            - fixed_dofs: Total fixed degrees of freedom.
            - fixed_io_dofs: Degrees of freedom of inlet/outlet where Dirichlet boundary conditions are applied.
            - io_dofs: Dirichlet values applied on inlet/outlet fixed nodes.
    """
  
  ly = mesh.nely*mesh.elem_dy
  
  inlet, outlet, top, bottom = get_boundary_box(mesh)
  
  
  inlet_start_1, inlet_end_1 = 1/6., 2/6. 
  inlet_u_velocity_profile_1 = get_y_parabolic_profile(ly*inlet_start_1,
                                                       ly*inlet_end_1,
                                                       char_vel)
  inlet_v_velocity_profile_1 = lambda x,y: default_dir*x*y

  
  inlet_start_2, inlet_end_2 = 4/6., 5/6. 
  inlet_u_velocity_profile_2 = get_y_parabolic_profile(ly*inlet_start_2,
                                                       ly*inlet_end_2,
                                                       char_vel)
  inlet_v_velocity_profile_2 = lambda x,y: default_dir*x*y
  
  
  inlet_start = np.array([inlet_start_1, inlet_start_2])
  
  inlet_end = np.array([inlet_end_1, inlet_end_2])
  inlet_u_velocity_profile = np.array([inlet_u_velocity_profile_1,
                                       inlet_u_velocity_profile_2])
  
  inlet_v_velocity_profile = np.array([inlet_v_velocity_profile_1,
                                       inlet_v_velocity_profile_2])
  
  

  (total_inlet_nodes, inlet_fixed_nodes, u_vel_in_bc,
   v_vel_in_bc) = getBC(mesh, inlet, inlet_start,
                                     inlet_end, inlet_u_velocity_profile,
                                     inlet_v_velocity_profile)
                           
  outlet_start_1, outlet_end_1 = 1/6., 2/6.
  outlet_u_velocity_profile_1 = lambda x,y: None
  outlet_v_velocity_profile_1 = lambda x,y: default_dir*x*y
  
  outlet_start_2, outlet_end_2 = 4/6., 5/6. 
  outlet_u_velocity_profile_2 = lambda x,y: None
  outlet_v_velocity_profile_2 = lambda x,y: default_dir*x*y
  
  
  outlet_start = np.array([outlet_start_1, outlet_start_2])
  
  outlet_end = np.array([outlet_end_1, outlet_end_2])
  outlet_u_velocity_profile = np.array([outlet_u_velocity_profile_1,
                                       outlet_u_velocity_profile_2])
  
  outlet_v_velocity_profile = np.array([outlet_v_velocity_profile_1,
                                      outlet_v_velocity_profile_2])
 
  
  (total_outlet_nodes, outlet_fixed_nodes, u_vel_out_bc,
   v_vel_out_bc) = getBC(mesh, outlet, outlet_start,
                                     outlet_end, outlet_u_velocity_profile,
                                     outlet_v_velocity_profile)
  
  
  
  top_u_velocity_profile = np.array([lambda x,y: default_dir*x*y])
  top_v_velocity_profile = np.array([lambda x,y: default_dir*x*y])
  top_start, top_end = np.array([0.]), np.array([1.]) 
  (total_top_nodes, top_fixed_nodes, u_vel_top_bc,
   v_vel_top_bc) = getBC(mesh, top, top_start, top_end,
                                                   top_u_velocity_profile,
                                                   top_v_velocity_profile)
  
  
  bottom_u_velocity_profile = np.array([lambda x,y: default_dir*x*y])
  bottom_v_velocity_profile = np.array([lambda x,y: default_dir*x*y])
  bottom_start, bottom_end = np.array([0.]), np.array([1.])  
  (total_bottom_nodes, bottom_fixed_nodes, u_vel_bottom_bc,
   v_vel_bottom_bc) = getBC(mesh, bottom, bottom_start, bottom_end,
                                                   bottom_u_velocity_profile,
                                                   bottom_v_velocity_profile)
                                             
  boundary_nodes = np.hstack((total_inlet_nodes.flatten(), total_outlet_nodes.flatten(),
                            total_top_nodes.flatten(), total_bottom_nodes.flatten()))
  
  inlet_nodes = inlet_fixed_nodes.flatten()
  outlet_nodes = outlet_fixed_nodes.flatten()
  
  u_fixed_io_nodes = inlet_nodes
  v_fixed_io_nodes = np.hstack((inlet_nodes, outlet_nodes))
  
  
  
  u_fixed_io_dofs = get_bc_dofs(mesh, u_fixed_io_nodes, 
                                fluid_mesher.Field.U_VEL)
  v_fixed_io_dofs = get_bc_dofs(mesh, v_fixed_io_nodes, 
                                fluid_mesher.Field.V_VEL)
  
  
  
  
  u_fixed_nodes = boundary_nodes
  v_fixed_nodes = boundary_nodes
  
  
  
  u_fixed_dofs = get_bc_dofs(mesh, u_fixed_nodes,
                             fluid_mesher.Field.U_VEL)
  v_fixed_dofs = get_bc_dofs(mesh, v_fixed_nodes,
                             fluid_mesher.Field.V_VEL)
  
  
  u_inlet_bc = u_vel_in_bc.flatten()
  v_inlet_bc = v_vel_in_bc.flatten()
  
  
  u_outlet_bc = u_vel_out_bc.flatten()
  v_outlet_bc = v_vel_out_bc.flatten()
  
  
  u_io_dir = np.hstack((u_inlet_bc, u_outlet_bc))
  v_io_dir = np.hstack((v_inlet_bc, v_outlet_bc))
  
  
  fixed_io_dofs = np.hstack((u_fixed_io_dofs, 
                             v_fixed_io_dofs)).astype(int)
  
  fixed_dofs = np.hstack((u_fixed_dofs, v_fixed_dofs)).astype(int)
  
  io_dir = np.hstack((u_io_dir, v_io_dir))
  
  io_dir = io_dir[io_dir != None]
  return fixed_io_dofs, fixed_dofs, io_dir

def diffuser(mesh, char_vel, default_dir =0.):
  
  """
    Identify boundary condition degrees of freedom for a diffuser configuration.

    Args:
        mesh: The mesh object.
        char_vel: Characteristic velocity.
        default_dir: Default dirichlet value. Default is 0.

    Returns:
        Tuple:
            - fixed_dofs: Total fixed degrees of freedom.
            - fixed_io_dofs: Degrees of freedom of inlet/outlet where Dirichlet boundary conditions are applied.
            - io_dofs: Dirichlet values applied on inlet/outlet fixed nodes.
    """
  
  ly = mesh.nely*mesh.elem_dy
  
  inlet, outlet, top, bottom = get_boundary_box(mesh)
  
  
  inlet_start, inlet_end =  np.array([0.]), np.array([1.]) 
  inlet_u_velocity_profile = np.array([get_y_parabolic_profile(ly*inlet_start,
                                                       ly*inlet_end,
                                                       char_vel)])
  inlet_v_velocity_profile = np.array([lambda x,y: default_dir*x*y]) 

  (total_inlet_nodes, inlet_fixed_nodes, u_vel_in_bc,
   v_vel_in_bc) = getBC(mesh, inlet, inlet_start,
                                     inlet_end, inlet_u_velocity_profile,
                                     inlet_v_velocity_profile)
                                     
                                     
                                     
                                     
  
  outlet_start, outlet_end = np.array([1./3.]), np.array([2./3.])                                   
  outlet_u_velocity_profile = np.array([get_y_parabolic_profile(ly*outlet_start,
                                                       ly*outlet_end,
                                                       3.*char_vel)])
  outlet_v_velocity_profile = np.array([lambda x,y: default_dir*x*y])
  (total_outlet_nodes, outlet_fixed_nodes, u_vel_outlet_bc,
   v_vel_outlet_bc) = getBC(mesh, outlet, outlet_start, outlet_end,
                                                   outlet_u_velocity_profile,
                                                   outlet_v_velocity_profile)
  
  
  
  top_u_velocity_profile = np.array([lambda x,y: default_dir*x*y])
  top_v_velocity_profile = np.array([lambda x,y: default_dir*x*y])
  top_start, top_end = np.array([0.]), np.array([1.]) 
  (total_top_nodes, top_fixed_nodes, u_vel_top_bc,
   v_vel_top_bc) = getBC(mesh, top, top_start, top_end,
                                                   top_u_velocity_profile,
                                                   top_v_velocity_profile)
  
  
  bottom_u_velocity_profile = np.array([lambda x,y: default_dir*x*y])
  bottom_v_velocity_profile = np.array([lambda x,y: default_dir*x*y])
  bottom_start, bottom_end = np.array([0.]), np.array([1.])  
  (total_bottom_nodes, bottom_fixed_nodes, u_vel_bottom_bc,
   v_vel_bottom_bc) = getBC(mesh, bottom, bottom_start, bottom_end,
                                                   bottom_u_velocity_profile,
                                                   bottom_v_velocity_profile)
                                             
  boundary_nodes = np.hstack((total_inlet_nodes.flatten(), total_outlet_nodes.flatten(),
                            total_top_nodes.flatten(), total_bottom_nodes.flatten()))
  
  inlet_nodes = inlet_fixed_nodes.flatten()
  outlet_nodes = outlet_fixed_nodes.flatten()
  
  
  u_fixed_io_nodes = np.hstack((inlet_nodes, outlet_nodes))
  v_fixed_io_nodes = np.hstack((inlet_nodes, outlet_nodes))
  
  
  
  u_fixed_io_dofs = get_bc_dofs(mesh, u_fixed_io_nodes, 
                                fluid_mesher.Field.U_VEL)
  v_fixed_io_dofs = get_bc_dofs(mesh, v_fixed_io_nodes, 
                                fluid_mesher.Field.V_VEL)
  
  
  
  
  u_fixed_nodes = boundary_nodes
  v_fixed_nodes = boundary_nodes
  
  
  
  u_fixed_dofs = get_bc_dofs(mesh, u_fixed_nodes,
                             fluid_mesher.Field.U_VEL)
  v_fixed_dofs = get_bc_dofs(mesh, v_fixed_nodes,
                             fluid_mesher.Field.V_VEL)
  
  
  u_inlet_bc = u_vel_in_bc.flatten()
  v_inlet_bc = v_vel_in_bc.flatten()
  u_outlet_bc = u_vel_outlet_bc.flatten()
  v_outlet_bc = v_vel_outlet_bc.flatten()
  
  u_io_dir = np.hstack((u_inlet_bc, u_outlet_bc))
  v_io_dir = np.hstack((v_inlet_bc, v_outlet_bc))
  
  
  fixed_io_dofs = np.hstack((u_fixed_io_dofs, 
                             v_fixed_io_dofs)).astype(int)
  
  fixed_dofs = np.hstack((u_fixed_dofs, v_fixed_dofs)).astype(int)
  
  io_dir = np.hstack((u_io_dir, v_io_dir))
  
  io_dir = io_dir[io_dir != None]
  return fixed_io_dofs, fixed_dofs, io_dir

def biffurcated_pipe(mesh, char_vel, default_dir =0.):
  
  """
    Identify boundary condition degrees of freedom for a biffurcated pipe configuration.

    Args:
        mesh: The mesh object.
        char_vel: Characteristic velocity.
        default_dir: Default dirichlet value. Default is 0.

    Returns:
        Tuple:
            - fixed_dofs: Total fixed degrees of freedom.
            - fixed_io_dofs: Degrees of freedom of inlet/outlet where Dirichlet boundary conditions are applied.
            - io_dofs: Dirichlet values applied on inlet/outlet fixed nodes.
    """
  
  ly = mesh.nely*mesh.elem_dy
  
  inlet, outlet, top, bottom = get_boundary_box(mesh)
  
  
  inlet_start, inlet_end =  np.array([1./4.]), np.array([3./4.]) 
  inlet_u_velocity_profile = np.array([get_y_parabolic_profile(ly*inlet_start,
                                                       ly*inlet_end,
                                                       char_vel)])
  inlet_v_velocity_profile = np.array([lambda x,y: default_dir*x*y]) 

  (total_inlet_nodes, inlet_fixed_nodes, u_vel_in_bc,
   v_vel_in_bc) = getBC(mesh, inlet, inlet_start,
                                     inlet_end, inlet_u_velocity_profile,
                                     inlet_v_velocity_profile)
                        
                        
  outlet_start_1, outlet_end_1 = 1/8., 2/8.
  outlet_u_velocity_profile_1 = get_y_parabolic_profile(ly*outlet_start_1,
                                                       ly*outlet_end_1,
                                                       1*char_vel)
  outlet_v_velocity_profile_1 = lambda x,y: default_dir*x*y
  
  outlet_start_2, outlet_end_2 = 7/8., 1. 
  outlet_u_velocity_profile_2 =  get_y_parabolic_profile(ly*outlet_start_2,
                                                       ly*outlet_end_2,
                                                       3*char_vel)
  outlet_v_velocity_profile_2 = lambda x,y: default_dir*x*y
  
  
  outlet_start = np.array([outlet_start_1, outlet_start_2])
  
  outlet_end = np.array([outlet_end_1, outlet_end_2])
  outlet_u_velocity_profile = np.array([outlet_u_velocity_profile_1,
                                       outlet_u_velocity_profile_2])
  
  outlet_v_velocity_profile = np.array([outlet_v_velocity_profile_1,
                                      outlet_v_velocity_profile_2])
 

  (total_outlet_nodes, outlet_fixed_nodes, u_vel_outlet_bc,
   v_vel_outlet_bc) = getBC(mesh, outlet, outlet_start,
                                     outlet_end, outlet_u_velocity_profile,
                                     outlet_v_velocity_profile)
  

  top_u_velocity_profile = np.array([lambda x,y: default_dir*x*y])
  top_v_velocity_profile = np.array([lambda x,y: default_dir*x*y])
  top_start, top_end = np.array([0.]), np.array([1.]) 
  (total_top_nodes, top_fixed_nodes, u_vel_top_bc,
   v_vel_top_bc) = getBC(mesh, top, top_start, top_end,
                                                   top_u_velocity_profile,
                                                   top_v_velocity_profile)
  
  
  bottom_u_velocity_profile = np.array([lambda x,y: default_dir*x*y])
  bottom_v_velocity_profile = np.array([lambda x,y: default_dir*x*y])
  bottom_start, bottom_end = np.array([0.]), np.array([1.])  
  (total_bottom_nodes, bottom_fixed_nodes, u_vel_bottom_bc,
   v_vel_bottom_bc) = getBC(mesh, bottom, bottom_start, bottom_end,
                                                   bottom_u_velocity_profile,
                                                   bottom_v_velocity_profile)
                                             
  boundary_nodes = np.hstack((total_inlet_nodes.flatten(), total_outlet_nodes.flatten(),
                            total_top_nodes.flatten(), total_bottom_nodes.flatten()))
  
  inlet_nodes = inlet_fixed_nodes.flatten()
  outlet_nodes = outlet_fixed_nodes.flatten()
  
  
  u_fixed_io_nodes = np.hstack((inlet_nodes, outlet_nodes))
  v_fixed_io_nodes = np.hstack((inlet_nodes, outlet_nodes))
  
  
  
  u_fixed_io_dofs = get_bc_dofs(mesh, u_fixed_io_nodes, 
                                fluid_mesher.Field.U_VEL)
  v_fixed_io_dofs = get_bc_dofs(mesh, v_fixed_io_nodes, 
                                fluid_mesher.Field.V_VEL)
  
  
  
  
  u_fixed_nodes = boundary_nodes
  v_fixed_nodes = boundary_nodes
  
  
  
  u_fixed_dofs = get_bc_dofs(mesh, u_fixed_nodes,
                             fluid_mesher.Field.U_VEL)
  v_fixed_dofs = get_bc_dofs(mesh, v_fixed_nodes,
                             fluid_mesher.Field.V_VEL)
  
  
  u_inlet_bc = u_vel_in_bc.flatten()
  v_inlet_bc = v_vel_in_bc.flatten()
  u_outlet_bc = u_vel_outlet_bc.flatten()
  v_outlet_bc = v_vel_outlet_bc.flatten()
  
  u_io_dir = np.hstack((u_inlet_bc, u_outlet_bc))
  v_io_dir = np.hstack((v_inlet_bc, v_outlet_bc))
  
  
  fixed_io_dofs = np.hstack((u_fixed_io_dofs, 
                             v_fixed_io_dofs)).astype(int)
  
  fixed_dofs = np.hstack((u_fixed_dofs, v_fixed_dofs)).astype(int)
  
  io_dir = np.hstack((u_io_dir, v_io_dir))
  
  io_dir = io_dir[io_dir != None]
  return fixed_io_dofs, fixed_dofs, io_dir



def bent_pipe(mesh, char_vel, default_dir =0.):
  
  """
    Identify boundary condition degrees of freedom for a bent pipe configuration.

    Args:
        mesh: The mesh object.
        char_vel: Characteristic velocity.
        default_dir: Default dirichlet value. Default is 0.

    Returns:
        Tuple:
            - fixed_dofs: Total fixed degrees of freedom.
            - fixed_io_dofs: Degrees of freedom of inlet/outlet where Dirichlet boundary conditions are applied.
            - io_dofs: Dirichlet values applied on inlet/outlet fixed nodes.
    """
  
  ly = mesh.nely*mesh.elem_dy
  
  inlet, outlet, top, bottom = get_boundary_box(mesh)
  
  
  inlet_start_1, inlet_end_1 =  4/5., 1.
  inlet_u_velocity_profile_1 = get_y_parabolic_profile(ly*inlet_start_1,
                                                       ly*inlet_end_1,
                                                       char_vel)
  inlet_v_velocity_profile_1 = lambda x,y: default_dir*x*y
 
  
  inlet_start_2, inlet_end_2 = 0., 1/5. 
  inlet_u_velocity_profile_2 = get_y_parabolic_profile(ly*inlet_start_2,
                                                       ly*inlet_end_2,
                                                       -char_vel)
  inlet_v_velocity_profile_2 = lambda x,y: default_dir*x*y
  
  
  inlet_start = np.array([inlet_start_1, inlet_start_2])
  
  inlet_end = np.array([inlet_end_1, inlet_end_2])
  inlet_u_velocity_profile = np.array([inlet_u_velocity_profile_1,
                                       inlet_u_velocity_profile_2])
  
  inlet_v_velocity_profile = np.array([inlet_v_velocity_profile_1,
                                       inlet_v_velocity_profile_2])
  
  

  (total_inlet_nodes, inlet_fixed_nodes, u_vel_in_bc,
   v_vel_in_bc) = getBC(mesh, inlet, inlet_start,
                                     inlet_end, inlet_u_velocity_profile,
                                     inlet_v_velocity_profile)
                                     
                                     
                                     
                                     
  
                                     
  outlet_u_velocity_profile = np.array([lambda x,y: default_dir*x*y])
  outlet_v_velocity_profile = np.array([lambda x,y: default_dir*x*y])
  outlet_start, outlet_end = np.array([0.]), np.array([1.]) 
  (total_outlet_nodes, outlet_fixed_nodes, u_vel_outlet_bc,
   v_vel_outlet_bc) = getBC(mesh, outlet, outlet_start, outlet_end,
                                                   outlet_u_velocity_profile,
                                                   outlet_v_velocity_profile)
  
  
  
  top_u_velocity_profile = np.array([lambda x,y: default_dir*x*y])
  top_v_velocity_profile = np.array([lambda x,y: default_dir*x*y])
  top_start, top_end = np.array([0.]), np.array([1.]) 
  (total_top_nodes, top_fixed_nodes, u_vel_top_bc,
   v_vel_top_bc) = getBC(mesh, top, top_start, top_end,
                                                   top_u_velocity_profile,
                                                   top_v_velocity_profile)
  
  
  bottom_u_velocity_profile = np.array([lambda x,y: default_dir*x*y])
  bottom_v_velocity_profile = np.array([lambda x,y: default_dir*x*y])
  bottom_start, bottom_end = np.array([0.]), np.array([1.])  
  (total_bottom_nodes, bottom_fixed_nodes, u_vel_bottom_bc,
   v_vel_bottom_bc) = getBC(mesh, bottom, bottom_start, bottom_end,
                                                   bottom_u_velocity_profile,
                                                   bottom_v_velocity_profile)
                                             
  boundary_nodes = np.hstack((total_inlet_nodes.flatten(), total_outlet_nodes.flatten(),
                            total_top_nodes.flatten(), total_bottom_nodes.flatten()))
  
  inlet_nodes = inlet_fixed_nodes.flatten()
  
  
  u_fixed_io_nodes = inlet_nodes
  v_fixed_io_nodes = inlet_nodes
  
  
  
  u_fixed_io_dofs = get_bc_dofs(mesh, u_fixed_io_nodes, 
                                fluid_mesher.Field.U_VEL)
  v_fixed_io_dofs = get_bc_dofs(mesh, v_fixed_io_nodes, 
                                fluid_mesher.Field.V_VEL)
  
  
  
  
  u_fixed_nodes = boundary_nodes
  v_fixed_nodes = boundary_nodes
  
  
  
  u_fixed_dofs = get_bc_dofs(mesh, u_fixed_nodes,
                             fluid_mesher.Field.U_VEL)
  v_fixed_dofs = get_bc_dofs(mesh, v_fixed_nodes,
                             fluid_mesher.Field.V_VEL)
  
  
  u_inlet_bc = u_vel_in_bc.flatten()
  v_inlet_bc = v_vel_in_bc.flatten()
  
  u_io_dir = np.hstack((u_inlet_bc))
  v_io_dir = np.hstack((v_inlet_bc))
  
  
  fixed_io_dofs = np.hstack((u_fixed_io_dofs, 
                             v_fixed_io_dofs)).astype(int)
  
  fixed_dofs = np.hstack((u_fixed_dofs, v_fixed_dofs)).astype(int)
  
  io_dir = np.hstack((u_io_dir, v_io_dir))
  
  io_dir = io_dir[io_dir != None]
  return fixed_io_dofs, fixed_dofs, io_dir


class FluidSampleProblems(Enum):
  DOUBLE_PIPE = auto()
  BENT_PIPE = auto()
  DIFFUSER = auto()
  BIFFURCATED_PIPE = auto()


def get_dirichlet_bc_and_fixed_dofs(mesh, char_vel, 
                                    example: FluidSampleProblems):
  """
  Args:
    mesh : spec of the mesh of dataclass of type Q2Q1Mesh
    example: value corresponds to the example boundary ondition we are solving for
  Returns: bc contains
            bc_dofs: dof numbers for the geometric 
           coordinates where bcs is applied and 
           force: contain the magnitude of bcs applied at at the bc_dofs
   """
  
  
  dir_BC = np.zeros((mesh.num_total_dofs))
  
  
  if(example == FluidSampleProblems.DOUBLE_PIPE):
    fixed_io_dofs, fixed_dofs, io_dir = double_pipe(mesh, char_vel)
  elif(example == FluidSampleProblems.BENT_PIPE):
    fixed_io_dofs, fixed_dofs, io_dir = bent_pipe(mesh, char_vel)
  elif(example == FluidSampleProblems.DIFFUSER):
    fixed_io_dofs, fixed_dofs, io_dir = diffuser(mesh, char_vel)
  elif(example == FluidSampleProblems.BIFFURCATED_PIPE):
    fixed_io_dofs, fixed_dofs, io_dir = biffurcated_pipe(mesh, char_vel)
  else:
    print('ERROR: Unknown problem')

  dir_BC[fixed_io_dofs] = io_dir
  free_dofs = np.setdiff1d(mesh.total_dofs, fixed_dofs)
  bc = FluidBoundaryCondition(dir_BC, fixed_dofs, free_dofs)

  return bc
  
  
  
  
  