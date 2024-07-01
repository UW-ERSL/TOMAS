"""
__author__ = "Rahul Kumar Padhy, Aaditya Chandrasekhar, and Krishnan Suresh"
__email__ = "rkpadhy@wisc.edu, cs.aaditya@gmail.com and ksuresh@wisc.edu"
"""
import numpy as np
import torch
import utils
import matplotlib.pyplot as plt
from matplotlib import colors
import scipy.io as sio
from scipy.ndimage import rotate
import sys
sys.path.insert(0, '../')
sys.path.insert(0, '../dataset')
import dataset.supershape as supershape
from typing import Dict

def plot_microstructures_in_macro_mesh(mstr_params: supershape.SuperShapes,
                                       mstr_rotation: np.ndarray,
                                       nelx: int, nely: int, epoch:int)-> None:
  """
  Plot microstructures within a macro mesh.

  Args:
      mstr_params: ataclass of `SuperShapes` that contain the parameter of
      (num_shapes,) shapes.
      mstr_rotation: Array of size (num_shapes,)
      nelx: Number of microstructures along the X direction.
      nely: Number of microstructures along the Y direction.
      epoch: Current epoch number.
  """
  
  if mstr_params.num_shapes != nelx*nely:
    print(f'ERROR: num microstructures should be equal to nelx*nely. Got'
          f' {mstr_params.num_shapes} microstructures and nelx*nely={nelx*nely}')
  x, y = supershape.get_euclidean_coords_of_points_on_surf_super_shape(mstr_params, mstr_rotation)
  fig, ax = plt.subplots(1, 1)
  ax.patch.set_facecolor('#DAE8FC') # blue
  # ax.patch.set_facecolor('white') # blue
  ctr = 0
  for rw in range(nelx):
    dx = 2*rw + 1.
    bb_x_min, bb_x_max = 2*rw, 2*rw + 2.
    for col in range(nely):
      dy = 2*col + 1.
      bb_y_min, bb_y_max = 2*col, 2*col + 2.
      x[ctr, :] = np.clip(x[ctr, :] + dx, a_min=bb_x_min, a_max=bb_x_max)
      y[ctr, :] = np.clip(y[ctr, :] + dy, a_min=bb_y_min, a_max=bb_y_max)
      # ax.fill(x[ctr, :], y[ctr, :], facecolor='black', edgecolor='black',
      #         linewidth=1.)
      ax.fill(x[ctr, :], y[ctr, :], facecolor='#F8CECC', edgecolor='black',
                linewidth=0.2)
      ctr += 1
  ax.axis('equal')
  plt.plot([0, 2*nelx, 2*nelx, 0, 0], [0, 0, 2*nely, 2*nely, 0], 'k')
  plt.show()
  plt.pause(0.01)

def plot_convergence( convg: Dict[str, np.ndarray]) -> None:
  """
  Plot convergence data.

  Args:
      convg (Dict[str, np.ndarray]): Dictionary containing convergence data. Keys represent
          different variables being tracked (e.g., loss, constraint, etc.), and values are arrays
          representing the values of these variables across epochs.
  """
  x = np.array(convg['epoch'])
  for key in convg:    
    if(key == 'epoch'):
      continue # epoch is x axis for all plots
    plt.figure()
    y = np.array(convg[key])
    plt.semilogy(x, y, label = str(key))
    plt.xlabel('Iterations')
    plt.ylabel(str(key))
    plt.grid('True')
    
def plot_elemental_field(field: np.ndarray, title: str, color_map: str) -> None:
  """
  Plot elemental field data.

  Args:
    field: Array containing the field data.
    title: Title of the plot.
    color_map: Colormap to be used in the plot.
  """
  plt.figure()
  P =plt.imshow(field, interpolation = 'none', origin = 'lower', cmap =color_map)
  color_range = None
  if (title == "pressure"):
    color_range = np.linspace(np.amin(field), np.amax(field), 10,
                    endpoint=True)
  plt.colorbar(P, ticks = color_range)
  plt.title(title)
  plt.axis('equal')
  ax = plt.gca()
  ax.axes.xaxis.set_ticks([])
  ax.axes.yaxis.set_ticks([])
  plt.show()
  plt.pause(0.001)


def plot_elemental_field_constitutive(field: np.ndarray, title: str, color_map: str) -> None:
  """
  Plot elemental field data with specified color range.

  Args:
    field: Array containing the field data.
    title: Title of the plot.
    color_map: Colormap to be used in the plot.
  """
  plt.figure()
  P =plt.imshow(field, interpolation = 'none', origin = 'lower',
                cmap =color_map, vmin=0., vmax=0.45)
  color_range = None
  if (title == "pressure"):
    color_range = np.linspace(np.amin(field), np.amax(field), 10,
                    endpoint=True)
  plt.colorbar(P, ticks = color_range)
  plt.title(title)
  plt.axis('equal')
  ax = plt.gca()
  ax.axes.xaxis.set_ticks([])
  ax.axes.yaxis.set_ticks([])
  plt.show()
  plt.pause(0.001)



  
  