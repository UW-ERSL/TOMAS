import numpy as np
import matplotlib.pyplot as plt
import dataclasses
import mesher
import geopandas as gpd
from shapely.geometry.polygon import Polygon
import shapely
from typing import Tuple
_BoundingBox = mesher.BoundingBox

@dataclasses.dataclass
class Extents:
  min_val: float
  max_val: float

  def __post_init__(self):
    if self.min_val == self.max_val:
      print('WARNING: The min val and the max val are the same. This could'
            f'result in numerical issues. minval = {self.min_val:.2E}, '
            f'maxval = {self.max_val:.2E}')
  @property
  def range(self)->float:
    return self.max_val - self.min_val

@dataclasses.dataclass
class SuperShapeExtents:
  a: Extents
  b: Extents
  m: Extents
  n1: Extents
  n2: Extents
  n3: Extents
  center_x: Extents
  center_y: Extents

def normalize(value: np.ndarray, extents: Extents):
  return (value - extents.min_val)/extents.range

def renormalize(value: np.ndarray, extents: Extents):
  return value*extents.range + extents.min_val

@dataclasses.dataclass
class SuperShapes:
  """ Returns a supershape in [-1, 1] X [-1, 1]
  Attributes:
    a: Array of size (num_shapes,) with values in range (0, 1]
    b: Array of size (num_shapes,) with values in range (0, 1]
    m: Array of size (num_shapes,) that dicates the number of lobes in the shape
    n1: Array of size (num_shapes,) in [0, inf] that controls sharpness of the shape
    n2: Array of size (num_shapes,) in [0, inf] that controls sharpness of the shape
    n3: Array of size (num_shapes,) in [0, inf] that controls sharpness of the shape
    center_x: Array of size (num_shapes,) with values in range [-1, 1]
    center_y: Array of size (num_shapes,) with values in range [-1, 1]
  """
  a: np.ndarray
  b: np.ndarray
  m: np.ndarray
  n1: np.ndarray
  n2: np.ndarray
  n3: np.ndarray
  center_x: np.ndarray
  center_y: np.ndarray

  @property
  def num_params_per_shape(self)->int:
    return 8

  @property
  def num_shapes(self)->int:
    return self.a.shape[0]

  @property
  def total_num_parameters(self)->int:
    return self.num_params_per_shape*self.num_shapes

  @property
  def bounding_box(self)->_BoundingBox:
    return _BoundingBox(x_min=-1., x_max=1., y_min=-1., y_max=1.)

  @property
  def domain_length_x(self)->float:
    return np.abs(self.bounding_box.x_max - self.bounding_box.x_min)

  @property
  def domain_length_y(self)->float:
    return np.abs(self.bounding_box.y_max - self.bounding_box.y_min)

  @classmethod
  def from_array(
      cls,
      state_array: np.ndarray,
      num_shapes: int,
      ) -> 'SuperShapes':
    """
    Convert a rank-1 array into a `SuperShapes` object.

    Args:
        state_array: Rank-1 numpy array containing the state/mstr data.
        num_shapes: Number of shapes.

    Returns:
        SuperShapes: SuperShapes object initialized from the array.
    """
    a = state_array[0:num_shapes]
    b = state_array[num_shapes:2*num_shapes]
    m = state_array[2*num_shapes:3*num_shapes]
    n1 = state_array[3*num_shapes:4*num_shapes]
    n2 = state_array[4*num_shapes:5*num_shapes]
    n3 = state_array[5*num_shapes:6*num_shapes]
    center_x = state_array[6*num_shapes:7*num_shapes]
    center_y = state_array[7*num_shapes:8*num_shapes]
    return SuperShapes(a, b, m, n1, n2, n3, center_x, center_y)

  def to_array(self) -> np.ndarray:
    """Converts the `ConvexPolys` into a rank-1 array."""
    return np.concatenate([f.reshape((-1)) for f in dataclasses.astuple(self)])

  def to_stacked_array(self) -> np.ndarray:
    """Returned a stacked array of size (num_shapes, num_params_per_shape).
        The stacking sequence is (center_x, center_y, radius)
    """
    return np.hstack((self.a.reshape((-1, 1)),
                      self.b.reshape((-1, 1)),
                      self.m.reshape((-1, 1)),
                      self.n1.reshape((-1, 1)),
                      self.n2.reshape((-1, 1)),
                      self.n3.reshape((-1, 1)),
                      self.center_x.reshape((-1, 1)),
                      self.center_y.reshape((-1, 1))
                      ))

  def to_normalized_array(self, shape_extents: SuperShapeExtents) -> np.ndarray:
    """Converts the `SuperShapes` into a rank-1 array with values normalized."""
    return np.concatenate((
      normalize(self.a, shape_extents.a).reshape((-1)),
      normalize(self.b, shape_extents.b).reshape((-1)),
      normalize(self.m, shape_extents.m).reshape((-1)),
      normalize(self.n1, shape_extents.n1).reshape((-1)),
      normalize(self.n2, shape_extents.n2).reshape((-1)),
      normalize(self.n3, shape_extents.n3).reshape((-1)),
      normalize(self.center_x, shape_extents.center_x).reshape((-1)),
      normalize(self.center_y, shape_extents.center_y).reshape((-1)),
    ))

  @classmethod
  def from_normalized_array(cls, state_array: np.ndarray, num_shapes: int,
      shape_extents: SuperShapeExtents)->'SuperShapes':
    """Converts a normalized rank-1 array into `SuperShapes`."""

    return SuperShapes(
      renormalize(state_array[0*num_shapes:1*num_shapes], shape_extents.a),
      renormalize(state_array[1*num_shapes:2*num_shapes], shape_extents.b),
      renormalize(state_array[2*num_shapes:3*num_shapes], shape_extents.m),
      renormalize(state_array[3*num_shapes:4*num_shapes], shape_extents.n1),
      renormalize(state_array[4*num_shapes:5*num_shapes], shape_extents.n2),
      renormalize(state_array[5*num_shapes:6*num_shapes], shape_extents.n3),
      renormalize(state_array[6*num_shapes:7*num_shapes], shape_extents.center_x),
      renormalize(state_array[7*num_shapes:8*num_shapes], shape_extents.center_y),)

def compute_radius_of_super_shape(theta: np.ndarray,
                                  super_shape_param: SuperShapes):
  """Computes the radius as a function of angle.
  NOTE: This assumes the center of the shape is at (0,0)
    radius(theta) = ( |cos(m*theta/4)/a|^n2 + |sin(m*theta/4)/b|^n3 )^(-1/n1)
  Args:
    theta: Array of size (num_shapes, num_pts,)
    super_shape_param: dataclass of `SuperShape` that
  Returns: Array of shape (num_shapes, num_pts)
  """
  # s -> shapes, p -> points
  costerm = np.cos(0.25*np.einsum('s, sp -> sp',2*np.round(super_shape_param.m), theta))
  sinterm = np.sin(0.25*np.einsum('s, sp -> sp',2*np.round(super_shape_param.m), theta))
  t1 = np.power(np.abs(np.einsum('sp, s -> sp',costerm, 1./super_shape_param.a)),
                super_shape_param.n2[:, np.newaxis])
  t2 = np.power(np.abs(np.einsum('sp, s -> sp',sinterm, 1./super_shape_param.b)),
                super_shape_param.n3[:, np.newaxis])
  return np.power(t1 + t2, -1./super_shape_param.n1[:, np.newaxis])


def compute_shapely_polygon_perimeter(polygons: list[Polygon])-> np.ndarray:
  """
  Compute the perimeter of each polygon in the list.

  Args:
    polygons: List of Shapely Polygon objects.

  Returns:
    perimeter: Array of shape (num_polys,) containing the perimeter of each polygon.
  """
  num_blobs = len(polygons)
  perimeter = np.zeros((num_blobs))
  for b in range(len(polygons)):
    perimeter[b] = polygons[b].length
  return perimeter

def compute_shapely_polygon_area(polygons: list[Polygon])-> np.ndarray:
  """
  Compute the area of each polygon in the list.

  Args:
    polygons: List of Shapely Polygon objects.

  Returns:
    perimeter: Array of shape (num_polys,) containing the area of each polygon.
  """
  num_blobs = len(polygons)
  area = np.zeros((num_blobs))
  for b in range(len(polygons)):
    area[b] = polygons[b].area
  return area


def super_shape_to_shapely_polygon(shape_param: SuperShapes,
                                   shape_rotation_rad: np.ndarray=None,
                                   prune_out_intersecting_shapes: bool = True,
                                   num_div: int=300)->list[Polygon]:
  """
    Convert SuperShapes to Shapely polygons.

    Args:
        shape_param: SuperShapes instance containing shape parameters.
        shape_rotation_rad: Array of shape rotations in radians. Default is None.
        prune_out_intersecting_shapes: Flag to prune out intersecting shapes with bounding box. Default is True.
        num_div: Number of divisions for shape discretization. Default is 300.

    Returns:
        Tuple containing a list of Shapely polygons and the pruned SuperShapes instance.
    """
  polygons = []
  bb = Polygon([
      (shape_param.bounding_box.x_min, shape_param.bounding_box.y_min),
      (shape_param.bounding_box.x_max, shape_param.bounding_box.y_min),
      (shape_param.bounding_box.x_max,shape_param.bounding_box.y_max),
      (shape_param.bounding_box.x_min,shape_param.bounding_box.y_max)])
  x_poly, y_poly = get_euclidean_coords_of_points_on_surf_super_shape(
      shape_param, shape_rotation_rad, num_div)
  num_poly = x_poly.shape[0]
  idx_to_delete = []
  for p in range(num_poly):
    xy = np.vstack((x_poly[p, :], y_poly[p, :])).T
    pgn = Polygon(xy)
    if (pgn.difference(bb).area > 0.) and prune_out_intersecting_shapes:
      idx_to_delete.append(p)
      continue
    polygons.append(pgn.intersection(bb))
  pruned_shape_param = SuperShapes(a=np.delete(shape_param.a, idx_to_delete),
                                   b=np.delete(shape_param.b, idx_to_delete),
                                   m=np.delete(shape_param.m, idx_to_delete),
                                   n1=np.delete(shape_param.n1, idx_to_delete),
                                   n2=np.delete(shape_param.n2, idx_to_delete),
                                   n3=np.delete(shape_param.n3, idx_to_delete),
                                   center_x=np.delete(shape_param.center_x, idx_to_delete),
                                   center_y=np.delete(shape_param.center_y, idx_to_delete))
  return polygons, pruned_shape_param
  

def project_shapely_polygons_to_density(polygons: list[Polygon], nelx: int,
                                        nely: int,
                  flip_inside_out: bool = False) -> np.ndarray:
  
  """
  Project Shapely polygons onto a discretized density field.

  Args:
    polygons: List of Shapely Polygon objects.
    nelx: Number of elements in the x-direction.
    nely: Number of elements in the y-direction.
    flip_inside_out: If True, flip the inside-outside classification.

  Returns:
    np.ndarray: Array of shape (num_polys, nelx, nely) with values in [0, 1].
  """
 
  bounding_box = _BoundingBox(x_min=-1., x_max=1., y_min=-1., y_max=1.)
  dx, dy = 2/nelx, 2/nely
  [x_grid, y_grid] = np.meshgrid(
               np.linspace(bounding_box.x_min + dx/2.,
                           bounding_box.x_max-dx/2.,
                           nelx),
               np.linspace(bounding_box.y_min + dy/2.,
                           bounding_box.y_max-dy/2.,
                           nely))
  elem_centers = np.stack((x_grid, y_grid)).T.reshape((-1, 2))
  num_blobs = len(polygons)
  density = np.zeros((num_blobs, nelx, nely))
  points = shapely.points(elem_centers)
  for b in range(num_blobs):
    density[b,:,:] = 1.*polygons[b].contains(points).reshape(
                                          (nelx, nely))

  if flip_inside_out:
    return 1. - density
  return density

def get_euclidean_coords_of_points_on_surf_super_shape(
    shape_param: SuperShapes, shape_rotation_rad: np.ndarray=None, num_div: int=300)-> Tuple[np.ndarray, np.ndarray]:
  """
  Args:
    shape_param: Contains the shape parameters of `num_shapes` supershapes.
    shape_rotation: A rigid body rotation (in rad) of the shapes about its
      origin. If None, then its assumed to be zero. Else pass an array of size
      (num_shapes,)
    num_div: An integer that dictates how close the points obtained are.
  Returns: A tuple of (x, y) each of size (num_shapes, num_div) that contains
    the x and y coordinates of the points on the boundary respectively.
  """
  if shape_rotation_rad is None:
    shape_rotation_rad = np.zeros((shape_param.num_shapes,))
  theta = np.tile(np.linspace(0.0, 2*np.pi, num_div),
                      (shape_param.num_shapes, 1))
  radius = compute_radius_of_super_shape(theta, shape_param)
  x = (shape_param.center_x[:, np.newaxis] +
       radius*np.cos(theta + shape_rotation_rad[:, np.newaxis]))
  y = (shape_param.center_y[:, np.newaxis] +
       radius*np.sin(theta + shape_rotation_rad[:, np.newaxis]))
  return x, y


def plot_super_shape(shape_param: SuperShapes, shape_no:int=0)-> None:
  """Plot a supershape.

    Args:
        shape_param: Contains the shape parameters of type supershapes.
        shape_no: The index of the shape to plot.
    """
  x, y = get_euclidean_coords_of_points_on_surf_super_shape(shape_param)
  ax = plt.subplot(111)
  ax.patch.set_facecolor('#DAE8FC') # blue
  ax.fill(x[shape_no, :], y[shape_no, :], facecolor='#F8CECC', edgecolor='black')
  ax.set_xlim([shape_param.bounding_box.x_min, shape_param.bounding_box.x_max])
  ax.set_ylim([shape_param.bounding_box.y_min, shape_param.bounding_box.y_max])


def generate_random_super_shapes(num_shapes: int,
                                 shape_extents:SuperShapeExtents,
                                 seed:int=27)->SuperShapes:
  """Generate random SuperShapes.

    Args:
        num_shapes: Number of SuperShapes to generate.
        shape_extents: Extents for the parameters of the SuperShapes.
        seed: Random seed for reproducibility.

    Returns:
        SuperShapes: Randomly generated SuperShapes parameters.

    """
  rng = np.random.default_rng(seed)
  a = rng.uniform(shape_extents.a.min_val, shape_extents.a.max_val,
                   (num_shapes,))
  b = rng.uniform(shape_extents.b.min_val, shape_extents.b.max_val,
                   (num_shapes,))
  m =rng.uniform(shape_extents.m.min_val, shape_extents.m.max_val,     
                   (num_shapes,))                                   
  n1 = rng.uniform(shape_extents.n1.min_val, shape_extents.n1.max_val,
                   (num_shapes,))
  n2 = rng.uniform(shape_extents.n2.min_val, shape_extents.n2.max_val,
                   (num_shapes,))
  n3 = rng.uniform(shape_extents.n3.min_val, shape_extents.n3.max_val,
                   (num_shapes,))
  cx = rng.uniform(shape_extents.center_x.min_val, shape_extents.center_x.max_val,
                   (num_shapes,))
  cy = rng.uniform(shape_extents.center_y.min_val, shape_extents.center_y.max_val,
                   (num_shapes,))
  return SuperShapes(a, b, m, n1, n2, n3, cx, cy)