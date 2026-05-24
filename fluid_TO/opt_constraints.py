import torch
import numpy as np
from enum import Enum, auto

class ConstraintType(Enum):
  VOLUME = auto()
  PERIMETER = auto()

def constraint_function(constraint_type: int, field: torch.DoubleTensor,
                        desired_vol_frac:float, desired_perimeter:float)->torch.Tensor:
  """
  This function calls the compute fluid volume constraint.
  
  Args:
    constraint_type: Integer value determines which type of constraint to be imposed.
    fluid_vol_frac: Tensor of size (num_elems,) and represents the fluid volume fraction 
    occupied by each microstructure.
  
  Returns:
    vol_cons: Tensor of size (1,) representing the volume constraint. 
  """


  def compute_volume_constraint(solid_vol_frac: torch.DoubleTensor)->torch.Tensor:
    """
    This function constraints the fluid volume fraction by providing an upper limit as
    the desired_fluid_vol_frac.
    
    Args:
      solid_vol_frac: Tensor of size (num_elems,) and represents the solid volume fraction 
      occupied by each microstructure.
    
    Returns:
      fluid_vol_cons: Tensor of size (1,) representing the fluid volume constraint. 
    """
    fluid_vol_frac = 1. - solid_vol_frac
    fluid_vol_cons = (torch.mean(fluid_vol_frac)/(desired_vol_frac)) - 1.
    return fluid_vol_cons
  
  def compute_perimeter_constraint(perimeter: torch.DoubleTensor)->torch.Tensor:
    """
    This function constraints the fluid volume fraction by providing a lower limit as
    the desired_perimeter.
    
    Args:
      fluid_vol_frac: Tensor of size (num_elems,) and represents the perimeter
      of each microstructure.
    
    Returns:
      perim_cons: Tensor of size (1,) representing the perimeter constraint. 
    """
    perim_cons = 1. - (torch.sum(perimeter)/(desired_perimeter))
    return perim_cons
  if(constraint_type == ConstraintType.VOLUME):
    field_cons = compute_volume_constraint(field)
  elif(constraint_type == ConstraintType.PERIMETER):
    field_cons = compute_perimeter_constraint(field)
  return field_cons


def min_area_penalty(solid_vol_frac: torch.DoubleTensor,
                     min_solid_vol: float) -> torch.Tensor:
  """Soft per-cell penalty implementing TOMAS paper §3.7's "minimum area
  constraint on each microstructure".

  Pushes the optimizer away from solutions where individual cells degenerate
  to nearly empty super-shapes (and the decoder's small perim/area is exploited
  to satisfy global constraints).

  Args:
    solid_vol_frac: Tensor of shape (num_elems,) — per-cell solid volume
      fraction reconstructed by the VAE decoder.
    min_solid_vol: Lower bound that each cell should satisfy. Set to 0 to
      disable.

  Returns:
    Mean of squared violations across cells. Always ≥ 0; equals 0 when every
    cell satisfies solid_vol_frac ≥ min_solid_vol.
  """
  if min_solid_vol <= 0.0:
    return torch.zeros(1, dtype=solid_vol_frac.dtype,
                       device=solid_vol_frac.device).squeeze()
  violation = torch.clamp(min_solid_vol - solid_vol_frac, min=0.0)
  return (violation ** 2).mean() 