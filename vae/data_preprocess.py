import torch
import numpy as np
from enum import Enum, auto
from typing import List, Tuple

class NomalizationType(Enum):
  LINEAR = auto()
  LOG = auto()

class VAE_Fields(Enum):
  shape_a = 0
  shape_b = 1
  shape_m = 2
  shape_n1 = 3
  shape_n2 = 4
  shape_n3 = 5
  shape_cx = 6
  shape_cy = 7
  homog_c00 = 8
  homog_c11 = 9
  shape_perim = 10
  shape_area = 11
  
def normalize_data(input_data: torch.Tensor,
                   normalization_type
                   )->tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Args: Array of size (num_samples, num_features) where each of the sample
      has features in similar range
    Returns: A tuple of (normalized_data, max_feature, min_feature). The
      entries in normalized_data is an array of shape (num_samples, num_features)
      with all entries in [0, 1]. max_feature and min_feature are arrays of
      shape (num_features,) that contain the maximum and minium entry of the
      input data array respectively. This information is necessary to retrieve
      the renormalized data from the normalized data.
    """
    if (normalization_type == NomalizationType.LINEAR):
      max_feature = torch.amax(input_data, dim=0)
      min_feature = torch.amin(input_data, dim=0)
      #TODO: ensure max not equal to min to avoid div by zero
      normalized_data = (input_data - min_feature)/(max_feature - min_feature)
    elif (normalization_type == NomalizationType.LOG):
      logarithmic_input_data = torch.log10(input_data)
      max_feature = torch.amax(logarithmic_input_data, dim=0)
      min_feature = torch.amin(logarithmic_input_data, dim=0)
      normalized_data = (logarithmic_input_data - min_feature)/(max_feature - min_feature)
    return normalized_data, max_feature, min_feature

def renormalize_data(normalized_data: torch.Tensor, max_feature: torch.Tensor,
                     min_feature: torch.Tensor,
                     normalization_type)->torch.Tensor:
    """
    Args:
      normalized_data:
      max_feature:
      min_feature:
    Returns: Array of size (num_samples, num_features) where all the entries of
      `normalized_data` have been rescaled to their input scale.
    """
    
    if (normalization_type == NomalizationType.LINEAR):
      renormalized_data = (normalized_data*(max_feature - min_feature) + min_feature)
    elif (normalization_type == NomalizationType.LOG):
      renormalized_data = 10.**(normalized_data*(max_feature - min_feature) + min_feature)
    return renormalized_data
  
  

def stack_train_data(mstr_data: torch.Tensor, normalization_types: List[NomalizationType]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  """
  Stack training data and compute normalization parameters.

  Args:
    mstr_data: Input tensor of shape (num_samples, num_features).
    normalization_types: List of normalization types for each feature.

  Returns:
    Tuple containing normalized parameters, max parameters, and min parameters.
  """
  num_samples, num_features = mstr_data.shape[0], mstr_data.shape[1]
  normalized_params = torch.zeros((num_samples, num_features)).double()
  min_params = torch.zeros((num_features,))
  max_params = torch.zeros((num_features,))

  index =  0
  for normalization_type in normalization_type:
    (normalized_params[:, index:index+1],
    max_params[index:index+1],
    min_params[index:index+1]) = normalize_data(
        mstr_data[:, index:index+1],
        normalization_type)
    index = index + 1


  return normalized_params, max_params, min_params


def stack_vae_output(vae_output:torch.Tensor, max_feature: torch.Tensor,
                      min_feature:torch.Tensor, normalization_type: List[NomalizationType])-> torch.Tensor:
  """
  Stack VAE output and renormalize it.

  Args:
    vae_output: Output tensor of shape (num_samples, num_features).
    max_features: Max parameters for each feature.
    min_features: Min parameters for each feature.
    normalization_types: List of normalization types for each feature.

  Returns:
    torch.Tensor: Renormalized VAE output tensor.
  """
  index = 0
  # Iterate through the normalization types
  for normalization_type in normalization_type:
    renormalized_param = renormalize_data(vae_output[:, index:index+1],
                                       max_feature[ index:index+1], min_feature[index:index+1],
                                       normalization_type)
    if (index == 0):
      renormalized_vae_output = renormalized_param
    else:
      renormalized_vae_output = torch.hstack((renormalized_vae_output,renormalized_param))
    index = index + 1
  return renormalized_vae_output