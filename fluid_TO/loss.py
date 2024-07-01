"""
__author__ = "Rahul Kumar Padhy, Aaditya Chandrasekhar, and Krishnan Suresh"
__email__ = "rkpadhy@wisc.edu, cs.aaditya@gmail.com and ksuresh@wisc.edu"
"""

""" Defines functions to convert constraint optimization
  problem to a unconstraint loss.
"""
from dataclasses import dataclass
from enum import Enum, auto
import torch
import numpy as np

class LossTypes(Enum):
  PENALTY = auto()
  LOG_BARRIER = auto()
  AUG_LAG = auto()

@dataclass
class PenaltyLossParameters:
  alpha0: float
  del_alpha: float

  def get_alpha(self, epoch: int):
    return self.alpha0 + epoch*self.del_alpha

@dataclass
class LogBarrierLossParameters:
  t0: float
  mu: float

  def get_t(self, epoch):
    return self.t0*self.mu**epoch

@dataclass
class AugmentedLagrangianParameters:
  alpha0 : float
  del_alpha: float
  alpha: float
  lamda: np.ndarray

def combined_loss(objective: float,
                  constraints: list[float],
                  loss_type: LossTypes,
                  loss_params,
                  epoch: int):
  """ Compute unconstrained loss term for a constrained optimization problem.
      
      The constraint optimization problem is of type
      min_(x) f(x)
      s.t. g_i(x) <= 0 , i = 1,2,...,N
      
      Note that we don't handle equality and box constraints
      LOG_BARRIER: Handles inequality constraints 
      AUG_LAGRANGIAN AND PENALTY: Handles equality constraints 
      
  Args:
    objective: float that is the objective of the problem
    constraints: list of size N of floats that are the constraint values
    loss_type: Supports augmented lagragian, penalty and log barrier methods
    loss_params: parameters that is correspondent of the loss_type
    epoch: integer of the current iteration number.
  Returns: a float that is the combined loss of the objective and constraints
  """
  if(loss_type == LossTypes.PENALTY):
    loss = objective
    alpha = loss_params.get_alpha(epoch)
    for c in constraints:
      loss = loss + alpha*c**2
  if(loss_type == LossTypes.LOG_BARRIER):
    loss = objective
    t = loss_params.get_t(epoch)
    for c in constraints:
      if(c < (-1/t**2)):
        loss = loss - torch.log(-c)/t
      else:
        loss = loss + (t*c - np.log(1/t**2)/t + 1./t)
  if(loss_type == LossTypes.AUG_LAG):
    loss = objective
    for i,c in enumerate(constraints):
      loss = loss + loss_params.alpha*c**2 + loss_params.lamda[i]*c
    
  return loss

def update_loss_parameters(epoch: int,
                          loss_type: LossTypes,
                          loss_params, constraints):
  """Update the loss parameters based on the current epoch and loss type.

  Args:
      epoch: The current epoch.
      loss_type: The type of loss.
      loss_params: Object containing the loss parameters.
      constraints: List of constraints.

  Returns:
      None
    """
  if(loss_type == LossTypes.PENALTY):
    loss_params.alpha = min(100.,loss_params.alpha0 +
              epoch*loss_params.del_alpha)
  if(loss_type == LossTypes.LOG_BARRIER):
    loss_params.t = loss_params.t0*loss_params.mu**epoch
  if(loss_type==LossTypes.AUG_LAG):
    loss_params.alpha = min(100,  loss_params.alpha0 +
                            (epoch)*loss_params.del_alpha)
    for i,c in enumerate(constraints):
      loss_params.lamda[i] = loss_params.lamda[i] +  loss_params.alpha*2*c.item()
      