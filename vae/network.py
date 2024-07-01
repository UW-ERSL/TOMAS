#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def set_seed(seed: int = 27):
  """
  Set seed for reproducibility.

  Args:
      seed (int): Seed value for random number generators.
  """
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  np.random.seed(seed)

@dataclass
class VAE_Params:
  input_dim: int
  encoder_hidden_dim: int
  latent_dim: int
  decoder_hidden_dim: int
  
  @property
  def output_dim(self):
    """Output dimensionality, same as the input dimension."""
    return self.input_dim


class Encoder(nn.Module):
  def __init__(self, vae_params: VAE_Params):
    super(Encoder, self).__init__()
    set_seed(1234)
    self.linear1 = nn.Linear(vae_params.input_dim, vae_params.encoder_hidden_dim,
                             dtype = torch.float64)
    nn.init.xavier_normal_(self.linear1.weight)
    nn.init.zeros_(self.linear1.bias)
    self.linear2 = nn.Linear(vae_params.encoder_hidden_dim, vae_params.encoder_hidden_dim,
                             dtype = torch.float64)
    nn.init.xavier_normal_(self.linear2.weight)
    nn.init.zeros_(self.linear2.bias)
    self.linear3 = nn.Linear(vae_params.encoder_hidden_dim, vae_params.latent_dim,
                             dtype = torch.float64)
    nn.init.xavier_normal_(self.linear3.weight)
    nn.init.zeros_(self.linear3.bias)
    self.linear4 = nn.Linear(vae_params.encoder_hidden_dim, vae_params.latent_dim,
                             dtype = torch.float64)
    nn.init.xavier_normal_(self.linear4.weight)
    nn.init.zeros_(self.linear4.bias)
    self.normal_dist = torch.distributions.Normal(0, 1)
    self.kl = 0
    self.is_training = False

  def forward(self, x: torch.Tensor)-> torch.Tensor:
    """
      Forward pass of the encoder network.

      Args:
        x: Input tensor of shape (batch_size, input_dim).

      Returns:
        torch.Tensor: Latent representation tensor of shape (batch_size, latent_dim).
    """
    x = F.leaky_relu(self.linear1(x))
    x = F.leaky_relu(self.linear2(x))
    mu =  self.linear3(x)
    sigma = torch.exp(self.linear4(x))
    if self.is_training:
      self.z = mu + sigma*self.normal_dist.sample(mu.shape)
    else:
      self.z = mu
    self.kl = (sigma**2 + mu**2 - torch.log(sigma) - 1/2).sum()
    return self.z

class Decoder(nn.Module):
  def __init__(self, vae_params: VAE_Params):
    super(Decoder, self).__init__()
    set_seed(1234)
    self.linear1 = nn.Linear(vae_params.latent_dim, vae_params.decoder_hidden_dim,
                             dtype = torch.float64)
    nn.init.xavier_normal_(self.linear1.weight)
    nn.init.zeros_(self.linear1.bias)
    self.linear2 = nn.Linear(vae_params.decoder_hidden_dim, vae_params.decoder_hidden_dim,
                             dtype = torch.float64)
    nn.init.xavier_normal_(self.linear2.weight)
    nn.init.zeros_(self.linear2.bias)
    self.linear3 = nn.Linear(vae_params.decoder_hidden_dim, vae_params.output_dim,
                             dtype = torch.float64)
    nn.init.xavier_normal_(self.linear3.weight)
    nn.init.zeros_(self.linear3.bias)

  def forward(self, z)-> torch.Tensor:
    """
    Forward pass of the decoder network.

    Args:
      z: Latent representation tensor of shape (batch_size, latent_dim).

    Returns:
      torch.Tensor: Output tensor of the decoder network in the range [0, 1].
    """
    z = F.leaky_relu(self.linear1(z))
    z = F.leaky_relu(self.linear2(z))
    z = torch.sigmoid(self.linear3(z)) # decoder op in range [0,1]
    return z

class VariationalAutoencoder(nn.Module):
  def __init__(self, vae_params: VAE_Params):
    super(VariationalAutoencoder, self).__init__()
    set_seed(1234)
    self.encoder = Encoder(vae_params)
    self.decoder = Decoder(vae_params)

  def forward(self, x)-> torch.Tensor:
    """
    Forward pass of the variational autoencoder.

    Args:
      x: Input tensor of shape (batch_size, input_dim).

    Returns:
      torch.Tensor: Reconstructed output tensor of shape (batch_size, output_dim).
    """
    z = self.encoder(x)
    return self.decoder(z)