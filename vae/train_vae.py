import torch
import network

def train_autoencoder(vae: network.VariationalAutoencoder,
                      train_data: torch.Tensor,
                      num_epochs: int,
                      kl_factor: float,
                      lr: float,
                      save_file: str,
                      print_every: int = 500)->dict:
  """Train the variational autoencoder.

  Args:
    vae: The variational autoencoder model.
    train_data: The training data.
    num_epochs: Number of training epochs.
    kl_factor: Weight factor for the Kullback-Leibler divergence loss.
    lr: Learning rate for the optimizer.
    save_file: File path to save the trained model.
    print_every: Frequency of printing training progress. Defaults to 500.

  Returns:
    dict: Dictionary containing the training history.
  """
  opt = torch.optim.Adam(vae.parameters(), lr)
  convg_history = {'recon_loss':[], 'kl_loss':[], 'loss':[]}
  vae.encoder.is_training = True
  for epoch in range(num_epochs):
    opt.zero_grad()
    pred_data = vae(train_data)
    kl_loss = vae.encoder.kl
    recon_loss =  ((train_data - pred_data)**2).mean()
    loss = recon_loss + kl_factor*kl_loss 
    loss.backward()
    convg_history['recon_loss'].append(recon_loss.item())
    convg_history['kl_loss'].append(kl_loss.item())
    convg_history['loss'].append(loss.item())
    opt.step()
    if epoch%print_every == 0:
      print(f'iter {epoch:d} \t recon_loss \t {recon_loss.item():.2E}'
            f' \t kl_loss {kl_loss.item():.2E} \t net_loss {loss.item():.2E}')

  vae.encoder.is_training = False
  save_vae(vae, save_file)
  return convg_history

def save_vae(vae: network.VariationalAutoencoder, file_name: str):
  """Save the trained variational autoencoder.

  Args:
      vae: The variational autoencoder model.
      file_name: File path to save the trained model.

  Returns:
      None
  """
  torch.save(vae.state_dict(), file_name)

def load_vae_weigths(vae: network.VariationalAutoencoder, file_name: str):
  """Load weights for the variational autoencoder.

    Args:
      vae: The variational autoencoder model.
      file_name: File path to load the weights from.

    Returns:
      None
    """
  vae.encoder.is_training = False
  vae.load_state_dict(torch.load(file_name))
  vae.eval()
