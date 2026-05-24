"""Train the TOMAS VAE on the 7000-sample dataset (dataset_num=2).

Outputs to vae/vae_net_v2.pt and vae/normalization_v2.pt — does NOT overwrite
the legacy 100-sample weights.
"""
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.io
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "vae")]

import vae.data_preprocess as data_preprocess
import vae.network as network
import vae.train_vae as train_vae


def main():
  with open(ROOT / "notebooks" / "vae_config.yaml", "r", encoding="utf-8") as f:
    vae_config = yaml.safe_load(f)
  with open(ROOT / "notebooks" / "datagen.yaml", "r", encoding="utf-8") as f:
    data_config = yaml.safe_load(f)

  dataset_num = data_config["DATASET"]["dataset_num"]
  print(f"Loading dataset_num={dataset_num} ...", flush=True)
  shape_params = scipy.io.loadmat(
      ROOT / "dataset" / f"mstr_shape_parameters_{dataset_num}.mat"
      )["mstr_shape_parameters"]
  homog = scipy.io.loadmat(
      ROOT / "dataset" / f"homogen_data_{dataset_num}.mat")
  area = scipy.io.loadmat(
      ROOT / "dataset" / f"mstr_area_{dataset_num}.mat")["mstr_area"]
  perim = scipy.io.loadmat(
      ROOT / "dataset" / f"mstr_perim_{dataset_num}.mat")["mstr_perim"]
  c00 = homog["c00"].reshape(-1, 1)
  c11 = homog["c11"].reshape(-1, 1)
  area = np.asarray(area).reshape(-1, 1)
  perim = np.asarray(perim).reshape(-1, 1)

  num_samples = shape_params.shape[0]
  print(f"  shape_params : {shape_params.shape}")
  print(f"  c00 / c11    : {c00.shape} / {c11.shape}")
  print(f"  area / perim : {area.shape} / {perim.shape}")
  print(f"  total samples: {num_samples}")

  # Some samples may have degenerate (area==0) polygons (super-shape pruned
  # to zero by Shapely). Drop them before normalization to avoid log(0).
  valid = ((c00.flatten() > 0)
           & (c11.flatten() > 0)
           & (area.flatten() > 0)
           & (perim.flatten() > 0))
  if not valid.all():
    print(f"  filtering out {(~valid).sum()} degenerate rows "
          f"(c<=0 or area/perim<=0)")
    shape_params = shape_params[valid]
    c00 = c00[valid]
    c11 = c11[valid]
    area = area[valid]
    perim = perim[valid]
    num_samples = shape_params.shape[0]

  mstr_data = torch.tensor(np.hstack((shape_params, c00, c11, perim, area))
                           ).double()
  # 8 LINEAR (a,b,m,n1,n2,n3,cx,cy) + 2 LOG (c00,c11) + 2 LINEAR (perim,area)
  norm_types = (
      [data_preprocess.NomalizationType.LINEAR] * 8
      + [data_preprocess.NomalizationType.LOG] * 2
      + [data_preprocess.NomalizationType.LINEAR] * 2
  )
  normalized, max_feat, min_feat = data_preprocess.stack_train_data(
      mstr_data, norm_types)
  print(f"  normalized data  : {normalized.shape}")
  print(f"  max_feature      : {max_feat.numpy().round(4)}")
  print(f"  min_feature      : {min_feat.numpy().round(4)}")

  norm_path = ROOT / "vae" / "normalization_v2.pt"
  torch.save({"max_feature": max_feat, "min_feature": min_feat}, norm_path)
  print(f"Wrote {norm_path}")

  vae_yaml = vae_config["NETWORK"]
  vae_params = network.VAE_Params(
      input_dim=normalized.shape[1],
      encoder_hidden_dim=vae_yaml["encoder_hidden_dim"],
      latent_dim=vae_yaml["latent_dim"],
      decoder_hidden_dim=vae_yaml["decoder_hidden_dim"],
  )
  vae_net = network.VariationalAutoencoder(vae_params=vae_params)

  net_path = ROOT / "vae" / "vae_net_v2.pt"
  opt_yaml = vae_config["OPTIMIZATION"]
  start = time.perf_counter()
  history = train_vae.train_autoencoder(
      vae=vae_net,
      train_data=normalized,
      num_epochs=opt_yaml["num_epochs"],
      kl_factor=opt_yaml["kl_factor"],
      lr=opt_yaml["lr"],
      save_file=str(net_path),
  )
  elapsed = time.perf_counter() - start
  print(f"\nTraining done in {elapsed / 60:.1f} min. Saved -> {net_path}")

  # Reconstruction sanity check on the training set
  vae_net.encoder.is_training = False
  vae_net.eval()
  with torch.no_grad():
    recon_norm = vae_net(normalized)
  recon = data_preprocess.stack_vae_output(
      recon_norm, max_feat, min_feat, norm_types)
  orig = data_preprocess.stack_vae_output(
      normalized, max_feat, min_feat, norm_types)
  abs_err = (recon - orig).abs().mean(dim=0).numpy()
  rel_err = ((recon - orig).abs() / (orig.abs() + 1e-12)).mean(dim=0).numpy()
  field_names = ["a", "b", "m", "n1", "n2", "n3", "cx", "cy",
                 "C00", "C11", "perim", "area"]
  print("\nReconstruction error on training set:")
  for n, ae, re in zip(field_names, abs_err, rel_err):
    print(f"  {n:6s}  mean_abs_err = {ae:.4E}   mean_rel_err = {re:.4E}")

  # Save loss curves
  fig, ax = plt.subplots(figsize=(8, 4))
  ax.semilogy(history["recon_loss"], label="recon_loss")
  ax.semilogy(history["kl_loss"], label="kl_loss")
  ax.semilogy(history["loss"], label="total")
  ax.set_xlabel("epoch"); ax.legend(); ax.grid(True, alpha=0.3)
  ax.set_title("VAE v2 training (7000 samples)")
  fig.tight_layout()
  out_png = ROOT / "vae" / "training_history_v2.png"
  fig.savefig(out_png, dpi=180)
  plt.close(fig)
  print(f"Wrote {out_png}")


if __name__ == "__main__":
  main()
