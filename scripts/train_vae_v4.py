"""Train VAE v4: relaxed filtering vs v3.

Goal: VAE should be able to output BOTH small (channel-forming) AND large
(wall-hugging) microstructures, so the topology can develop genuine fluid
channels. The min_area_penalty in optimization will keep it from collapsing
to fully empty.

Filter:  area >= 0.03  and  perim >= 0.5   (vs v3's 0.10 / 1.0)
This still drops the most-degenerate shapes (rasterization near-empties) but
preserves enough small-mstr coverage to form channels.

Outputs: vae/vae_net_v4.pt, vae/normalization_v4.pt, vae/training_history_v4.png
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


AREA_MIN = 0.03
PERIM_MIN = 0.5


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

  total = shape_params.shape[0]
  area_flat = area.flatten()
  perim_flat = perim.flatten()

  valid = ((c00.flatten() > 0)
           & (c11.flatten() > 0)
           & (area_flat >= AREA_MIN)
           & (perim_flat >= PERIM_MIN))
  print(f"Filter (area>={AREA_MIN}, perim>={PERIM_MIN}, c00>0, c11>0): "
        f"keep {int(valid.sum())} / {total} "
        f"({100*valid.mean():.1f}%)", flush=True)

  shape_params = shape_params[valid]
  c00 = c00[valid]; c11 = c11[valid]
  area = area[valid]; perim = perim[valid]
  num_samples = shape_params.shape[0]

  print(f"\nAfter filtering:")
  print(f"  area  : min={area.min():.4f}  median={np.median(area):.4f}"
        f"  max={area.max():.4f}")
  print(f"  perim : min={perim.min():.4f}  median={np.median(perim):.4f}"
        f"  max={perim.max():.4f}")

  mstr_data = torch.tensor(np.hstack((shape_params, c00, c11, perim, area))
                           ).double()
  norm_types = (
      [data_preprocess.NomalizationType.LINEAR] * 8
      + [data_preprocess.NomalizationType.LOG] * 2
      + [data_preprocess.NomalizationType.LINEAR] * 2
  )
  normalized, max_feat, min_feat = data_preprocess.stack_train_data(
      mstr_data, norm_types)
  print(f"  normalized data  : {normalized.shape}")

  norm_path = ROOT / "vae" / "normalization_v4.pt"
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

  net_path = ROOT / "vae" / "vae_net_v4.pt"
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
  print(f"\nTraining done in {(time.perf_counter()-start)/60:.1f} min. "
        f"Saved -> {net_path}")

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
  for n, ae, re in zip(["a","b","m","n1","n2","n3","cx","cy","C00","C11",
                        "perim","area"], abs_err, rel_err):
    print(f"  {n:6s}  mean_abs_err = {ae:.4E}   mean_rel_err = {re:.4E}")

  fig, ax = plt.subplots(figsize=(8, 4))
  ax.semilogy(history["recon_loss"], label="recon_loss")
  ax.semilogy(history["kl_loss"], label="kl_loss")
  ax.semilogy(history["loss"], label="total")
  ax.set_xlabel("epoch"); ax.legend(); ax.grid(True, alpha=0.3)
  ax.set_title(f"VAE v4 training ({num_samples} filtered samples)")
  fig.tight_layout()
  fig.savefig(ROOT / "vae" / "training_history_v4.png", dpi=180)
  plt.close(fig)


if __name__ == "__main__":
  main()
