"""Train the TOMAS VAE on a FILTERED subset of the 7000-sample dataset.

v3 changes vs v2:
  - Drops degenerate samples (area < 0.10 or perim < 1.0) to avoid teaching the
    VAE that "near-empty" shapes are normal. Without this filter, the global
    contact-area / volume constraints can be satisfied by collapsing each cell's
    microstructure to almost nothing.

Outputs to vae/vae_net_v3.pt and vae/normalization_v3.pt — does NOT touch v1
or v2 artifacts.
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


AREA_MIN = 0.10
PERIM_MIN = 1.0


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
  print(f"  raw count: {total}")

  area_flat = area.flatten()
  perim_flat = perim.flatten()
  print(f"\nBefore filtering:")
  print(f"  area  : min={area_flat.min():.4f}  median={np.median(area_flat):.4f}"
        f"  max={area_flat.max():.4f}  "
        f"<{AREA_MIN}: {int((area_flat < AREA_MIN).sum())} "
        f"({100*(area_flat < AREA_MIN).mean():.1f}%)")
  print(f"  perim : min={perim_flat.min():.4f}  "
        f"median={np.median(perim_flat):.4f}  max={perim_flat.max():.4f}  "
        f"<{PERIM_MIN}: {int((perim_flat < PERIM_MIN).sum())} "
        f"({100*(perim_flat < PERIM_MIN).mean():.1f}%)")

  valid = ((c00.flatten() > 0)
           & (c11.flatten() > 0)
           & (area_flat >= AREA_MIN)
           & (perim_flat >= PERIM_MIN))
  print(f"\nFilter (area>={AREA_MIN}, perim>={PERIM_MIN}, c00>0, c11>0): "
        f"keep {int(valid.sum())} / {total} "
        f"({100*valid.mean():.1f}%)", flush=True)

  shape_params = shape_params[valid]
  c00 = c00[valid]
  c11 = c11[valid]
  area = area[valid]
  perim = perim[valid]
  num_samples = shape_params.shape[0]

  area_flat = area.flatten()
  perim_flat = perim.flatten()
  print(f"\nAfter filtering:")
  print(f"  area  : min={area_flat.min():.4f}  median={np.median(area_flat):.4f}"
        f"  max={area_flat.max():.4f}")
  print(f"  perim : min={perim_flat.min():.4f}  "
        f"median={np.median(perim_flat):.4f}  max={perim_flat.max():.4f}")

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
  print(f"\n  normalized data  : {normalized.shape}")
  print(f"  max_feature      : {max_feat.numpy().round(4)}")
  print(f"  min_feature      : {min_feat.numpy().round(4)}")

  norm_path = ROOT / "vae" / "normalization_v3.pt"
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

  net_path = ROOT / "vae" / "vae_net_v3.pt"
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

  # Reconstruction sanity check on the (filtered) training set
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
  print(f"\nReconstruction error on filtered training set "
        f"({num_samples} samples):")
  for n, ae, re in zip(field_names, abs_err, rel_err):
    print(f"  {n:6s}  mean_abs_err = {ae:.4E}   mean_rel_err = {re:.4E}")

  fig, ax = plt.subplots(figsize=(8, 4))
  ax.semilogy(history["recon_loss"], label="recon_loss")
  ax.semilogy(history["kl_loss"], label="kl_loss")
  ax.semilogy(history["loss"], label="total")
  ax.set_xlabel("epoch"); ax.legend(); ax.grid(True, alpha=0.3)
  ax.set_title(f"VAE v3 training ({num_samples} filtered samples)")
  fig.tight_layout()
  out_png = ROOT / "vae" / "training_history_v3.png"
  fig.savefig(out_png, dpi=180)
  plt.close(fig)
  print(f"Wrote {out_png}")


if __name__ == "__main__":
  main()
