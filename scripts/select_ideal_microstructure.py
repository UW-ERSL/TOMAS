"""TOMAS paper §3.1: pick the microstructure with vol_frac ~ 0.25 that maximizes
trace(C) = C00 + C11. Samples the decoder on a 200x200 grid in latent space
[-3, 3]^2 and exports the selected micro-structure + a latent-space figure that
mimics Fig.10.
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "dataset"), str(ROOT / "vae")]

import dataset.supershape as supershape
import vae.data_preprocess as vae_data_prep
import vae.network as vae_network


VAE_FILES = {
    "v1": ("vae_net.pt", "nomalization.pt"),
    "v2": ("vae_net_v2.pt", "normalization_v2.pt"),
    "v3": ("vae_net_v3.pt", "normalization_v3.pt"),
    "v4": ("vae_net_v4.pt", "normalization_v4.pt"),
    "v5": ("vae_net_v5.pt", "normalization_v5.pt"),
    "v6": ("vae_net_v6.pt", "normalization_v6.pt"),
}
VAE_HIDDEN_OVERRIDE = {"v6": 1200}


def load_vae(version):
  with open(ROOT / "notebooks" / "vae_config.yaml", "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)["NETWORK"]
  hidden = VAE_HIDDEN_OVERRIDE.get(version, cfg["encoder_hidden_dim"])
  params = vae_network.VAE_Params(
      input_dim=12,
      encoder_hidden_dim=hidden,
      latent_dim=cfg["latent_dim"],
      decoder_hidden_dim=hidden,
  )
  model = vae_network.VariationalAutoencoder(params)
  model.encoder.is_training = False
  net_name, norm_name = VAE_FILES[version]
  model.load_state_dict(torch.load(ROOT / "vae" / net_name))
  model.eval()
  norm = torch.load(ROOT / "vae" / norm_name)
  normalization_types = (
      [vae_data_prep.NomalizationType.LINEAR] * 8
      + [vae_data_prep.NomalizationType.LOG] * 2
      + [vae_data_prep.NomalizationType.LINEAR] * 2
  )
  return model, norm["max_feature"], norm["min_feature"], normalization_types


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--vae-version", default="v6", choices=list(VAE_FILES))
  parser.add_argument("--out",
                      default=str(ROOT / "output" / "paper_reproduction"
                                  / "ideal_microstructure"))
  parser.add_argument("--grid", type=int, default=200,
                      help="Grid resolution per latent axis (paper uses 200).")
  parser.add_argument("--vol-target", type=float, default=0.25)
  parser.add_argument("--vol-tol", type=float, default=0.001,
                      help="vol_frac kept if |vf - target| <= tol "
                           "(paper uses 0.001).")
  args = parser.parse_args()

  out_dir = Path(args.out)
  out_dir.mkdir(parents=True, exist_ok=True)

  vae_net, max_feat, min_feat, norm_types = load_vae(args.vae_version)

  axis = np.linspace(-3.0, 3.0, args.grid)
  z1, z2 = np.meshgrid(axis, axis, indexing="ij")
  latent = torch.tensor(np.stack([z1.ravel(), z2.ravel()], axis=1),
                        dtype=torch.float64)

  with torch.no_grad():
    decoded = vae_net.decoder(latent)
    renorm = vae_data_prep.stack_vae_output(decoded, max_feat, min_feat,
                                            norm_types).numpy()

  # column layout from data_preprocess.VAE_Fields:
  #   0:a, 1:b, 2:m, 3:n1, 4:n2, 5:n3, 6:cx, 7:cy, 8:C00, 9:C11, 10:perim,
  #   11:area (solid volume fraction)
  vol_frac = renorm[:, 11]
  c00 = renorm[:, 8]
  c11 = renorm[:, 9]
  trace_c = c00 + c11

  mask = np.abs(vol_frac - args.vol_target) <= args.vol_tol
  if not mask.any():
    raise RuntimeError(
        f"No latent samples satisfy |vol_frac - {args.vol_target}| <= "
        f"{args.vol_tol}. vol_frac range: "
        f"[{vol_frac.min():.4f}, {vol_frac.max():.4f}].")

  # Find argmax over the masked set
  masked_trace = np.where(mask, trace_c, -np.inf)
  best_flat = int(np.argmax(masked_trace))
  best_z1 = float(z1.ravel()[best_flat])
  best_z2 = float(z2.ravel()[best_flat])
  best_row = renorm[best_flat]

  # Print top-5 candidates for sanity (within the filtered set only)
  filtered_indices = np.where(mask)[0]
  top_k = filtered_indices[np.argsort(trace_c[filtered_indices])[-5:][::-1]]
  print(f"Filtered {mask.sum()} / {mask.size} latent points "
        f"with vol_frac in [{args.vol_target - args.vol_tol:.4f}, "
        f"{args.vol_target + args.vol_tol:.4f}].")
  print("Top 5 trace(C) within the filtered set:")
  for idx in top_k:
    print(f"  z=({z1.ravel()[idx]:+.4f}, {z2.ravel()[idx]:+.4f})  "
          f"trace_C={trace_c[idx]:.4f}  "
          f"C00={renorm[idx, 8]:.4f}  C11={renorm[idx, 9]:.4f}  "
          f"vf={renorm[idx, 11]:.4f}  perim={renorm[idx, 10]:.4f}  "
          f"M={renorm[idx, :6].round(4).tolist()}")

  shape_params = {
      "a":  float(best_row[0]),
      "b":  float(best_row[1]),
      "m":  float(best_row[2]),
      "n1": float(best_row[3]),
      "n2": float(best_row[4]),
      "n3": float(best_row[5]),
      "cx": float(best_row[6]),
      "cy": float(best_row[7]),
  }
  selected = {
      "vae_version": args.vae_version,
      "latent": [best_z1, best_z2],
      "shape_params": shape_params,
      "C00": float(best_row[8]),
      "C11": float(best_row[9]),
      "trace_C": float(best_row[8] + best_row[9]),
      "norm_perim": float(best_row[10]),
      "vol_frac": float(best_row[11]),
      "grid": args.grid,
      "vol_target": args.vol_target,
      "vol_tol": args.vol_tol,
      "paper_reported_M": {
          "a": 0.7158, "b": 0.3757, "m": 0.6039,
          "n1": 1.4787, "n2": 0.4349, "n3": 0.5857,
      },
  }
  with open(out_dir / "selected.json", "w", encoding="utf-8") as f:
    json.dump(selected, f, indent=2)
  print(f"\nSelected microstructure -> {out_dir / 'selected.json'}")
  print(json.dumps(selected, indent=2))

  # Latent-space scatter (Fig.10 mimic)
  fig, ax = plt.subplots(figsize=(6, 6))
  density_plot = ax.scatter(z1.ravel(), z2.ravel(),
                            c=trace_c, s=2, cmap="viridis", alpha=0.3)
  ax.scatter(z1.ravel()[mask], z2.ravel()[mask], c=trace_c[mask],
             s=8, cmap="plasma", edgecolors="k", linewidths=0.1)
  ax.scatter([best_z1], [best_z2], facecolors="red", edgecolors="black",
             s=200, marker="*", zorder=10,
             label=f"selected ({best_z1:+.3f}, {best_z2:+.3f})")
  fig.colorbar(density_plot, ax=ax, label="trace(C) (all latent)")
  ax.set_xlabel("z1")
  ax.set_ylabel("z2")
  ax.set_title(f"Ideal microstructure search (vol_frac~{args.vol_target})")
  ax.legend(loc="upper right", fontsize=8)
  ax.set_xlim(-3, 3); ax.set_ylim(-3, 3)
  ax.set_aspect("equal")
  fig.tight_layout()
  fig.savefig(out_dir / "latent_scatter.png", dpi=240)
  plt.close(fig)

  # Reconstructed shape preview
  recon = supershape.SuperShapes(
      a=np.array([shape_params["a"]]), b=np.array([shape_params["b"]]),
      m=np.array([shape_params["m"]]), n1=np.array([shape_params["n1"]]),
      n2=np.array([shape_params["n2"]]), n3=np.array([shape_params["n3"]]),
      center_x=np.array([shape_params["cx"]]),
      center_y=np.array([shape_params["cy"]]))
  x, y = supershape.get_euclidean_coords_of_points_on_surf_super_shape(recon)
  fig, ax = plt.subplots(figsize=(5, 5))
  ax.patch.set_facecolor("#DAE8FC")
  ax.fill(x[0, :], y[0, :], facecolor="#F8CECC", edgecolor="black")
  ax.set_xlim(recon.bounding_box.x_min, recon.bounding_box.x_max)
  ax.set_ylim(recon.bounding_box.y_min, recon.bounding_box.y_max)
  ax.set_aspect("equal")
  ax.set_title("Ideal microstructure (decoder reconstruction)")
  ax.set_xticks([]); ax.set_yticks([])
  fig.tight_layout()
  fig.savefig(out_dir / "shape_preview.png", dpi=240)
  plt.close(fig)


if __name__ == "__main__":
  main()
