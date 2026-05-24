"""Compose a montage of paper-figure reproductions."""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]


def add(ax, path, title):
  if path is None or not path.exists():
    ax.text(0.5, 0.5, f"(missing)\n{title}",
            ha="center", va="center", fontsize=10)
    ax.axis("off")
    return
  ax.imshow(mpimg.imread(path))
  ax.set_title(title, fontsize=10)
  ax.axis("off")


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--root", default=str(ROOT / "output"
                                            / "paper_reproduction"))
  parser.add_argument("--out", default=None)
  args = parser.parse_args()
  root = Path(args.root)
  out_path = Path(args.out) if args.out else root / "figures_montage.png"

  panels = [
      (root / "ideal_microstructure" / "latent_scatter.png",
       "Fig.10 latent (§3.1)"),
      (root / "ideal_microstructure" / "shape_preview.png",
       "Fig.10 selected mstr"),
      (root / "bent_pipe_fixed_mstr" / "final_topology.png",
       "Fig.11d bent pipe fixed mstr"),
      (root / "bent_pipe_volume" / "final_topology.png",
       "Fig.12a bent pipe vol=0.75"),
      (root / "bent_pipe_perim" / "final_topology.png",
       "Fig.12b bent pipe area=75.69"),
      (root / "diffuser_a60" / "final_topology.png",
       "Fig.13 diffuser final"),
      (root / "diffuser_pareto" / "pareto.png", "Fig.14 Pareto"),
      (root / "bifurcated_pipe" / "final_topology.png",
       "Fig.16b bifurcated"),
  ]
  fig, axes = plt.subplots(2, 4, figsize=(20, 10))
  for ax, (path, title) in zip(axes.flatten(), panels):
    add(ax, path, title)
  fig.suptitle("TOMAS paper figure reproduction", fontsize=14)
  fig.tight_layout()
  fig.savefig(out_path, dpi=180)
  plt.close(fig)
  print(f"Wrote {out_path}")


if __name__ == "__main__":
  main()
