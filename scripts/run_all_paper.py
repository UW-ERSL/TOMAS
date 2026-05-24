"""Top-level driver: trigger every TOMAS paper case once Phase A (dataset +
MATLAB homogenization + VAE training) is done.

This script assumes:
  - dataset/homogen_data_2.mat exists (MATLAB Phase A3 done)
  - vae/vae_net_v2.pt + vae/normalization_v2.pt exist (Phase A4 done)
  - output/paper_reproduction/ideal_microstructure/selected.json exists
    (Phase B3 done)

Run the missing prerequisites explicitly first; this driver only wires the
Phase C-F runs.
"""
import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
PR_OUT = ROOT / "output" / "paper_reproduction"


def run(args, log):
  log_path = ROOT / "logs" / f"{log}.log"
  log_path.parent.mkdir(parents=True, exist_ok=True)
  print(f"\n[run_all] >>> {' '.join(str(a) for a in args)}\n  log: {log_path}",
        flush=True)
  with open(log_path, "w", encoding="utf-8") as f:
    p = subprocess.Popen(args, stdout=f, stderr=subprocess.STDOUT, cwd=ROOT)
    p.wait()
  if p.returncode != 0:
    raise RuntimeError(f"Subcommand failed (exit {p.returncode}). "
                       f"See {log_path}")


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--vae-version", default="v6")
  parser.add_argument("--skip", nargs="*", default=[],
                      choices=["c1", "c2", "c3", "d1", "d2", "e", "f"])
  args = parser.parse_args()
  selected = (PR_OUT / "ideal_microstructure" / "selected.json")
  if not selected.exists():
    sys.exit(f"Missing {selected}. Run select_ideal_microstructure.py first.")

  def skipped(name):
    if name in args.skip:
      print(f"[run_all] skipping {name}", flush=True)
      return True
    return False

  if not skipped("c1"):
    run([PY, "-u", "scripts/run_bent_pipe_fixed_mstr.py",
         "--config", "notebooks/config_bent_pipe.yaml",
         "--selected-mstr", str(selected),
         "--out", str(PR_OUT / "bent_pipe_fixed_mstr"),
         "--epochs", "301", "--plot-every", "20"],
        "c1_bent_pipe_fixed")

  if not skipped("c2"):
    run([PY, "-u", "scripts/reproduce_tomas.py",
         "--config", "notebooks/config_bent_pipe_volume.yaml",
         "--vae-version", args.vae_version,
         "--out", str(PR_OUT / "bent_pipe_volume")],
        "c2_bent_pipe_volume")

  if not skipped("c3"):
    run([PY, "-u", "scripts/reproduce_tomas.py",
         "--config", "notebooks/config_bent_pipe_perim.yaml",
         "--vae-version", args.vae_version,
         "--out", str(PR_OUT / "bent_pipe_perim")],
        "c3_bent_pipe_perim")

  if not skipped("d1"):
    run([PY, "-u", "scripts/reproduce_tomas.py",
         "--config", "notebooks/config_diffuser_a60.yaml",
         "--vae-version", args.vae_version,
         "--out", str(PR_OUT / "diffuser_a60"),
         "--save-every", "20"],
        "d1_diffuser_a60")

  if not skipped("d2"):
    run([PY, "-u", "scripts/run_pareto.py",
         "--base-config", "notebooks/config_diffuser_a60.yaml",
         "--areas", "50", "60", "70", "80",
         "--out", str(PR_OUT / "diffuser_pareto"),
         "--vae-version", args.vae_version,
         "--seeds", "77", "11", "42", "123", "7"],
        "d2_diffuser_pareto")

  if not skipped("e"):
    run([PY, "-u", "scripts/reproduce_tomas.py",
         "--config", "notebooks/config_bifurcated_pipe.yaml",
         "--vae-version", args.vae_version,
         "--out", str(PR_OUT / "bifurcated_pipe")],
        "e_bifurcated_pipe")

  if not skipped("f"):
    run([PY, "-u", "scripts/build_montage.py",
         "--root", str(PR_OUT)],
        "f_montage")


if __name__ == "__main__":
  main()
