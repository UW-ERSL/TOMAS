"""TOMAS paper §3.5: scan the diffuser problem across multiple contact-area
targets and report the Pareto curve (dissipated power vs. contact area).

With --seeds N, runs N seeds per area target and picks the best (lowest J among
solutions that satisfy the constraint within tolerance).
"""
import argparse
import csv
import json
import sys
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

import reproduce_tomas


DEFAULT_SEEDS = [77, 11, 42, 123, 7]


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--base-config",
                      default=str(ROOT / "notebooks"
                                  / "config_diffuser_a60.yaml"))
  parser.add_argument("--areas", type=float, nargs="+",
                      default=[50.0, 60.0, 70.0, 80.0])
  parser.add_argument("--out",
                      default=str(ROOT / "output" / "paper_reproduction"
                                  / "diffuser_pareto"))
  parser.add_argument("--vae-version", default="v6")
  parser.add_argument("--epochs", type=int)
  parser.add_argument("--seeds", type=int, nargs="+", default=[77],
                      help="One or more NN seeds. If multiple, picks the "
                           "best per area.")
  parser.add_argument("--constraint-tol", type=float, default=0.05,
                      help="When choosing best seed, require |final_constraint| "
                           "<= tol (with constraint expressed as 1 - sum/target "
                           "or sum/target - 1).")
  args = parser.parse_args()

  out_root = Path(args.out)
  out_root.mkdir(parents=True, exist_ok=True)

  with open(args.base_config, "r", encoding="utf-8") as f:
    base = yaml.safe_load(f)

  rows = []
  all_runs = []  # full per-seed log
  tmp_root = Path(tempfile.mkdtemp(prefix="pareto_cfg_"))
  for area in args.areas:
    area_dir = out_root / f"area_{int(area)}"
    area_dir.mkdir(parents=True, exist_ok=True)
    config = yaml.safe_load(yaml.safe_dump(base))  # deep copy
    config["OPTIMIZATION"]["desired_perimeter"] = float(area)

    tmp_cfg = tmp_root / f"config_diffuser_a{int(area)}.yaml"
    with open(tmp_cfg, "w", encoding="utf-8") as f:
      yaml.safe_dump(config, f, sort_keys=False)

    seed_summaries = []
    for seed in args.seeds:
      if len(args.seeds) == 1:
        sub_dir = area_dir
      else:
        sub_dir = area_dir / f"seed_{seed}"
      sub_dir.mkdir(parents=True, exist_ok=True)

      print(f"\n=== Pareto: area={area} seed={seed} ===", flush=True)
      reproduce_tomas.run_case(
          config_path=tmp_cfg,
          output_dir=sub_dir,
          epochs_override=args.epochs,
          save_every=None,
          vae_version=args.vae_version,
          nn_seed=seed,
      )
      with open(sub_dir / "summary.json", "r", encoding="utf-8") as f:
        summary = json.load(f)
      summary["seed"] = seed
      seed_summaries.append(summary)
      all_runs.append({
          "area_target": area, "seed": seed,
          "J": summary["final_fluid_loss_decoder"],
          "contact": summary["final_contact_decoder"],
          "shapely": summary.get("shapely_contact_posthoc"),
          "constraint": summary["final_constraint"],
      })

    # Pick best seed: min J among those satisfying |constraint| <= tol
    feasible = [s for s in seed_summaries
                if abs(s["final_constraint"]) <= args.constraint_tol]
    if feasible:
      best = min(feasible, key=lambda s: s["final_fluid_loss_decoder"])
      basis = "feasible"
    else:
      # No seed satisfies tol — fall back to min (J + 100*|constraint|)
      best = min(seed_summaries,
                 key=lambda s: (s["final_fluid_loss_decoder"]
                                + 100 * abs(s["final_constraint"])))
      basis = "fallback (no feasible)"
    print(f"  -> best seed {best['seed']} ({basis}): J={best['final_fluid_loss_decoder']:.3f}, "
          f"contact={best['final_contact_decoder']:.2f}", flush=True)

    rows.append({
        "area_target": area,
        "best_seed": best["seed"],
        "selection_basis": basis,
        "decoder_dissipated_power": best["final_fluid_loss_decoder"],
        "decoder_contact_area": best["final_contact_decoder"],
        "shapely_contact_posthoc": best.get("shapely_contact_posthoc"),
        "final_constraint": best["final_constraint"],
        "elapsed_seconds_sum": sum(s["elapsed_seconds"]
                                   for s in seed_summaries),
    })

    # If multi-seed, also link best seed dir to area_dir/best/
    if len(args.seeds) > 1:
      with open(area_dir / "best.json", "w", encoding="utf-8") as f:
        json.dump({"seed": best["seed"], "basis": basis,
                   "summary": best}, f, indent=2)

  # Pareto.csv (best per area)
  csv_path = out_root / "pareto.csv"
  with open(csv_path, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
  print(f"\nWrote {csv_path}")

  # All-seeds detailed log
  all_csv = out_root / "pareto_all_seeds.csv"
  with open(all_csv, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(all_runs[0].keys()))
    w.writeheader(); w.writerows(all_runs)
  print(f"Wrote {all_csv}")

  # Pareto plot (best curve + all seed points)
  fig, ax = plt.subplots(figsize=(7, 5))
  if len(args.seeds) > 1:
    all_xs = [r["contact"] for r in all_runs]
    all_ys = [r["J"] for r in all_runs]
    ax.scatter(all_xs, all_ys, c="lightgray", s=30,
               label=f"all seeds (n={len(args.seeds)})")
  xs = [r["decoder_contact_area"] for r in rows]
  ys = [r["decoder_dissipated_power"] for r in rows]
  ax.plot(xs, ys, "o-", color="tab:blue", markersize=10, label="best per area")
  for r in rows:
    ax.annotate(f'tgt={r["area_target"]:.0f}',
                (r["decoder_contact_area"], r["decoder_dissipated_power"]),
                textcoords="offset points", xytext=(6, 6), fontsize=8)
  ax.set_xlabel("Contact area")
  ax.set_ylabel("Dissipated power")
  ax.set_title(f"Diffuser Pareto (TOMAS §3.5, {len(args.seeds)} seed{'s' if len(args.seeds) > 1 else ''})")
  ax.legend()
  ax.grid(True, alpha=0.3)
  fig.tight_layout()
  fig.savefig(out_root / "pareto.png", dpi=240)
  plt.close(fig)
  print(f"Wrote {out_root / 'pareto.png'}")


if __name__ == "__main__":
  main()
