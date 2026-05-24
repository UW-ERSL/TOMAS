"""Generate output/paper_reproduction/README.md by collecting summary.json files
from every paper case. Run AFTER Phase C-E completes.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PR_OUT = ROOT / "output" / "paper_reproduction"


def load(path):
  if not path.exists():
    return None
  with open(path, "r", encoding="utf-8") as f:
    return json.load(f)


def fmt(v, fmt_spec=".3f"):
  if v is None:
    return "n/a"
  if isinstance(v, float):
    return format(v, fmt_spec)
  return str(v)


def main():
  cases = {
      "ideal": load(PR_OUT / "ideal_microstructure" / "selected.json"),
      "c1": load(PR_OUT / "bent_pipe_fixed_mstr" / "summary.json"),
      "c2": load(PR_OUT / "bent_pipe_volume" / "summary.json"),
      "c3": load(PR_OUT / "bent_pipe_perim" / "summary.json"),
      "d1": load(PR_OUT / "diffuser_a60" / "summary.json"),
      "e":  load(PR_OUT / "bifurcated_pipe" / "summary.json"),
  }

  lines = []
  L = lines.append
  L("# TOMAS Paper Reproduction\n")
  L("Source paper: Padhy, Suresh, Chandrasekhar. "
    "*TOMAS: topology optimization of multiscale fluid flow devices using "
    "variational auto-encoders and super-shapes.* "
    "Structural and Multidisciplinary Optimization 67:119 (2024).\n")
  L("This reproduction follows the paper's offline-then-online recipe:\n")
  L("1. Sample 7000 super-shapes within the parameter ranges of paper §2.3.")
  L("2. Run MATLAB-based numerical homogenization on every cell.")
  L("3. Train a 2D-latent VAE (Eq. 6) on the 12-feature dataset.")
  L("4. Use the trained decoder + a coordinate NN to optimise each "
    "fluid-flow case.\n")
  L("All re-runs live under `output/paper_reproduction/`. The legacy "
    "100-sample reproduction is preserved untouched at `output/reproduction/`.\n")

  L("## Setup\n")
  L("| Quantity | Value |\n| --- | --- |")
  L("| Python | 3.10 (.venv) |")
  L("| PyTorch | 2.1.2 (CPU) |")
  L("| MATLAB | D:\\Matlab\\bin\\matlab.exe |")
  L("| Super-shape range | 0.05 ≤ a,b ≤ 0.75; 1 ≤ m ≤ 22; "
    "0.5 ≤ n1,n2,n3 ≤ 10 |")
  L("| Sample count | 7000 (after Shapely pruning) |")
  L("| VAE | latent=2, encoder/decoder=600x600, 17000 epochs, "
    "lr=8e-3, kl=1e-7 |\n")

  L("## §3.1 Ideal microstructure selection (Fig. 10)\n")
  if cases["ideal"]:
    s = cases["ideal"]
    L(f"- Selected latent point: z = ({s['latent'][0]:+.4f}, "
      f"{s['latent'][1]:+.4f})")
    L(f"- C00, C11, trace(C) : {s['C00']:.4f}, {s['C11']:.4f}, "
      f"{s['trace_C']:.4f}")
    L(f"- Volume fraction    : {s['vol_frac']:.4f} (target 0.25 ± 0.001)")
    sp = s["shape_params"]
    pp = s["paper_reported_M"]
    L("\n| Parameter | Decoder selection | Paper Fig.10 value |")
    L("| --- | --- | --- |")
    for k in ["a", "b", "m", "n1", "n2", "n3"]:
      L(f"| {k} | {sp[k]:.4f} | {pp[k]:.4f} |")
    L("\n_(Exact values depend on VAE training randomness; the paper Fig.10 "
      "fish shape and our selection should both lie on the high-trace(C) "
      "ridge inside the vf~0.25 band.)_\n")
    L("Figures: `ideal_microstructure/latent_scatter.png`, "
      "`ideal_microstructure/shape_preview.png`.\n")
  else:
    L("(Not yet run.)\n")

  L("## §3.2 Bent pipe with fixed microstructure (Fig. 11d)\n")
  if cases["c1"]:
    s = cases["c1"]
    L(f"- Final dissipated power : {s['final_dissipated_power']:.4f}")
    L(f"- Paper reported         : 15.1")
    if "truth_validation" in s:
      t = s["truth_validation"]
      L(f"- **MATLAB FE truth J**  : **{t['J_truth_matlab_fe']:.4f}** "
        f"(C00 rel_err vs decoder: {t['C00_decoder_vs_truth_mean_rel_err']:.3f})")
    L("")
  else:
    L("(Not yet run.)\n")

  L("## §3.3 Bent pipe, full design space (Fig. 12)\n")
  if cases["c2"]:
    s = cases["c2"]
    L(f"### (a) Volume constraint v_f ≤ 0.75")
    L(f"- Decoder dissipated power : {fmt(s['final_fluid_loss_decoder'])}")
    L(f"- Decoder contact area     : {fmt(s['final_contact_decoder'])}")
    L(f"- Shapely post-hoc area    : {fmt(s.get('shapely_contact_posthoc'))}")
    if "truth_validation" in s:
      t = s["truth_validation"]
      L(f"- **MATLAB FE truth J**    : **{t['J_truth_matlab_fe']:.4f}** "
        f"(C00 rel_err: {t['C00_decoder_vs_truth_mean_rel_err']:.3f})")
    L(f"- Paper reported obj       : 9.61\n")
  if cases["c3"]:
    s = cases["c3"]
    L(f"### (b) Contact-area constraint Γ ≥ 75.69")
    L(f"- Decoder dissipated power : {fmt(s['final_fluid_loss_decoder'])}")
    L(f"- Decoder contact area     : {fmt(s['final_contact_decoder'])}")
    L(f"- Shapely post-hoc area    : {fmt(s.get('shapely_contact_posthoc'))}")
    if "truth_validation" in s:
      t = s["truth_validation"]
      L(f"- **MATLAB FE truth J**    : **{t['J_truth_matlab_fe']:.4f}** "
        f"(C00 rel_err: {t['C00_decoder_vs_truth_mean_rel_err']:.3f})")
    L(f"- Paper reported (decoder) : 7.56")
    L(f"- Paper reported (validated): 7.87 / contact 78.49\n")

  L("## §3.4 Diffuser convergence (Fig. 13)\n")
  if cases["d1"]:
    s = cases["d1"]
    if "truth_validation" in s:
      t = s["truth_validation"]
      L(f"- **MATLAB FE truth J**    : **{t['J_truth_matlab_fe']:.4f}** "
        f"(C00 rel_err: {t['C00_decoder_vs_truth_mean_rel_err']:.3f})")
    L(f"- Decoder dissipated power : {fmt(s['final_fluid_loss_decoder'])}")
    L(f"- Decoder contact area     : {fmt(s['final_contact_decoder'])}")
    L(f"- Shapely post-hoc area    : {fmt(s.get('shapely_contact_posthoc'))}")
    L(f"Snapshots in `diffuser_a60/topologies/` at epochs 0, 20, ..., 300.\n")
  else:
    L("(Not yet run.)\n")

  L("## §3.5 Pareto front (Fig. 14)\n")
  pareto_csv = PR_OUT / "diffuser_pareto" / "pareto.csv"
  if pareto_csv.exists():
    L("| Area target | Decoder J | Decoder contact | Elapsed (s) |")
    L("| ---: | ---: | ---: | ---: |")
    import csv
    with open(pareto_csv, "r", encoding="utf-8") as f:
      for row in csv.DictReader(f):
        elapsed_key = ("elapsed_seconds_sum" if "elapsed_seconds_sum" in row
                       else "elapsed_seconds")
        L(f"| {row['area_target']} | "
          f"{float(row['decoder_dissipated_power']):.3f} | "
          f"{float(row['decoder_contact_area']):.3f} | "
          f"{float(row[elapsed_key]):.1f} |")
    L("\n![Pareto](diffuser_pareto/pareto.png)\n")
  else:
    L("(Not yet run.)\n")

  L("## §3.7 Bifurcated pipe (Fig. 16)\n")
  if cases["e"]:
    s = cases["e"]
    L(f"- Decoder dissipated power : {fmt(s['final_fluid_loss_decoder'])}")
    L(f"- Decoder contact area     : {fmt(s['final_contact_decoder'])}")
    L(f"- Shapely post-hoc area    : {fmt(s.get('shapely_contact_posthoc'))}")
    if "truth_validation" in s:
      t = s["truth_validation"]
      L(f"- **MATLAB FE truth J**    : **{t['J_truth_matlab_fe']:.4f}** "
        f"(C00 rel_err: {t['C00_decoder_vs_truth_mean_rel_err']:.3f})")
    L("")
  else:
    L("(Not yet run.)\n")

  L("## Overview montage\n")
  L("`figures_montage.png` collects one panel per paper figure.\n")

  L("## Reproduction commands\n")
  L("```powershell")
  L("cd \"C:\\Users\\Bingxiao Du\\Documents\\TOMAS\"")
  L("# Phase A (one-time, ~4-5h):")
  L("python -u scripts\\run_offline_dataset.py")
  L("& \"D:\\Matlab\\bin\\matlab.exe\" -batch "
    "\"cd('dataset'); run('run_homogenization_v2.m'); exit;\"")
  L("python -u scripts\\train_vae_v2.py")
  L("")
  L("# Phase B-F:")
  L("python -u scripts\\select_ideal_microstructure.py --vae-version v2 "
    "--out output\\paper_reproduction\\ideal_microstructure")
  L("python -u scripts\\run_all_paper.py --vae-version v2")
  L("python -u scripts\\write_final_readme.py")
  L("```")

  out_path = PR_OUT / "README.md"
  out_path.parent.mkdir(parents=True, exist_ok=True)
  out_path.write_text("\n".join(lines), encoding="utf-8")
  print(f"Wrote {out_path}")


if __name__ == "__main__":
  main()
