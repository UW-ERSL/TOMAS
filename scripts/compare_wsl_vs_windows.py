"""Compare WSL fast-path vs Windows scipy-fallback v9 run, case by case."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WSL = ROOT / "output" / "paper_reproduction"
WIN = ROOT / "output" / "paper_reproduction_v9_windows"


def load(p):
  if not p.exists():
    return None
  with open(p, "r", encoding="utf-8") as f:
    return json.load(f)


def fmt(v):
  if v is None:
    return "n/a"
  if isinstance(v, float):
    return f"{v:.3f}"
  return str(v)


def diff_pct(a, b):
  if a is None or b is None:
    return "n/a"
  if abs(b) < 1e-12:
    return "div0"
  return f"{100.0 * (a - b) / abs(b):+.2f}%"


def case_row(name, wsl_sum, win_sum, j_key, contact_key=None):
  wsl_j = wsl_sum.get(j_key) if wsl_sum else None
  win_j = win_sum.get(j_key) if win_sum else None
  row = {"case": name, "J_wsl": wsl_j, "J_win": win_j, "J_diff": diff_pct(wsl_j, win_j)}
  if contact_key:
    wsl_c = wsl_sum.get(contact_key) if wsl_sum else None
    win_c = win_sum.get(contact_key) if win_sum else None
    row["C_wsl"] = wsl_c; row["C_win"] = win_c
    row["C_diff"] = diff_pct(wsl_c, win_c)
  if wsl_sum and "elapsed_seconds" in wsl_sum:
    row["t_wsl"] = wsl_sum["elapsed_seconds"]
  if win_sum and "elapsed_seconds" in win_sum:
    row["t_win"] = win_sum["elapsed_seconds"]
  return row


cases = [
    ("C1 bent_fixed",     "bent_pipe_fixed_mstr",  "final_dissipated_power",   None),
    ("C2 bent_vol",       "bent_pipe_volume",      "final_fluid_loss_decoder", "final_contact_decoder"),
    ("C3 bent_perim",     "bent_pipe_perim",       "final_fluid_loss_decoder", "final_contact_decoder"),
    ("D1 diffuser_a60",   "diffuser_a60",          "final_fluid_loss_decoder", "final_contact_decoder"),
    ("E  bifurcated",     "bifurcated_pipe",       "final_fluid_loss_decoder", "final_contact_decoder"),
]

print(f"{'Case':<18} {'J_wsl':>10} {'J_win':>10} {'J_diff':>10} "
      f"{'C_wsl':>10} {'C_win':>10} {'C_diff':>10} {'t_wsl':>8} {'t_win':>8}")
print("-" * 110)
for label, dirname, j_key, c_key in cases:
  wsl_sum = load(WSL / dirname / "summary.json")
  win_sum = load(WIN / dirname / "summary.json")
  r = case_row(label, wsl_sum, win_sum, j_key, c_key)
  print(f"{r['case']:<18} {fmt(r['J_wsl']):>10} {fmt(r['J_win']):>10} {r['J_diff']:>10} "
        f"{fmt(r.get('C_wsl')):>10} {fmt(r.get('C_win')):>10} {r.get('C_diff','n/a'):>10} "
        f"{fmt(r.get('t_wsl')):>8} {fmt(r.get('t_win')):>8}")

# Pareto comparison
print(f"\nPareto best (multi-seed):")
import csv
def load_pareto(p):
  if not p.exists(): return None
  with open(p, "r", encoding="utf-8") as f:
    return list(csv.DictReader(f))

wsl_par = load_pareto(WSL / "diffuser_pareto" / "pareto.csv")
win_par = load_pareto(WIN / "diffuser_pareto" / "pareto.csv")
if wsl_par and win_par:
  print(f"{'area':>6} {'J_wsl':>10} {'J_win':>10} {'J_diff':>10} "
        f"{'C_wsl':>10} {'C_win':>10}")
  for wr, vr in zip(wsl_par, win_par):
    wj = float(wr['decoder_dissipated_power']); vj = float(vr['decoder_dissipated_power'])
    wc = float(wr['decoder_contact_area']);     vc = float(vr['decoder_contact_area'])
    print(f"{wr['area_target']:>6} {wj:>10.3f} {vj:>10.3f} {diff_pct(wj, vj):>10} "
          f"{wc:>10.3f} {vc:>10.3f}")
else:
  print("  (pareto.csv missing on one side)")
