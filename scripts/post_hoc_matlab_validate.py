"""TOMAS Tier 2.4: post-hoc MATLAB FE validation.

Workflow per case:
  1. Load final_state.npz (per-cell super-shape params + theta + decoder C00/C11)
  2. Rasterize every cell to 150x150 binary image via shapely
  3. Write images stack to dataset/_posthoc_<case>.mat
  4. Run MATLAB generate_homogenized_data -> get true C00/C11 per cell
  5. Build the FluidSolver, swap in true C00/C11, recompute J via Stokes solve
  6. Append truth_validation block to summary.json (J_true vs J_decoder)
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import scipy.io
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [
    str(ROOT), str(ROOT / "dataset"), str(ROOT / "fluid_TO"),
]

import supershape
import fluid_bcs
import fluid_fe
import fluid_material
import fluid_mesher
import material_constants


MATLAB_EXE = r"D:\Matlab\bin\matlab.exe"
RASTER = 150  # match dataset_4 resolution


def rasterize_cells(shape_params: dict) -> np.ndarray:
  """Build a (num_cells, RASTER, RASTER) array of density images."""
  num = shape_params["shape_a"].size
  ss = supershape.SuperShapes(
      a=shape_params["shape_a"], b=shape_params["shape_b"],
      m=shape_params["shape_m"], n1=shape_params["shape_n1"],
      n2=shape_params["shape_n2"], n3=shape_params["shape_n3"],
      center_x=shape_params["shape_cx"], center_y=shape_params["shape_cy"])
  polygons, _ = supershape.super_shape_to_shapely_polygon(
      ss, prune_out_intersecting_shapes=False)
  density = supershape.project_shapely_polygons_to_density(
      polygons, RASTER, RASTER, True)
  if density.shape[0] != num:
    raise RuntimeError(
        f"Rasterization returned {density.shape[0]} cells, expected {num}.")
  return density


def run_matlab_homogenization(input_mat: Path, output_mat: Path) -> None:
  """Call MATLAB in batch mode on the generate_homogenized_data routine."""
  script = (
      f"cd('{ROOT / 'dataset'}'); "
      f"addpath('{ROOT / 'dataset'}'); "
      f"generate_homogenized_data('{input_mat.as_posix()}', "
      f"'{output_mat.as_posix()}'); exit;"
  )
  print(f"  [MATLAB] running homogenization on "
        f"{input_mat.name} ...", flush=True)
  t0 = time.perf_counter()
  result = subprocess.run([MATLAB_EXE, "-batch", script],
                          cwd=str(ROOT / "dataset"),
                          capture_output=True, text=True)
  elapsed = time.perf_counter() - t0
  print(f"  [MATLAB] finished in {elapsed/60:.1f} min "
        f"(exit {result.returncode})", flush=True)
  if result.returncode != 0:
    print("  STDERR:", result.stderr[-800:], flush=True)
    raise RuntimeError("MATLAB homogenization failed.")


def build_fluid_solver(config):
  bb = fluid_mesher.BoundingBox(
      x_min=config["BOUNDING_BOX"]["x_min"],
      x_max=config["BOUNDING_BOX"]["x_max"],
      y_min=config["BOUNDING_BOX"]["y_min"],
      y_max=config["BOUNDING_BOX"]["y_max"])
  mesh = fluid_mesher.fluid_mesher(
      nelx=config["MESH"]["nelx"], nely=config["MESH"]["nely"],
      bounding_box=bb)
  bc = fluid_bcs.get_dirichlet_bc_and_fixed_dofs(
      mesh, config["BOUNDARY_CONDITIONS"]["char_velocity"],
      fluid_bcs.FluidSampleProblems[
          config["BOUNDARY_CONDITIONS"]["example"]])
  constants = material_constants.MaterialConstants(
      kinematic_viscosity=config["MATERIAL_CONSTANTS"]["kinematic_viscosity"])
  material = fluid_material.FluidMaterial(
      mesh.elem_dx, mesh.elem_dy, constants)
  solver = fluid_fe.FluidSolver(
      mesh, bc, config["BOUNDARY_CONDITIONS"]["fixture_const"])
  return mesh, material, solver


def validate(case_dir: Path) -> dict:
  state_path = case_dir / "final_state.npz"
  if not state_path.exists():
    raise FileNotFoundError(f"Missing {state_path}. Re-run reproduce_tomas "
                            f"with the new save_npz step first.")
  config_path = next(case_dir.glob("config_*.yaml"), None)
  if config_path is not None:
    with open(config_path, "r", encoding="utf-8") as f:
      config = yaml.safe_load(f)
  else:
    rc = case_dir / "run_config.json"
    if not rc.exists():
      raise FileNotFoundError(f"No config_*.yaml or run_config.json in {case_dir}")
    with open(rc, "r", encoding="utf-8") as f:
      config = json.load(f)

  state = np.load(state_path)
  shape_dict = {k: state[k] for k in state.files
                if k.startswith("shape_")}
  theta = state["theta"]
  c00_decoder = state["c00_decoder"]
  c11_decoder = state["c11_decoder"]
  num_cells = shape_dict["shape_a"].size
  print(f"\n=== Post-hoc MATLAB validation: {case_dir.name} ({num_cells} cells) ===",
        flush=True)

  # Step 1-2: rasterize
  t0 = time.perf_counter()
  density = rasterize_cells(shape_dict)
  print(f"  Rasterized {num_cells} cells -> shape {density.shape} "
        f"in {time.perf_counter()-t0:.1f}s")

  # Step 3: write .mat
  tmp_dir = ROOT / "dataset"
  posthoc_in = tmp_dir / f"_posthoc_{case_dir.name}_in.mat"
  posthoc_out = tmp_dir / f"_posthoc_{case_dir.name}_out.mat"
  scipy.io.savemat(posthoc_in, {"mstr_images": density})
  print(f"  Wrote {posthoc_in}")

  # Step 4: MATLAB homog
  run_matlab_homogenization(posthoc_in, posthoc_out)
  homog = scipy.io.loadmat(posthoc_out)
  c00_true = homog["c00"].flatten()
  c11_true = homog["c11"].flatten()
  print(f"  MATLAB: c00 range [{c00_true.min():.4e}, {c00_true.max():.4e}], "
        f"c11 range [{c11_true.min():.4e}, {c11_true.max():.4e}]")

  # Step 5: Python FE recompute with truth
  mesh, material, solver = build_fluid_solver(config)
  c00_t = torch.as_tensor(c00_true, dtype=torch.float64) + 1e-6
  c11_t = torch.as_tensor(c11_true, dtype=torch.float64) + 1e-6
  theta_t = torch.as_tensor(theta, dtype=torch.float64)
  with torch.no_grad():
    J_true, _ = solver.fluid_objective_function(material, c00_t, c11_t, theta_t)
  # Decoder J recompute (sanity check it matches summary)
  c00_d = torch.as_tensor(c00_decoder, dtype=torch.float64)
  c11_d = torch.as_tensor(c11_decoder, dtype=torch.float64)
  with torch.no_grad():
    J_dec_recheck, _ = solver.fluid_objective_function(material, c00_d, c11_d, theta_t)

  # Decoder vs truth distribution of C
  rel_err_c00 = (np.abs(c00_decoder - c00_true)
                 / (np.abs(c00_true) + 1e-12)).mean()
  rel_err_c11 = (np.abs(c11_decoder - c11_true)
                 / (np.abs(c11_true) + 1e-12)).mean()

  # Read existing summary, append truth block
  summary_path = case_dir / "summary.json"
  with open(summary_path, "r", encoding="utf-8") as f:
    summary = json.load(f)
  block = {
      "J_truth_matlab_fe": float(J_true.item()),
      "J_decoder_recheck": float(J_dec_recheck.item()),
      "J_decoder_summary": summary.get("final_fluid_loss_decoder"),
      "C00_decoder_vs_truth_mean_rel_err": float(rel_err_c00),
      "C11_decoder_vs_truth_mean_rel_err": float(rel_err_c11),
      "num_cells": int(num_cells),
  }
  summary["truth_validation"] = block
  with open(summary_path, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)
  print("\n  Truth validation:")
  for k, v in block.items():
    print(f"    {k}: {v}")
  print(f"  Updated {summary_path}")
  return block


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--cases", nargs="+", required=True,
                      help="Paths to case dirs containing final_state.npz")
  args = parser.parse_args()
  results = {}
  for case in args.cases:
    p = Path(case).resolve()
    results[p.name] = validate(p)
  print("\n=== Tier 2.4 summary ===")
  for name, block in results.items():
    print(f"  {name}: J_truth={block['J_truth_matlab_fe']:.3f}  "
          f"J_decoder={block['J_decoder_summary']}  "
          f"C00_relerr={block['C00_decoder_vs_truth_mean_rel_err']:.3f}")


if __name__ == "__main__":
  main()
