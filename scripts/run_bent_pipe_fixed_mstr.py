"""TOMAS paper §3.2: bent pipe filled by ONE fixed fish-like microstructure.
Only the orientation theta is optimized; latent coordinates are not part of
the design space.

The microstructure (and its homogenized C00, C11) is supplied via the JSON
produced by `scripts/select_ideal_microstructure.py`.
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [
    str(ROOT),
    str(ROOT / "dataset"),
    str(ROOT / "fluid_TO"),
    str(ROOT / "vae"),
]

import dataset.supershape as supershape
import fluid_bcs
import fluid_fe
import fluid_material
import fluid_mesher
import material_constants
import neural_network
import projection
import utils


def load_config(path):
  with open(path, "r", encoding="utf-8") as f:
    return yaml.safe_load(f)


def build_problem(config):
  bb = fluid_mesher.BoundingBox(
      x_min=config["BOUNDING_BOX"]["x_min"],
      x_max=config["BOUNDING_BOX"]["x_max"],
      y_min=config["BOUNDING_BOX"]["y_min"],
      y_max=config["BOUNDING_BOX"]["y_max"],
  )
  mesh = fluid_mesher.fluid_mesher(
      nelx=config["MESH"]["nelx"], nely=config["MESH"]["nely"], bounding_box=bb)
  bc = fluid_bcs.get_dirichlet_bc_and_fixed_dofs(
      mesh, config["BOUNDARY_CONDITIONS"]["char_velocity"],
      fluid_bcs.FluidSampleProblems[config["BOUNDARY_CONDITIONS"]["example"]])
  constants = material_constants.MaterialConstants(
      kinematic_viscosity=config["MATERIAL_CONSTANTS"]["kinematic_viscosity"])
  material = fluid_material.FluidMaterial(mesh.elem_dx, mesh.elem_dy, constants)
  solver = fluid_fe.FluidSolver(mesh, bc, config["BOUNDARY_CONDITIONS"]
                                              ["fixture_const"])

  fourier = projection.FourierMap(
      mesh,
      projection.FourierActivation[config["FOURIER_MAP_PARAMS"]
                                   ["fourier_map_activation"]],
      num_fourier_terms=config["FOURIER_MAP_PARAMS"]["num_fourier_terms"],
      max_radius=config["FOURIER_MAP_PARAMS"]["max_radius"],
      min_radius=config["FOURIER_MAP_PARAMS"]["min_radius"])
  nn_params = neural_network.NeuralNetworkParameters(
      input_dim=2 * config["FOURIER_MAP_PARAMS"]["num_fourier_terms"],
      output_dim=1,
      num_layers=config["NEURAL_NETWORK_PARAMS"]["num_layers"],
      num_neurons_per_layer=config["NEURAL_NETWORK_PARAMS"]
                                  ["num_neurons_per_layer"])
  net = neural_network.TopOptNet(nn_params)
  return mesh, material, solver, fourier, net


def projected_coords(mesh, fourier, config):
  sym_params = projection.SymParams(
      sym_x_axis_mid_pt=0.5 * config["BOUNDING_BOX"]["y_max"],
      sym_y_axis_mid_pt=0.5 * config["BOUNDING_BOX"]["x_max"],
  )
  xy = torch.tensor(mesh.elem_centers, dtype=torch.float64)
  xy_r, signs = projection.apply_reflection(
      xy,
      projection.SymmetryActivation[config["FOURIER_MAP_PARAMS"]
                                    ["symmetry_activation_y_axis"]],
      projection.SymmetryActivation[config["FOURIER_MAP_PARAMS"]
                                    ["symmetry_activation_x_axis"]],
      sym_params)
  return fourier.apply_fourier_map(xy_r), signs


def save_topology(mstr_params, theta, nelx, nely, path, title):
  x, y = supershape.get_euclidean_coords_of_points_on_surf_super_shape(
      mstr_params, theta)
  fig, ax = plt.subplots(figsize=(min(16, max(5, nelx * 0.22)),
                                  min(16, max(5, nely * 0.22))))
  ax.patch.set_facecolor("#DAE8FC")
  ctr = 0
  for rw in range(nelx):
    dx = 2 * rw + 1.0
    bb_x_min, bb_x_max = 2 * rw, 2 * rw + 2.0
    for col in range(nely):
      dy = 2 * col + 1.0
      bb_y_min, bb_y_max = 2 * col, 2 * col + 2.0
      sx = np.clip(x[ctr, :] + dx, a_min=bb_x_min, a_max=bb_x_max)
      sy = np.clip(y[ctr, :] + dy, a_min=bb_y_min, a_max=bb_y_max)
      ax.fill(sx, sy, facecolor="#F8CECC", edgecolor="black", linewidth=0.15)
      ctr += 1
  ax.plot([0, 2 * nelx, 2 * nelx, 0, 0],
          [0, 0, 2 * nely, 2 * nely, 0], "k", linewidth=0.8)
  ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
  ax.set_title(title)
  fig.tight_layout()
  fig.savefig(path, dpi=240)
  plt.close(fig)


def save_field(field, nelx, nely, path, title, cmap="turbo"):
  fig, ax = plt.subplots(figsize=(5, 5))
  im = ax.imshow(field.reshape((nelx, nely)).T,
                 interpolation="none", origin="lower", cmap=cmap)
  fig.colorbar(im, ax=ax)
  ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
  ax.set_title(title)
  fig.tight_layout()
  fig.savefig(path, dpi=240)
  plt.close(fig)


def save_history(history, out_dir):
  with open(out_dir / "history.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["epoch", "fluid_loss"])
    for i, e in enumerate(history["epoch"]):
      w.writerow([e, history["fluid_loss"][i]])
  fig, ax = plt.subplots(figsize=(7, 4))
  ax.plot(history["epoch"], history["fluid_loss"], color="tab:red")
  ax.set_xlabel("epoch"); ax.set_ylabel("dissipated power")
  ax.grid(True, alpha=0.25)
  fig.tight_layout()
  fig.savefig(out_dir / "history.png", dpi=240)
  plt.close(fig)


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--config", default=str(ROOT / "notebooks"
                                              / "config_bent_pipe.yaml"))
  parser.add_argument("--selected-mstr", required=True,
                      help="Path to selected.json from "
                           "select_ideal_microstructure.py")
  parser.add_argument("--out", required=True)
  parser.add_argument("--epochs", type=int, default=301)
  parser.add_argument("--plot-every", type=int, default=20)
  args = parser.parse_args()

  out_dir = Path(args.out)
  out_dir.mkdir(parents=True, exist_ok=True)
  (out_dir / "topologies").mkdir(exist_ok=True)

  with open(args.selected_mstr, "r", encoding="utf-8") as f:
    selected = json.load(f)
  sp_ = selected["shape_params"]
  c00_scalar = selected["C00"]
  c11_scalar = selected["C11"]

  config = load_config(args.config)
  config["OPTIMIZATION"]["num_epochs"] = args.epochs
  config["OPTIMIZATION"]["plot_interval"] = args.plot_every
  with open(out_dir / "run_config.json", "w", encoding="utf-8") as f:
    json.dump(config, f, indent=2)
  with open(out_dir / "selected_used.json", "w", encoding="utf-8") as f:
    json.dump(selected, f, indent=2)

  mesh, material, solver, fourier, net = build_problem(config)
  xy_f, signs = projected_coords(mesh, fourier, config)

  num_elems = mesh.nelx * mesh.nely
  c00 = torch.full((num_elems,), c00_scalar, dtype=torch.float64) + 1e-6
  c11 = torch.full((num_elems,), c11_scalar, dtype=torch.float64) + 1e-6

  optimizer = torch.optim.Adam(net.parameters(),
                               lr=config["OPTIMIZATION"]["learning_rate"])

  # Broadcast a single supershape across all cells for topology plotting.
  mstr_for_plot = supershape.SuperShapes(
      a=np.full(num_elems, sp_["a"]), b=np.full(num_elems, sp_["b"]),
      m=np.full(num_elems, sp_["m"]), n1=np.full(num_elems, sp_["n1"]),
      n2=np.full(num_elems, sp_["n2"]), n3=np.full(num_elems, sp_["n3"]),
      center_x=np.full(num_elems, sp_["cx"]),
      center_y=np.full(num_elems, sp_["cy"]))

  history = {"epoch": [], "fluid_loss": []}
  start = time.perf_counter()
  final_state = None

  for epoch in range(args.epochs):
    optimizer.zero_grad()
    _, theta = net(xy_f)
    theta = torch.einsum("i,i->i", theta, signs["X"])
    theta = torch.einsum("i,i->i", theta, signs["Y"])
    fluid_loss, vp = solver.fluid_objective_function(material, c00, c11, theta)
    fluid_loss.backward()
    if config["OPTIMIZATION"]["grad_clip_activation"] == "GRAD_CLIP_ON":
      torch.nn.utils.clip_grad_norm_(net.parameters(),
                                     config["OPTIMIZATION"]["grad_clip_norm"])
    optimizer.step()

    with torch.no_grad():
      _, theta_eval = net(xy_f)
      theta_eval = torch.einsum("i,i->i", theta_eval, signs["X"])
      theta_eval = torch.einsum("i,i->i", theta_eval, signs["Y"])
      eval_loss, vp_eval = solver.fluid_objective_function(
          material, c00, c11, theta_eval)

    history["epoch"].append(epoch)
    history["fluid_loss"].append(eval_loss.item())
    final_state = (theta_eval.detach().numpy(), vp_eval.detach())
    print(f"Iter {epoch:3d}  J: {eval_loss.item():.4E}", flush=True)

    if epoch % args.plot_every == 0 or epoch == args.epochs - 1:
      save_topology(mstr_for_plot, theta_eval.detach().numpy(),
                    mesh.nelx, mesh.nely,
                    out_dir / "topologies" / f"epoch_{epoch:04d}.png",
                    f"BENT_PIPE fixed-mstr epoch {epoch}")
      save_history(history, out_dir)

  save_history(history, out_dir)
  theta_np, vp = final_state
  u, v, p = solver.get_element_velocity_pressure(vp)
  vm = torch.sqrt(u ** 2 + v ** 2).detach().numpy()
  # Save final state for post-hoc MATLAB validation (Tier 2.4).
  c00_arr = np.full(num_elems, c00_scalar, dtype=np.float64)
  c11_arr = np.full(num_elems, c11_scalar, dtype=np.float64)
  np.savez(out_dir / "final_state.npz",
           shape_a=mstr_for_plot.a, shape_b=mstr_for_plot.b,
           shape_m=mstr_for_plot.m, shape_n1=mstr_for_plot.n1,
           shape_n2=mstr_for_plot.n2, shape_n3=mstr_for_plot.n3,
           shape_cx=mstr_for_plot.center_x, shape_cy=mstr_for_plot.center_y,
           theta=theta_np,
           c00_decoder=c00_arr, c11_decoder=c11_arr,
           nelx=np.array([mesh.nelx]), nely=np.array([mesh.nely]))
  save_topology(mstr_for_plot, theta_np, mesh.nelx, mesh.nely,
                out_dir / "final_topology.png",
                "BENT_PIPE fixed-mstr final")
  save_field(vm, mesh.nelx, mesh.nely, out_dir / "velocity_magnitude.png",
             "velocity magnitude", "turbo")
  save_field(p.detach().numpy(), mesh.nelx, mesh.nely,
             out_dir / "pressure.png", "pressure", "viridis")

  summary = {
      "config": str(args.config),
      "selected_mstr": str(args.selected_mstr),
      "example": "BENT_PIPE",
      "case": "fixed_microstructure_only_theta",
      "epochs": args.epochs,
      "elapsed_seconds": time.perf_counter() - start,
      "final_dissipated_power": history["fluid_loss"][-1],
      "paper_reported": 15.1,
  }
  with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)
  print(json.dumps(summary, indent=2))


if __name__ == "__main__":
  main()
