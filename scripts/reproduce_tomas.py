import argparse
import csv
import json
import shutil
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
import loss
import material_constants
import neural_network
import opt_constraints
import projection
import utils
import vae.data_preprocess as vae_data_prep
import vae.network as vae_network


def _enum(enum_cls, value):
  return enum_cls[value]


def load_yaml(path):
  with open(path, "r", encoding="utf-8") as handle:
    return yaml.safe_load(handle)


def build_problem(config, nn_seed: int = 77):
  bounding_box = fluid_mesher.BoundingBox(
      x_min=config["BOUNDING_BOX"]["x_min"],
      x_max=config["BOUNDING_BOX"]["x_max"],
      y_min=config["BOUNDING_BOX"]["y_min"],
      y_max=config["BOUNDING_BOX"]["y_max"],
  )
  mesh = fluid_mesher.fluid_mesher(
      nelx=config["MESH"]["nelx"],
      nely=config["MESH"]["nely"],
      bounding_box=bounding_box,
  )
  bc = fluid_bcs.get_dirichlet_bc_and_fixed_dofs(
      mesh,
      config["BOUNDARY_CONDITIONS"]["char_velocity"],
      fluid_bcs.FluidSampleProblems[config["BOUNDARY_CONDITIONS"]["example"]],
  )
  constants = material_constants.MaterialConstants(
      kinematic_viscosity=config["MATERIAL_CONSTANTS"]["kinematic_viscosity"])
  material = fluid_material.FluidMaterial(mesh.elem_dx, mesh.elem_dy, constants)
  solver = fluid_fe.FluidSolver(
      mesh, bc, config["BOUNDARY_CONDITIONS"]["fixture_const"])

  fourier = projection.FourierMap(
      mesh,
      projection.FourierActivation[
          config["FOURIER_MAP_PARAMS"]["fourier_map_activation"]],
      num_fourier_terms=config["FOURIER_MAP_PARAMS"]["num_fourier_terms"],
      max_radius=config["FOURIER_MAP_PARAMS"]["max_radius"],
      min_radius=config["FOURIER_MAP_PARAMS"]["min_radius"],
  )
  nn_settings = neural_network.NeuralNetworkParameters(
      input_dim=2 * config["FOURIER_MAP_PARAMS"]["num_fourier_terms"],
      output_dim=config["NEURAL_NETWORK_PARAMS"]["output_dim"],
      num_layers=config["NEURAL_NETWORK_PARAMS"]["num_layers"],
      num_neurons_per_layer=config["NEURAL_NETWORK_PARAMS"]
      ["num_neurons_per_layer"],
  )
  topopt_net = neural_network.TopOptNet(nn_settings, seed=nn_seed)
  return mesh, material, solver, fourier, topopt_net


VAE_FILES = {
    "v1": ("vae_net.pt", "nomalization.pt"),
    "v2": ("vae_net_v2.pt", "normalization_v2.pt"),
    "v3": ("vae_net_v3.pt", "normalization_v3.pt"),
    "v4": ("vae_net_v4.pt", "normalization_v4.pt"),
    "v5": ("vae_net_v5.pt", "normalization_v5.pt"),
    "v6": ("vae_net_v6.pt", "normalization_v6.pt"),
}

# Hidden-dim override per VAE version (v6 uses 1200, others use vae_config.yaml).
VAE_HIDDEN_OVERRIDE = {"v6": 1200}


def load_vae(version: str = "v1"):
  vae_config = load_yaml(ROOT / "notebooks" / "vae_config.yaml")
  vae_yaml = vae_config["NETWORK"]
  hidden = VAE_HIDDEN_OVERRIDE.get(version, vae_yaml["encoder_hidden_dim"])
  params = vae_network.VAE_Params(
      input_dim=12,
      encoder_hidden_dim=hidden,
      latent_dim=vae_yaml["latent_dim"],
      decoder_hidden_dim=hidden,
  )
  model = vae_network.VariationalAutoencoder(params)
  model.encoder.is_training = False
  net_name, norm_name = VAE_FILES[version]
  net_path = ROOT / "vae" / net_name
  norm_path = ROOT / "vae" / norm_name
  if not net_path.exists() or not norm_path.exists():
    raise FileNotFoundError(
        f"VAE {version} not found. Expected {net_path} and {norm_path}.")
  model.load_state_dict(torch.load(net_path))
  model.eval()
  normalization = torch.load(norm_path)
  normalization_types = (
      [vae_data_prep.NomalizationType.LINEAR] * 8
      + [vae_data_prep.NomalizationType.LOG] * 2
      + [vae_data_prep.NomalizationType.LINEAR] * 2
  )
  return model, normalization["max_feature"], normalization["min_feature"], normalization_types


def projected_coordinates(mesh, fourier, config, sym_params, resolution):
  if resolution == 1:
    xy = torch.tensor(mesh.elem_centers, dtype=torch.float64)
  else:
    elem_centers = utils.generate_points_in_domain(
        mesh.nelx, mesh.nely, mesh.elem_dx, mesh.elem_dy, mesh.num_dim,
        resolution)
    xy = torch.tensor(elem_centers, dtype=torch.float64)

  xy_reflected, reflection_signs = projection.apply_reflection(
      xy,
      projection.SymmetryActivation[
          config["FOURIER_MAP_PARAMS"]["symmetry_activation_y_axis"]],
      projection.SymmetryActivation[
          config["FOURIER_MAP_PARAMS"]["symmetry_activation_x_axis"]],
      sym_params,
  )
  return fourier.apply_fourier_map(xy_reflected), reflection_signs


def save_microstructure_plot(mstr_params, theta, nelx, nely, path, title):
  x, y = supershape.get_euclidean_coords_of_points_on_surf_super_shape(
      mstr_params, theta)
  fig_width = min(16, max(5, nelx * 0.22))
  fig_height = min(16, max(5, nely * 0.22))
  fig, ax = plt.subplots(figsize=(fig_width, fig_height))
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
  ax.set_aspect("equal")
  ax.set_title(title)
  ax.set_xticks([])
  ax.set_yticks([])
  fig.tight_layout()
  fig.savefig(path, dpi=240)
  plt.close(fig)


def save_field(field, nelx, nely, path, title, color_map="viridis"):
  fig, ax = plt.subplots(figsize=(5, 5))
  image = ax.imshow(field.reshape((nelx, nely)).T,
                    interpolation="none",
                    origin="lower",
                    cmap=color_map)
  fig.colorbar(image, ax=ax)
  ax.set_title(title)
  ax.set_aspect("equal")
  ax.set_xticks([])
  ax.set_yticks([])
  fig.tight_layout()
  fig.savefig(path, dpi=240)
  plt.close(fig)


def save_history_plot(history, path):
  epochs = np.array(history["epoch"])
  fig, ax1 = plt.subplots(figsize=(7, 4))
  ax1.plot(epochs, history["contact"], color="tab:blue", label="contact area")
  ax1.set_xlabel("epoch")
  ax1.set_ylabel("contact area", color="tab:blue")
  ax1.tick_params(axis="y", labelcolor="tab:blue")
  ax2 = ax1.twinx()
  ax2.plot(epochs, history["fluid_loss"], color="tab:red",
           label="dissipated power")
  ax2.set_ylabel("dissipated power", color="tab:red")
  ax2.tick_params(axis="y", labelcolor="tab:red")
  ax1.grid(True, alpha=0.25)
  fig.tight_layout()
  fig.savefig(path, dpi=240)
  plt.close(fig)


def write_history(history, path):
  with open(path, "w", newline="", encoding="utf-8") as handle:
    writer = csv.writer(handle)
    writer.writerow(["epoch", "fluid_loss", "contact", "constraint", "net_loss"])
    for idx, epoch in enumerate(history["epoch"]):
      writer.writerow([
          epoch,
          history["fluid_loss"][idx],
          history["contact"][idx],
          history["constraint"][idx],
          history["net_loss"][idx],
      ])


def run_case(config_path, output_dir, epochs_override=None, save_every=None,
             vae_version: str = "v1", nn_seed: int = 77):
  config_path = Path(config_path).resolve()
  output_dir = Path(output_dir).resolve()
  output_dir.mkdir(parents=True, exist_ok=True)
  (output_dir / "topologies").mkdir(exist_ok=True)

  config = load_yaml(config_path)
  if epochs_override is not None:
    config["OPTIMIZATION"]["num_epochs"] = epochs_override
  if save_every is not None:
    config["OPTIMIZATION"]["plot_interval"] = save_every

  shutil.copy2(config_path, output_dir / config_path.name)
  with open(output_dir / "run_config.json", "w", encoding="utf-8") as handle:
    json.dump(config, handle, indent=2)

  mesh, material, solver, fourier, topopt_net = build_problem(config,
                                                               nn_seed=nn_seed)
  vae_net, max_feature, min_feature, normalization_types = load_vae(vae_version)
  constraint_type = opt_constraints.ConstraintType[
      config["OPTIMIZATION"]["constraint_type"]]
  sym_params = projection.SymParams(
      sym_x_axis_mid_pt=0.5 * config["BOUNDING_BOX"]["y_max"],
      sym_y_axis_mid_pt=0.5 * config["BOUNDING_BOX"]["x_max"],
  )
  fluid_xy_f, reflection_signs = projected_coordinates(
      mesh, fourier, config, sym_params, resolution=1)

  loss_type = loss.LossTypes[config["LOSS"]["method"]]
  num_constraints = int(config["OPTIMIZATION"].get("num_constraints", 1))
  if loss_type == loss.LossTypes.PENALTY:
    loss_params = loss.PenaltyLossParameters(
        alpha0=config["LOSS"]["alpha0"],
        del_alpha=config["LOSS"]["del_alpha"],
    )
  elif loss_type == loss.LossTypes.AUG_LAG:
    lambda_init = float(config["LOSS"].get("lambda_init", 0.0))
    loss_params = loss.AugmentedLagrangianParameters(
        alpha0=config["LOSS"]["alpha0"],
        del_alpha=config["LOSS"]["del_alpha"],
        alpha=config["LOSS"]["alpha0"],
        lamda=np.full(num_constraints, lambda_init, dtype=np.float64),
    )
  else:
    raise ValueError(f"Unsupported LOSS.method: {config['LOSS']['method']}")

  optimizer = torch.optim.Adam(topopt_net.parameters(),
                               lr=config["OPTIMIZATION"]["learning_rate"])

  # Tier 2.1: optional cosine annealing learning-rate schedule.
  lr_schedule = str(config["OPTIMIZATION"].get("lr_schedule", "none")).lower()
  if lr_schedule == "cosine":
    eta_min = float(config["OPTIMIZATION"].get(
        "lr_min", config["OPTIMIZATION"]["learning_rate"] * 0.025))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config["OPTIMIZATION"]["num_epochs"], eta_min=eta_min)
  else:
    scheduler = None

  # Tier 2.2: optional restart perturbation epochs.
  restart_epochs = config["OPTIMIZATION"].get("restart_perturb_epochs", [])
  restart_sigma = float(config["OPTIMIZATION"].get("restart_perturb_sigma", 0.0))

  # Tier 2.5: optional EMA for J0 (replaces hard epoch 0/20 snapshot).
  j0_ema_alpha = float(config["OPTIMIZATION"].get("j0_ema_alpha", 0.0))

  history = {"epoch": [], "fluid_loss": [], "contact": [],
             "constraint": [], "net_loss": []}
  opt_initial_objective = config["OPTIMIZATION"]["init_objective"]
  start = time.perf_counter()
  final_state = None

  def evaluate(epoch):
    latent_space, theta = topopt_net(fluid_xy_f)
    theta = torch.einsum("i,i->i", theta, reflection_signs["X"])
    theta = torch.einsum("i,i->i", theta, reflection_signs["Y"])
    decoded = vae_net.decoder(latent_space)
    renorm = vae_data_prep.stack_vae_output(
        decoded, max_feature, min_feature, normalization_types)

    if constraint_type == opt_constraints.ConstraintType.VOLUME:
      constraint_field = renorm[:, vae_data_prep.VAE_Fields.shape_area.value]
    else:
      constraint_field = (
          mesh.elem_dx
          * 2.0
          * renorm[:, vae_data_prep.VAE_Fields.shape_perim.value]
      )

    c00 = renorm[:, vae_data_prep.VAE_Fields.homog_c00.value] + 1e-6
    c11 = renorm[:, vae_data_prep.VAE_Fields.homog_c11.value] + 1e-6
    fluid_loss, velocity_pressure = solver.fluid_objective_function(
        material, c00, c11, theta)
    field_constraint = opt_constraints.constraint_function(
        constraint_type,
        constraint_field,
        config["OPTIMIZATION"]["desired_vol_frac"],
        config["OPTIMIZATION"]["desired_perimeter"],
    )
    if j0_ema_alpha > 0.0:
      # EMA J0: smooth update each epoch (replaces 0/20 snapshot).
      objective_scale = opt_initial_objective
    else:
      objective_scale = (fluid_loss.item() if epoch in (0, 20)
                         else opt_initial_objective)
    objective_scale = max(objective_scale, 1e-12)
    net_loss = loss.combined_loss(
        objective=fluid_loss / objective_scale,
        constraints=[field_constraint],
        loss_type=loss_type,
        loss_params=loss_params,
        epoch=epoch,
    )
    # Per-cell minimum-area penalty (TOMAS §3.7). Off by default (weight=0).
    solid_vol_frac = renorm[:, vae_data_prep.VAE_Fields.shape_area.value]
    min_solid_vol = float(
        config["OPTIMIZATION"].get("min_solid_vol_per_cell", 0.0))
    min_area_weight = float(
        config["OPTIMIZATION"].get("min_area_penalty_weight", 0.0))
    min_area_pen = opt_constraints.min_area_penalty(
        solid_vol_frac, min_solid_vol)
    if min_area_weight > 0.0:
      net_loss = net_loss + min_area_weight * min_area_pen
    mstr_data = renorm[:, :vae_data_prep.VAE_Fields.homog_c00.value].detach().numpy()
    mstr_params = supershape.SuperShapes(
        mstr_data[:, 0], mstr_data[:, 1], mstr_data[:, 2], mstr_data[:, 3],
        mstr_data[:, 4], mstr_data[:, 5], mstr_data[:, 6], mstr_data[:, 7])
    return (net_loss, fluid_loss, field_constraint, constraint_field,
            mstr_params, theta, velocity_pressure, solid_vol_frac,
            min_area_pen, c00, c11)

  min_solid_vol_per_cell = float(
      config["OPTIMIZATION"].get("min_solid_vol_per_cell", 0.0))
  history["min_area_penalty"] = []
  history["min_area_violation_cells"] = []
  for epoch in range(config["OPTIMIZATION"]["num_epochs"]):
    optimizer.zero_grad()
    (net_loss, fluid_loss, constraint, constraint_field, mstr_params,
     theta, velocity_pressure, solid_vol_frac,
     min_area_pen, _c00_dec, _c11_dec) = evaluate(epoch)
    net_loss.backward()
    if config["OPTIMIZATION"]["grad_clip_activation"] == "GRAD_CLIP_ON":
      torch.nn.utils.clip_grad_norm_(
          topopt_net.parameters(), config["OPTIMIZATION"]["grad_clip_norm"])
    optimizer.step()
    if scheduler is not None:
      scheduler.step()
    loss.update_loss_parameters(epoch, loss_type, loss_params, [constraint])

    # Tier 2.2: restart perturbation at configured epochs.
    if restart_sigma > 0.0 and epoch in restart_epochs:
      with torch.no_grad():
        for p in topopt_net.parameters():
          p.add_(restart_sigma * torch.randn_like(p))
      print(f"  [restart] perturbed weights with sigma={restart_sigma} "
            f"at epoch {epoch}", flush=True)

    with torch.no_grad():
      (net_loss, fluid_loss, constraint, constraint_field, mstr_params,
       theta, velocity_pressure, solid_vol_frac,
       min_area_pen, c00_dec, c11_dec) = evaluate(epoch)
    # Update opt_initial_objective: EMA (Tier 2.5) or snapshot at 0/20.
    if j0_ema_alpha > 0.0:
      if epoch == 0:
        opt_initial_objective = fluid_loss.item()
      else:
        opt_initial_objective = (
            j0_ema_alpha * opt_initial_objective
            + (1.0 - j0_ema_alpha) * fluid_loss.item())
    elif epoch in (0, 20):
      opt_initial_objective = fluid_loss.item()

    contact = torch.sum(constraint_field).item()
    violation_cells = int((solid_vol_frac < min_solid_vol_per_cell
                           ).sum().item()) if min_solid_vol_per_cell > 0 else 0
    history["epoch"].append(epoch)
    history["fluid_loss"].append(fluid_loss.item())
    history["contact"].append(contact)
    history["constraint"].append(constraint.item())
    history["net_loss"].append(net_loss.item())
    history["min_area_penalty"].append(float(min_area_pen.item()))
    history["min_area_violation_cells"].append(violation_cells)
    final_state = (mstr_params, theta.detach().numpy(),
                   velocity_pressure.detach(),
                   c00_dec.detach().numpy(), c11_dec.detach().numpy())

    print(f"Iter {epoch:3d} J: {fluid_loss.item():.4E}; contact: {contact:.4F}; "
          f"constraint: {constraint.item():.4E}; "
          f"min_area_pen: {min_area_pen.item():.4E}; "
          f"violations: {violation_cells}", flush=True)

    if (epoch % config["OPTIMIZATION"]["plot_interval"] == 0
        or epoch == config["OPTIMIZATION"]["num_epochs"] - 1):
      save_microstructure_plot(
          mstr_params,
          theta.detach().numpy(),
          mesh.nelx,
          mesh.nely,
          output_dir / "topologies" / f"topology_epoch_{epoch:04d}.png",
          f"{config['BOUNDARY_CONDITIONS']['example']} epoch {epoch}",
      )
      write_history(history, output_dir / "history.csv")

  write_history(history, output_dir / "history.csv")
  save_history_plot(history, output_dir / "history.png")

  mstr_params, theta_np, velocity_pressure, c00_decoder, c11_decoder = final_state
  # Save final state for post-hoc validation (Tier 2.4).
  np.savez(output_dir / "final_state.npz",
           shape_a=mstr_params.a, shape_b=mstr_params.b,
           shape_m=mstr_params.m, shape_n1=mstr_params.n1,
           shape_n2=mstr_params.n2, shape_n3=mstr_params.n3,
           shape_cx=mstr_params.center_x, shape_cy=mstr_params.center_y,
           theta=theta_np,
           c00_decoder=c00_decoder, c11_decoder=c11_decoder,
           nelx=np.array([mesh.nelx]), nely=np.array([mesh.nely]))
  u_vel, v_vel, pressure = solver.get_element_velocity_pressure(velocity_pressure)
  velocity_magnitude = torch.sqrt(u_vel ** 2 + v_vel ** 2).detach().numpy()
  save_microstructure_plot(
      mstr_params, theta_np, mesh.nelx, mesh.nely,
      output_dir / "final_topology.png",
      f"{config['BOUNDARY_CONDITIONS']['example']} final topology",
  )
  save_field(velocity_magnitude, mesh.nelx, mesh.nely,
             output_dir / "velocity_magnitude.png",
             "velocity magnitude", "turbo")
  save_field(pressure.detach().numpy(), mesh.nelx, mesh.nely,
             output_dir / "pressure.png", "pressure", "viridis")

  # Post-hoc shapely re-evaluation of contact area (TOMAS §3.3 validation).
  # Rebuilds each cell's super-shape polygon from the decoder-predicted
  # shape parameters and re-computes its perimeter using Shapely.
  try:
    polygons, _ = supershape.super_shape_to_shapely_polygon(
        mstr_params, prune_out_intersecting_shapes=False)
    shapely_perims = supershape.compute_shapely_polygon_perimeter(polygons)
    domain_sum = (mstr_params.domain_length_x
                  + mstr_params.domain_length_y)
    normalized_perim = shapely_perims / domain_sum
    shapely_contact = float(
        mesh.elem_dx * 2.0 * normalized_perim.sum())
  except Exception as exc:  # noqa: BLE001
    shapely_contact = None
    print(f"WARNING: shapely post-hoc perim failed: {exc}", flush=True)

  summary = {
      "config": str(config_path),
      "vae_version": vae_version,
      "nn_seed": nn_seed,
      "example": config["BOUNDARY_CONDITIONS"]["example"],
      "epochs": config["OPTIMIZATION"]["num_epochs"],
      "elapsed_seconds": time.perf_counter() - start,
      "final_fluid_loss_decoder": history["fluid_loss"][-1],
      "final_contact_decoder": history["contact"][-1],
      "shapely_contact_posthoc": shapely_contact,
      "final_constraint": history["constraint"][-1],
      "min_solid_vol_per_cell": min_solid_vol_per_cell,
      "min_area_penalty_weight": float(
          config["OPTIMIZATION"].get("min_area_penalty_weight", 0.0)),
      "final_min_area_penalty": history["min_area_penalty"][-1],
      "final_min_area_violation_cells": history["min_area_violation_cells"][-1],
      "total_cells": mesh.nelx * mesh.nely,
  }
  with open(output_dir / "summary.json", "w", encoding="utf-8") as handle:
    json.dump(summary, handle, indent=2)
  print(json.dumps(summary, indent=2), flush=True)


def run_case_cli(config_path, output_dir, epochs_override=None, save_every=None,
                 vae_version: str = "v1"):
  """Backwards-compatible CLI alias."""
  return run_case(config_path, output_dir, epochs_override, save_every,
                  vae_version)


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--config", required=True)
  parser.add_argument("--out", required=True)
  parser.add_argument("--epochs", type=int)
  parser.add_argument("--save-every", type=int)
  parser.add_argument("--vae-version", default="v1",
                      choices=list(VAE_FILES.keys()))
  args = parser.parse_args()
  run_case(args.config, args.out, args.epochs, args.save_every,
           args.vae_version)


if __name__ == "__main__":
  main()
