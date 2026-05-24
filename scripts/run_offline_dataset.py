"""Generate the offline super-shape dataset described in TOMAS paper §2.3.

Reads notebooks/datagen.yaml, draws random super-shape parameters, projects each
shape onto a 150x150 raster (for MATLAB homogenization), and computes per-sample
normalized contact area and volume fraction. Writes four .mat files in
dataset/, named with the dataset_num suffix from the YAML.
"""
import sys
import time
from pathlib import Path

import numpy as np
import scipy.io
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "dataset")]

import supershape as ss


def main():
  config_path = ROOT / "notebooks" / "datagen.yaml"
  with open(config_path, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

  shape_yaml = config["SUPERSHAPE"]
  extents = ss.SuperShapeExtents(
      a=ss.Extents(shape_yaml["min_a"], shape_yaml["max_a"]),
      b=ss.Extents(shape_yaml["min_b"], shape_yaml["max_b"]),
      m=ss.Extents(shape_yaml["min_m"], shape_yaml["max_m"]),
      n1=ss.Extents(shape_yaml["min_n1"], shape_yaml["max_n1"]),
      n2=ss.Extents(shape_yaml["min_n2"], shape_yaml["max_n2"]),
      n3=ss.Extents(shape_yaml["min_n3"], shape_yaml["max_n3"]),
      center_x=ss.Extents(shape_yaml["min_cx"], shape_yaml["max_cx"]),
      center_y=ss.Extents(shape_yaml["min_cy"], shape_yaml["max_cy"]),
  )
  seed = shape_yaml.get("shape_seed", 27)
  target = config["DATASET"]["num_samples"]
  dataset_num = config["DATASET"]["dataset_num"]
  nelx = config["MESH"]["nelx"]
  nely = config["MESH"]["nely"]

  # Oversample so the pruning step leaves us with >= target samples; trim later.
  oversample = int(np.ceil(target * 1.5))
  print(f"Sampling {oversample} candidate super-shapes (target {target})...",
        flush=True)
  start = time.perf_counter()
  random_param = ss.generate_random_super_shapes(oversample, extents, seed=seed)
  print(f"  ... done in {time.perf_counter() - start:.1f}s", flush=True)

  print("Converting to Shapely polygons + pruning out-of-bounds shapes...",
        flush=True)
  start = time.perf_counter()
  polygons, pruned = ss.super_shape_to_shapely_polygon(random_param)
  print(f"  ... {len(polygons)} polygons kept "
        f"(dropped {oversample - len(polygons)}) in "
        f"{time.perf_counter() - start:.1f}s",
        flush=True)

  if len(polygons) < target:
    print(f"WARNING: kept ({len(polygons)}) < target ({target}); "
          f"using all surviving samples.", flush=True)
    keep = len(polygons)
  else:
    keep = target
  polygons = polygons[:keep]
  pruned = ss.SuperShapes(
      a=pruned.a[:keep], b=pruned.b[:keep], m=pruned.m[:keep],
      n1=pruned.n1[:keep], n2=pruned.n2[:keep], n3=pruned.n3[:keep],
      center_x=pruned.center_x[:keep], center_y=pruned.center_y[:keep])

  print(f"Rasterizing {keep} shapes to {nelx}x{nely} density field...",
        flush=True)
  start = time.perf_counter()
  shape_density = ss.project_shapely_polygons_to_density(
      polygons, nelx, nely, True)
  print(f"  ... done in {time.perf_counter() - start:.1f}s "
        f"(shape={shape_density.shape}, dtype={shape_density.dtype})",
        flush=True)

  print("Computing perimeter and area...", flush=True)
  shape_perim = ss.compute_shapely_polygon_perimeter(polygons)
  shape_area = ss.compute_shapely_polygon_area(polygons)
  # Normalize to the unit cell (the bounding box is 2x2; lengths are 2 and 2).
  normalized_area = (shape_area
                     / (pruned.domain_length_x * pruned.domain_length_y))
  normalized_perim = (shape_perim
                      / (pruned.domain_length_x + pruned.domain_length_y))

  shape_params = pruned.to_stacked_array()

  out_dir = ROOT / "dataset"
  files = {
      f"mstr_shape_parameters_{dataset_num}.mat": ("mstr_shape_parameters",
                                                   shape_params),
      f"mstr_images_{dataset_num}.mat": ("mstr_images", shape_density),
      f"mstr_area_{dataset_num}.mat": ("mstr_area", normalized_area),
      f"mstr_perim_{dataset_num}.mat": ("mstr_perim", normalized_perim),
  }
  for fname, (key, value) in files.items():
    path = out_dir / fname
    scipy.io.savemat(path, {key: value})
    print(f"  wrote {path}  key={key}  shape={np.asarray(value).shape}",
          flush=True)

  # Quick sanity prints aligned with §2.3 ranges
  print(f"\nSanity check:")
  print(f"  a   : [{pruned.a.min():.3f}, {pruned.a.max():.3f}]")
  print(f"  b   : [{pruned.b.min():.3f}, {pruned.b.max():.3f}]")
  print(f"  m   : [{pruned.m.min():.3f}, {pruned.m.max():.3f}]")
  print(f"  n1  : [{pruned.n1.min():.3f}, {pruned.n1.max():.3f}]")
  print(f"  n2  : [{pruned.n2.min():.3f}, {pruned.n2.max():.3f}]")
  print(f"  n3  : [{pruned.n3.min():.3f}, {pruned.n3.max():.3f}]")
  print(f"  norm_area : [{normalized_area.min():.3f}, "
        f"{normalized_area.max():.3f}]")
  print(f"  norm_perim: [{normalized_perim.min():.3f}, "
        f"{normalized_perim.max():.3f}]")


if __name__ == "__main__":
  main()
