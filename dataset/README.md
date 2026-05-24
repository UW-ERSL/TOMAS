# Dataset directory

This directory holds the **homogenization inputs/outputs** used to train the VAE.
Several files are **excluded from git** because they are too large for GitHub
(>100 MB per file) but are trivially re-generatable:

| File pattern | Size | Status | How to regenerate |
| --- | --- | --- | --- |
| `mstr_shape_parameters_{1..4}.mat` | ~6 KB | **tracked** | `scripts/run_offline_dataset.py` |
| `mstr_area_{1..4}.mat` | ~1 KB | **tracked** | (same) |
| `mstr_perim_{1..4}.mat` | ~1 KB | **tracked** | (same) |
| `homogen_data_{1..4}.mat` | ~4-220 KB | **tracked** | MATLAB `run_homogenization_v{2,3,4}.m` |
| `mstr_images_{1..4}.mat` | 17 MB – 1.2 GB | git-ignored | `scripts/run_offline_dataset.py` |
| `_posthoc_*_in.mat` | 40-206 MB | git-ignored | `scripts/post_hoc_matlab_validate.py` |
| `_posthoc_*_out.mat` | ~5-25 KB | tracked | (same) |

## Dataset versions

| `dataset_num` | super-shape `m` range | sample count | used by VAE |
| --- | --- | --- | --- |
| 1 | [0.5, 11] | 100 | VAE v1 (legacy) |
| 2 | [1, 22] | 7000 | VAE v2 / v3 |
| 3 | [2, 8] | 7000 | VAE v5 |
| 4 | [3, 6] | 7000 | **VAE v6 (final)** |

## End-to-end regeneration of `dataset_4`

```bash
# 1. Update notebooks/datagen.yaml: dataset_num: 4, min_m: 3, max_m: 6
# 2. Generate super-shape rasters (~2 min):
python scripts/run_offline_dataset.py

# 3. MATLAB homogenization (~75 min, Windows binary):
matlab -batch "cd('dataset'); run('run_homogenization_v4.m'); exit;"

# 4. Train VAE v6 (~3 h on CPU):
python scripts/train_vae_v6.py
```
