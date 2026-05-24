# TOMAS — WSL workflow

## TL;DR

Run all Python on WSL Ubuntu 22.04 with the `tomas-wsl` conda env. This
unlocks the `torch_sparse_solve` fast path in
[fluid_TO/fluid_fe.py](fluid_TO/fluid_fe.py) (vs. scipy `splu` fallback on
Windows). Measured speedup on bent_pipe (1200 cells, 50 epochs):

| Backend            | 50-epoch time | Speedup |
| ------------------ | ------------- | ------- |
| Windows scipy splu | ~310 s        | 1.0x    |
| **WSL torch_sparse_solve** | **126 s**     | **~2.5x** |

Diffuser (225 cells) speedup is negligible (~1.1x) because the sparse
solve is not the bottleneck on small meshes.

## Environment

Conda env `tomas-wsl` lives at `/home/dubin/miniconda3/envs/tomas-wsl`.

Key facts:
- Python 3.10.20
- torch 2.1.2+cpu
- torch_sparse_solve 0.0.5 — **rebuilt from source against this env's
  torch**, NOT the wheel from PyPI (which had ABI mismatch).
- SuiteSparse / KLU shipped via `conda install -c conda-forge suitesparse=5`
- numpy 1.26.3, scipy 1.10.0 — matches paper requirements.txt
- pyyaml, matplotlib, shapely, geopandas, pandas

## Project location

Code lives on `/mnt/c/Users/Bingxiao Du/Documents/TOMAS` (Windows
filesystem). The Linux env reads files over the 9p mount. Heavy IO
(homogenization MATLAB output, VAE checkpoints) is OK; the bottleneck
that mattered was the linear solve in fluid_fe.py, not file IO.

## Running

Use the wrapper script which sets `PYTHONNOUSERSITE=1`:

```bash
# From inside WSL:
cd /mnt/c/Users/Bingxiao\ Du/Documents/TOMAS
bash scripts/run_wsl.sh scripts/reproduce_tomas.py --config notebooks/config_diffuser_a60.yaml --vae-version v6 --out output/wsl_diffuser

# From Windows PowerShell:
wsl -d Ubuntu-22.04 -- bash "/mnt/c/Users/Bingxiao Du/Documents/TOMAS/scripts/run_wsl.sh" scripts/reproduce_tomas.py --config notebooks/config_diffuser_a60.yaml --vae-version v6 --out output/wsl_diffuser
```

### Why PYTHONNOUSERSITE=1 matters

`~/.local/lib/python3.10/site-packages/torch_sparse_solve_cpp.cpython-310-x86_64-linux-gnu.so`
was compiled against a *different* torch version (likely the system's
former install). Without isolation it shadows the env-internal build and
raises:

```
ImportError: ... undefined symbol: _ZN2at4_ops25sparse_coo_tensor_indices4callE...
```

The wrapper sets `PYTHONNOUSERSITE=1` so the env's own `.so` is loaded.

### Verify the fast path is active

```bash
bash scripts/run_wsl.sh scripts/test_wsl_env.py
```

Expected output ends with:
```
fluid_fe _USING_SCIPY_SPARSE_SOLVE: False (False = torch_sparse_solve fast path)
```

If `True`, the fallback is in use — check torch_sparse_solve import in the
env.

## MATLAB integration (still Windows)

MATLAB homogenization remains a Windows binary call. From WSL it's
reachable as `/mnt/d/Matlab/bin/matlab.exe`. The existing `.m` files
under `dataset/` work as-is when launched from inside WSL via PowerShell
`-batch` (see `logs/matlab_homog_v*.log` for past runs).

If you need to re-run MATLAB homogenization from within WSL:

```bash
cd /mnt/c/Users/Bingxiao\ Du/Documents/TOMAS/dataset
/mnt/d/Matlab/bin/matlab.exe -batch "cd('C:\Users\Bingxiao Du\Documents\TOMAS\dataset'); run('run_homogenization_v4.m'); exit;"
```

(MATLAB needs Windows paths even when launched from WSL.)

## Re-creating the env from scratch

```bash
~/miniconda3/bin/conda create -n tomas-wsl python=3.10 -y
~/miniconda3/bin/conda install -n tomas-wsl -c conda-forge suitesparse=5 -y

PY=~/miniconda3/envs/tomas-wsl/bin/python
PIP="PYTHONNOUSERSITE=1 $PY -m pip install --force-reinstall"

$PIP torch==2.1.2 --index-url https://download.pytorch.org/whl/cpu
$PIP --no-deps 'numpy==1.26.3' 'scipy==1.10.0' pyyaml pyparsing
$PIP --no-deps typing_extensions sympy networkx filelock jinja2 MarkupSafe fsspec mpmath
$PIP --no-deps 'setuptools<81'  # need pkg_resources for torch.utils.cpp_extension
$PIP --no-deps matplotlib shapely geopandas pyogrio pyproj pytz certifi six \
                packaging contourpy cycler kiwisolver fonttools pillow python-dateutil
$PIP --no-deps 'pandas<2.3'

# Build torch_sparse_solve from source against THIS env's torch:
git clone https://github.com/flaport/torch_sparse_solve.git /tmp/torch_sparse_solve
cd /tmp/torch_sparse_solve
PYTHONNOUSERSITE=1 $PY setup.py build_ext --inplace
PYTHONNOUSERSITE=1 $PY setup.py install
```

## Output directory layout reminder

```
output/
├── paper_reproduction/         # v9 final (run on Windows scipy)
├── paper_reproduction_v{2..8}/ # history
└── reproduction/               # original 100-sample baseline
```

If you re-run on WSL, output paths are identical (Windows fs is shared).
Consider backing up `paper_reproduction/` first if you don't want to
overwrite v9.
