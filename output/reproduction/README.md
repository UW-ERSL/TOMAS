# TOMAS Reproduction Summary

Source paper: `C:\Users\Bingxiao Du\Downloads\s00158-024-03835-6.pdf`

Important scope note: the paper reports training on 7000 microstructures. The
checked-in repository configuration `notebooks/datagen.yaml` uses 100 samples,
and this local reproduction uses that repository-default dataset plus the VAE
trained locally from it. The numerical trends and constraints reproduce the
paper workflow, but exact paper values require regenerating the 7000-sample
offline dataset and retraining the VAE.

## Environment

- Python: `.venv` with CPython 3.10.20
- PyTorch: 2.1.2+cpu
- NumPy: 1.26.3
- SciPy: 1.10.0
- MATLAB: `D:\Matlab\bin\matlab.exe` for homogenized microstructure data

## Paper Targets

- Fig. 12b bent pipe, contact-area constraint 75.69: paper reports decoder
  dissipated power 7.56 and validation dissipated power 7.87/contact area 78.49.
- Fig. 13 diffuser convergence, contact-area constraint 60: paper illustrates
  convergence at epochs 0, 20, 100, and 300.
- Fig. 16 fabrication/bifurcated-style case: design uses a contact-area
  constraint near 70 in the paper; repository config uses 75.

## Local Runs

| Case | Config | Epochs | Final dissipated power | Final contact area | Output |
| --- | --- | ---: | ---: | ---: | --- |
| Diffuser | `notebooks/config_diffuser.yaml` | 301 | 17.5102 | 59.5128 | `output/reproduction/diffuser` |
| Bent pipe | `notebooks/config_bent_pipe.yaml` | 301 | 6.2470 | 76.2769 | `output/reproduction/bent_pipe` |
| Bifurcated pipe | `notebooks/config_biffurcated_pipe.yaml` | 351 | 21.6099 | 75.1554 | `output/reproduction/bifurcated_pipe` |

## Commands

```powershell
cd "C:\Users\Bingxiao Du\Documents\TOMAS"
.\.venv\Scripts\python.exe -u scripts\reproduce_tomas.py --config notebooks\config_diffuser.yaml --out output\reproduction\diffuser
.\.venv\Scripts\python.exe -u scripts\reproduce_tomas.py --config notebooks\config_bent_pipe.yaml --out output\reproduction\bent_pipe
.\.venv\Scripts\python.exe -u scripts\reproduce_tomas.py --config notebooks\config_biffurcated_pipe.yaml --out output\reproduction\bifurcated_pipe
```

Each output directory contains:

- `summary.json`: final scalar values and elapsed time
- `history.csv`: epoch, dissipated power, contact area, constraint, loss
- `history.png`: convergence plot
- `final_topology.png`: final macro/microstructure layout
- `velocity_magnitude.png` and `pressure.png`: final flow fields
- `topologies/`: topology snapshots at configured plot intervals

The combined preview image is `output/reproduction/final_topologies_montage.png`.
