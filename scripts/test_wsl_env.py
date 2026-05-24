"""Sanity check that WSL conda env has torch + torch_sparse_solve fast path."""
import sys
import torch
print("Python   :", sys.version.split()[0])
print("torch    :", torch.__version__)
try:
    import torch_sparse_solve
    from torch_sparse_solve import solve
    print("torch_sparse_solve:", "ok (fast path enabled)")
except ImportError as e:
    print("torch_sparse_solve:", "NOT AVAILABLE -", e)
import numpy, scipy, yaml, matplotlib, shapely
print("numpy    :", numpy.__version__)
print("scipy    :", scipy.__version__)
print("yaml     :", yaml.__version__)
print("mpl      :", matplotlib.__version__)
print("shapely  :", shapely.__version__)

# Verify fluid_fe.py picks the fast path
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "fluid_TO"))
import fluid_fe
print("fluid_fe _USING_SCIPY_SPARSE_SOLVE:",
      fluid_fe._USING_SCIPY_SPARSE_SOLVE,
      "(False = torch_sparse_solve fast path)")
