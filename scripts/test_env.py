import os
from scipy.io import loadmat

# Get project root (one level above scripts)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

fix_path = os.path.join(PROJECT_ROOT,
                        "data",
                        "FIXATIONLOCS",
                        "Action",
                        "053.mat")

mat = loadmat(fix_path)

print(mat.keys())
print(mat["fixLocs"].shape)
print(mat["fixLocs"].sum())