"""Download the Dong et al. 2023 dataset via pertpy and save it to data/ for analysis.ipynb."""
from pathlib import Path

import pertpy as pt

OUT = Path("data/dong_2023_raw.h5ad")
OUT.parent.mkdir(parents=True, exist_ok=True)

if OUT.exists():
    print("Already present:", OUT)
else:
    adata = pt.dt.dong_2023()
    adata.write_h5ad(OUT)
    print("Saved", OUT, adata.shape)
