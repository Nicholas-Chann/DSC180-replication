# Martiny et al. (2026) Sci. Adv. 12, eaed8089 — reproduction notebook

- `martiny2026_reproduction.ipynb`: the notebook, executed on the full data set (MODIS SST, WOA 2023, in situ tables, official indices), with figures and numbers inside.
- `martiny2026_reproduction.py`: the same notebook as a plain Python script (jupytext percent format).
- `figures/`: every figure from the full-data run, in the paper's style, plus `summary.csv` (paper vs reproduced).
- `comparison/`: each published figure (top) stacked over the reproduction (bottom).
- Data included: official ONI and PDO, Gerace et al. Dataset S1, and the authors' in situ and match-up tables (Zenodo).
- `DOWNLOADS.md`: folder layout and what to do with the large Zenodo files.
- `shrink_sst.py`: turns the large `SST_v2_25-04.csv` into a small `SST_v2_lowlat.npz` (keeps each row's line number, needed for the coordinate repair below).

Notes on the authors' `SST_v2_25-04.csv`:
- Its longitude column does not match its rows: the grid is written column by column (180 latitudes per longitude) but the longitude list was repeated. The notebook detects this, rebuilds each row's longitude from its row number, checks the result against ERSSTv5 and prints which ordering it used.
- MODIS SST has cloud gaps. As the authors do, pixels missing >25% of composites are dropped and the rest are gap-filled and smoothed (5-point lowess) before trends and ANOVA.
- If you made `SST_v2_lowlat.npz` with an earlier `shrink_sst.py`, delete it and run the script again.

To run: add `thetaPrime_MODIS.mat` and `thetaPrime_seawifs.mat` to this folder, then run all cells (~2 min).
Requirements: `pip install numpy scipy pandas xarray netCDF4 h5py matplotlib scikit-learn openpyxl cmocean`
The processing follows the authors' own MATLAB/R code: github.com/ljustick/genomic_to_space_nut_stress
