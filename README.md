# Replication of Martiny et al. (2026), *Genomic-to-space measurements reveal large-scale ocean nutrient stress*

*Science Advances* 12, eaed8089 — doi:[10.1126/sciadv.aed8089](https://doi.org/10.1126/sciadv.aed8089)

This repository rebuilds the paper's figures and headline numbers in Python. The processing follows the authors' own
MATLAB/R code ([ljustick/genomic_to_space_nut_stress](https://github.com/ljustick/genomic_to_space_nut_stress)):
pre-processing of Θ′, two-way ANOVA (year × month), EOF analysis, climate-index comparison, trends by region, and
random forests on the in situ data. Low Θ′ means more nutrient stress.

## Key results (paper vs this replication)

| Quantity | Paper | This replication |
|---|---|---|
| Variance in EOF PC1–4 / PC5–8 | 59.4% / 4.2% | 58.5% / 4.1% |
| PC5 vs ONI (ENSO) / PC6 vs PDO | R = 0.81 / 0.59 | 0.83 / 0.62 |
| Θ′ vs nitracline depth (Fig. 1A) | r = −0.52 | −0.66 (WOA 2023) |
| In situ correlations (Fig. S2) | published bars | all within 0.002 |
| Median trend: Θ′ / SST | −1.3×10⁻³ / 2.3×10⁻² yr⁻¹ | −1.23×10⁻³ / 2.34×10⁻² yr⁻¹ |
| Ocean area with SST warming | 89% | 89% |
| Θ′ trend ×10, SH low-nutrient / NH low-nutrient / SH high-nutrient | ≈ +0.03 / −0.05 / −0.007 | +0.026 / −0.042 / −0.005 |
| SST vs Θ′, seasonal / year-to-year | r = −0.51 / −0.23 | −0.43 / −0.20 |
| RF R², nitracline + biomarkers / nitracline only | ≈ 0.62 / 0.36 | 0.51 / 0.27 |

Full table: `figures/summary.csv`. Side-by-side images: `comparison/` (published figure on top, replication below).

Known differences: random-forest R² values are lower (scikit-learn vs R `randomForest`); two Fig. 1C bars
(Ω N,med and Ω Fe,high) have the opposite sign; the SeaWiFS–MODIS all-points correlation is 0.89 vs 0.74. Fig. 1D
and Fig. S3 cannot be rebuilt from public data (gene-level tables and incubation experiments are not released).

## Repository contents

| Path | What it is |
|---|---|
| `martiny2026_reproduction.ipynb` | The notebook, executed on the full data set |
| `martiny2026_reproduction.py` | Same notebook as a plain script (jupytext percent format) |
| `figures/` | All reproduced figures + `summary.csv` |
| `comparison/` | Each published figure stacked over the reproduction |
| `oni.ascii.txt`, `ersst.v5.pdo.dat` | Official NOAA ONI and PDO indices |
| `Dataset_S1.xlsx` | Gerace et al. (2025) nitracline depths (Fig. 4D regions) |
| `zenodo_8064615/` | Authors' in situ table and satellite match-ups (Fig. 1B–C, S2, S4, S5, S17) |
| `shrink_sst.py` | Shrinks the 630 MB SST csv to a small `.npz` |
| `DOWNLOADS.md` | Detailed data notes |

## Data not included (too large for GitHub)

These files are listed in `.gitignore`. Download them and place them in the repository root:

| File | Size | Needed for | Source |
|---|---|---|---|
| `thetaPrime_MODIS.mat`, `thetaPrime_seawifs.mat` | 106 MB, 59 MB | **Required**: all satellite analyses | Dryad, doi:[10.5061/dryad.f4qrfj71p](https://doi.org/10.5061/dryad.f4qrfj71p) |
| `SST_v2_25-04.csv` | 630 MB | Optional: MODIS SST for Fig. 4, S15, S16 | Zenodo, doi:[10.5281/zenodo.8064615](https://doi.org/10.5281/zenodo.8064615) |
| `woa23_all_n00_01.nc` | ~80 MB | Optional: nitracline map (Fig. 2B) and Fig. 1A | [NOAA World Ocean Atlas 2023, nitrate, annual, 1°](https://www.ncei.noaa.gov/thredds-ocean/fileServer/woa23/DATA/nitrate/netcdf/all/1.00/woa23_all_n00_01.nc) |
| `ersstv5.nc` | 19 MB | SST fallback (downloaded automatically) | NOAA ERSSTv5 via the xarray tutorial data |

Without the optional files the notebook falls back to ERSSTv5 for SST and to Dataset S1 for the nitracline, and prints
which source it used. `theta_v2_25-04.csv` from Zenodo is not needed (it duplicates `thetaPrime_MODIS.mat`).

## How to run

```
pip install numpy scipy pandas xarray netCDF4 h5py matplotlib scikit-learn openpyxl cmocean jupyter
jupyter notebook martiny2026_reproduction.ipynb      # then Run All (a few minutes)
```

Figures are written to `figures/`.

## Notes on the authors' SST file

- The longitude column of `SST_v2_25-04.csv` does not match its rows: the grid is written column by column
  (180 latitudes per longitude) but the longitude list was repeated. The notebook detects this, rebuilds each row's
  longitude from its row number, checks the result against ERSSTv5 and prints which ordering it used.
- MODIS SST has cloud gaps. As the authors do, pixels missing more than 25% of composites are dropped and the rest are
  gap-filled and smoothed (5-point lowess) before trends and ANOVA.
- If you made `SST_v2_lowlat.npz` with an earlier `shrink_sst.py`, delete it and run the script again.

## Credits

Paper, figures and code by Martiny, Ustick, Westberry & Behrenfeld (CC BY 4.0). Data from Dryad, Zenodo, NOAA (ERSSTv5,
ONI, PDO, WOA 2023) and Gerace et al. (2025, *PNAS* 122, e2411835122). Published figures in `comparison/` are
reproduced under CC BY 4.0 for comparison.