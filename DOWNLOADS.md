# Data status

Expected folder layout (everything sits next to the notebook):

```
martiny2026_reproduction.ipynb
thetaPrime_MODIS.mat              <- you have (Dryad)
thetaPrime_seawifs.mat            <- you have (Dryad)
ersstv5.nc                        <- downloaded automatically on first run
oni.ascii.txt                     <- included (NOAA CPC)
ersst.v5.pdo.dat                  <- included (NOAA NCEI)
Dataset_S1.xlsx                   <- included (Gerace et al. 2025)
SST_v2_lowlat.npz                 <- optional: make it with shrink_sst.py (see below)
woa23_all_n00_01.nc               <- optional
zenodo_8064615/
    data_insitu_v2_25-04.csv      <- included
    sat_data_v2_25-04/genomic_OCmatchups_v3_medians_{1p,3p,5p,1d,2d,5d}.csv   <- included
```

## The two large Zenodo files

**`theta_v2_25-04.csv` — not needed.** It holds the same 1°, 8-day Θ′ grid as `thetaPrime_MODIS.mat`, which the
notebook already uses. Skip it.

**`SST_v2_25-04.csv` — optional, and only for the SST panels** (Fig. 4A/C/D SST values, S15, S16). Without it the
notebook uses NOAA ERSSTv5, which gives SST warming over 83% of the area (paper: 89%) and a median of 1.65e-2 °C/yr
(paper: 2.3e-2). None of the Θ′ results depend on it. Three ways to use it:

The file's longitude column is mislabelled (see README); the notebook repairs it automatically and prints
`SST csv coordinates: ... -> using: ...`. If it reports "as written in the file" or a match below ~0.95, something else is off.

1. **Shrink it, then upload or use it** (recommended). On your computer:
   ```
   python shrink_sst.py SST_v2_25-04.csv
   ```
   This writes `SST_v2_lowlat.npz`. It keeps only 40.5°S–40.5°N and stores SST to 0.01 °C as compact integers,
   typically a few tens of MB. Put it next to the notebook, or upload it here and I'll rerun.
   The script reads the csv in chunks, so it needs only numpy and pandas and a few GB of RAM.
2. **Compress it** (`gzip SST_v2_25-04.csv` → `SST_v2_25-04.csv.gz`). The notebook reads `.csv.gz` directly. A text
   table of numbers usually shrinks 3–6×, which may be enough to upload.
3. **Run the notebook on your own computer** with the full csv in the folder. It detects the file automatically.

## Optional: World Ocean Atlas 2023
`woa23_all_n00_01.nc` (nitrate, annual, 1°) is the paper's source for the nitracline map in Fig. 2B:
https://www.ncei.noaa.gov/thredds-ocean/fileServer/woa23/DATA/nitrate/netcdf/all/1.00/woa23_all_n00_01.nc
Without it the notebook uses Dataset S1 (`WOD Means (no3=3)`), which already gives Fig. 1A r = −0.55 (paper −0.52).
If you add WOA, Fig. 1A and 2B switch to it, while Fig. 4D keeps Dataset S1 as in the paper.

## Not reproducible from public files
- Fig. 1D (individual nitrogen-trait genes): needs per-gene abundance tables from the Bio-GO-SHIP metagenomes.
- Fig. S3 (dilution and nutrient-addition experiments): from Sherman et al. 2016 and Browning & Moore 2023, not in the
  deposit.
