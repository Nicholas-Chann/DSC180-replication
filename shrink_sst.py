"""Shrink the authors' SST_v2_25-04.csv (Zenodo 10.5281/zenodo.8064615) to a small file the notebook can read.

Keeps only 40.5S-40.5N (the paper's domain) and stores SST as int16 hundredths of a degree (exact to 0.01 C),
compressed. Reads the csv in chunks, so it works on an ordinary laptop.

    python shrink_sst.py SST_v2_25-04.csv            ->  SST_v2_lowlat.npz   (put it next to the notebook)

Needs only numpy and pandas.
"""
import sys
import numpy as np
import pandas as pd

src = sys.argv[1]
dst = sys.argv[2] if len(sys.argv) > 2 else "SST_v2_lowlat.npz"
LATMAX = 40.51

head = pd.read_csv(src, header=None, nrows=3)
r0 = pd.to_numeric(head.iloc[0, 2:], errors="coerce").values
time_in_columns = np.nanmin(r0) > 1990 and np.nanmax(r0) < 2030
print("layout:", "time in columns, pixels in rows" if time_in_columns else "time in rows, pixels in columns")


def to_centi(a):
    a = np.asarray(a, dtype=float)
    out = np.round(a * 100)
    out[~np.isfinite(out)] = -32768
    return out.astype(np.int16)


if time_in_columns:
    year = head.iloc[0, 2:].astype(int).values
    day = head.iloc[1, 2:].astype(int).values
    lats, lons, rws, blocks = [], [], [], []
    start = 0
    for k, ch in enumerate(pd.read_csv(src, header=None, skiprows=2, chunksize=2000)):
        la = ch.iloc[:, 0].astype(float).values
        keep = np.abs(la) <= LATMAX
        if keep.any():
            lats.append(la[keep]); lons.append(ch.iloc[:, 1].astype(float).values[keep])
            rws.append(start + np.where(keep)[0]); blocks.append(to_centi(ch.iloc[:, 2:].values[keep]))
        start += len(ch)
        print(f"\rchunk {k + 1}", end="")
    lat, lon, sst, row = np.concatenate(lats), np.concatenate(lons), np.vstack(blocks), np.concatenate(rws)
else:
    lat_all = pd.read_csv(src, header=None, nrows=1).iloc[0, 2:].astype(float).values
    lon_all = pd.read_csv(src, header=None, skiprows=1, nrows=1).iloc[0, 2:].astype(float).values
    keep = np.where(np.abs(lat_all) <= LATMAX)[0]
    cols = [0, 1] + list(keep + 2)
    body = pd.read_csv(src, header=None, skiprows=2, usecols=cols)
    year, day = body.iloc[:, 0].astype(int).values, body.iloc[:, 1].astype(int).values
    lat, lon, sst, row = lat_all[keep], lon_all[keep], to_centi(body.iloc[:, 2:].values.T), keep

np.savez_compressed(dst, year=year, day=day, lat=lat, lon=lon, row=row, sst_centiC=sst)   # row = line number in the csv
print(f"\nsaved {dst}: {sst.shape[0]} pixels x {sst.shape[1]} composites, {year[0]}/{day[0]} - {year[-1]}/{day[-1]}")
