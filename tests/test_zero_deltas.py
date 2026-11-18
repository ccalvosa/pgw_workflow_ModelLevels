#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Round-trip test: with ALL deltas set to zero, the PGW output must equal the
input up to GRIB repacking noise (16 bits).

Checks the internal consistency of the package (q -> RH -> q, ps adjustment,
vertical interpolation, GRIB writing).

Usage:
  python3 tests/test_zero_deltas.py <ifs_grib> <deltas_on_grid.nc> <tmp_dir>
"""
import os
import sys

import numpy as np
import xarray as xr

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import eccodes as ec                                          # noqa: E402
from pgw_ml.apply import PGWOptions, apply_pgw_to_file        # noqa: E402
from pgw_ml.deltas import Deltas                              # noqa: E402

src, deltas_nc, tmp = sys.argv[1:4]
os.makedirs(tmp, exist_ok=True)
ds = xr.open_dataset(deltas_nc).load()
for v in ds.data_vars:
    ds[v][:] = 0.0
zero_nc = os.path.join(tmp, "deltas_zero.nc")
ds.to_netcdf(zero_nc)

out = os.path.join(tmp, "zero_pgw.grb")
opt = PGWOptions(month="auto", perturb_wind=True)
diag = apply_pgw_to_file(src, out, Deltas(zero_nc), opt)
print("diagnostics:", {k: diag[k] for k in ("ps_iterations", "dps_mean", "dT_lowest_mean")})

worst = {}
with open(src, "rb") as fa, open(out, "rb") as fb:
    while True:
        ga, gb = ec.codes_grib_new_from_file(fa), ec.codes_grib_new_from_file(fb)
        if ga is None:
            break
        sn = ec.codes_get(ga, "shortName")
        va, vb = ec.codes_get_values(ga), ec.codes_get_values(gb)
        # tolerance: 1 packing unit (range / 2^bits)
        tol = (va.max() - va.min()) / 2 ** ec.codes_get(ga, "bitsPerValue") + 1e-12
        rel = np.abs(va - vb).max() / tol
        worst[sn] = max(worst.get(sn, 0.0), rel)
        ec.codes_release(ga)
        ec.codes_release(gb)

ok = True
for sn, r in sorted(worst.items()):
    flag = "OK" if r <= 1.0 else "FAIL"
    ok &= r <= 1.0
    print(f"  {sn:6s} max|Δ| = {r:.3f} packing units  [{flag}]")
print("TEST PASSED" if ok else "TEST FAILED")
sys.exit(0 if ok else 1)
