#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
STEP 03: integrity check of the PGW files against the originals (one member).

For each (original, PGW) pair it checks that:
  * the number of messages and the metadata (paramId, level, date, step,
    grid, packing) are identical and in the same order;
  * NON-perturbed fields are bit-identical in their values;
  * there are no non-finite values, q >= 0 and the SST mask is preserved.

It prints a summary and, if matplotlib is available, one PNG per file
(profiles + maps of Δps, ΔT of the lowest level and ΔSST).

Usage:
  python3 step03_check_pgw.py <input_root> <member_output_root> [--plots N]
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eccodes as ec                                                    # noqa: E402
from pgw_ml.gribio import IFSFile                                       # noqa: E402
from pgw_ml.thermo import q_to_rh                                       # noqa: E402
from pgw_ml.vertical import split_pv, full_level_pressure               # noqa: E402

META_KEYS = ("paramId", "typeOfLevel", "level", "dataDate", "dataTime", "stepRange",
             "gridType", "Ni", "Nj", "packingType", "bitmapPresent", "editionNumber")
PERTURBED = {130, 133, 152, 235, 34, 139, 170, 183, 236}   # t q lnsp skt sst stl1-4
WIND = {131, 132}


def compare(src, dst, wind_perturbed):
    fa, fb = IFSFile(src), IFSFile(dst)
    rep = {"file": dst, "problems": []}
    try:
        if len(fa.handles) != len(fb.handles):
            rep["problems"].append(f"number of messages {len(fa.handles)} != {len(fb.handles)}")
            return rep, None, None
        changed = set()
        for ga, gb in zip(fa.handles, fb.handles):
            ma = [ec.codes_get(ga, k) for k in META_KEYS]
            mb = [ec.codes_get(gb, k) for k in META_KEYS]
            if ma != mb:
                rep["problems"].append(f"different metadata: {ma} vs {mb}")
                continue
            va, vb = ec.codes_get_values(ga), ec.codes_get_values(gb)
            if not np.all(np.isfinite(vb)):
                rep["problems"].append(f"non-finite values in paramId={ma[0]} level={ma[2]}")
            pid = ma[0]
            expected = pid in PERTURBED or (wind_perturbed and pid in WIND)
            if not np.array_equal(va, vb):
                changed.add(pid)
                if not expected:
                    rep["problems"].append(f"paramId={pid} level={ma[2]} changed unexpectedly")
        rep["changed_params"] = sorted(changed)

        # Physics
        a, b = split_pv(fa.pv())
        ps0 = np.exp(fa.get2d("lnsp", "hybrid", 1)[0])
        ps1 = np.exp(fb.get2d("lnsp", "hybrid", 1)[0])
        t0, t1 = fa.get3d("t"), fb.get3d("t")
        q0, q1 = fa.get3d("q"), fb.get3d("q")
        p0, p1 = full_level_pressure(a, b, ps0), full_level_pressure(a, b, ps1)
        rh0, rh1 = q_to_rh(q0, t0, p0), q_to_rh(q1, t1, p1)
        if (q1 < 0).any():
            rep["problems"].append("negative q")
        sst0, m0 = fa.get2d("sst")
        sst1, m1 = fb.get2d("sst")
        if not np.array_equal(m0, m1):
            rep["problems"].append("SST mask altered")
        rep.update({
            "dps_mean": float((ps1 - ps0).mean()),
            "dsst_mean": float((sst1 - sst0)[~m0].mean()),
            "q_min": float(q1.min()),
            "rh_max_pgw": float(rh1.max()),
        })
        prof = {
            "p_hPa": (p0.mean(axis=1) / 100).round(3).tolist(),
            "dT": (t1 - t0).mean(axis=1).round(4).tolist(),
            "dRH_pct": (100 * (rh1 - rh0)).mean(axis=1).round(4).tolist(),
            "dq_rel_pct": (100 * (q1 - q0) / np.maximum(q0, 1e-12)).mean(axis=1).round(4).tolist(),
        }
        maps = {"dps": ps1 - ps0, "dT_lowest": t1[-1] - t0[-1],
                "dsst": np.where(m0, np.nan, sst1 - sst0),
                "shape": (fa.grid_info()["Nj"], fa.grid_info()["Ni"])}
        return rep, prof, maps
    finally:
        fa.close()
        fb.close()


def plot(prof, maps, title, png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 3, figsize=(15, 9))
    p = np.array(prof["p_hPa"])
    for i, (k, lab) in enumerate([("dT", "ΔT [K]"), ("dRH_pct", "ΔRH [%]"),
                                  ("dq_rel_pct", "Δq/q [%]")]):
        ax[0, i].plot(prof[k], p, ".-", ms=3)
        ax[0, i].set_yscale("log")
        ax[0, i].set_ylim(1050, 1)
        ax[0, i].axvline(0, color="k", lw=0.5)
        ax[0, i].set_xlabel(lab)
        ax[0, i].set_ylabel("mean level pressure [hPa]")
        ax[0, i].grid(alpha=0.3)
    nj, ni = maps["shape"]
    for i, (k, lab, cm) in enumerate([("dps", "Δps [Pa]", "RdBu_r"),
                                      ("dT_lowest", "ΔT level 137 [K]", "RdBu_r"),
                                      ("dsst", "ΔSST [K]", "RdBu_r")]):
        f2 = maps[k].reshape(nj, ni)
        lim = np.nanmax(np.abs(f2))
        im = ax[1, i].imshow(f2, origin="upper", cmap=cm, vmin=-lim, vmax=lim)
        ax[1, i].set_title(lab)
        fig.colorbar(im, ax=ax[1, i], shrink=0.8)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(png, dpi=90)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input_root")
    ap.add_argument("output_root")
    ap.add_argument("--plots", type=int, default=1, help="number of files to plot")
    args = ap.parse_args()

    opts = json.load(open(os.path.join(args.output_root, "pgw_options.json")))["options"]
    diags = [json.loads(l) for l in open(os.path.join(args.output_root, "pgw_diagnostics.jsonl"))]
    n_bad = 0
    for i, dg in enumerate(d for d in diags if d["status"] == "ok"):
        src, dst = dg["file"], dg["output"]
        rep, prof, maps = compare(src, dst, opts["perturb_wind"])
        rel = os.path.relpath(dst, args.output_root)
        if prof is None:
            n_bad += 1
            print(f"[BAD] {rel}: {rep['problems']}")
            continue
        status = "OK " if not rep["problems"] else "BAD"
        n_bad += bool(rep["problems"])
        print(f"[{status}] {rel}: changed paramId={rep['changed_params']}  "
              f"Δps={rep['dps_mean']:+.1f} Pa  ΔSST={rep['dsst_mean']:+.2f} K  "
              f"q_min={rep['q_min']:.2e}  RHmax={rep['rh_max_pgw']:.3f}")
        for pr in rep["problems"][:10]:
            print("      -", pr)
        if i < args.plots:
            png = os.path.join(args.output_root, "check_" + rel.replace("/", "_") + ".png")
            try:
                plot(prof, maps, rel, png)
                print(f"      figure: {png}")
            except ImportError:
                print("      (no matplotlib: no figures)")
            with open(os.path.join(args.output_root, "check_profile_" +
                                   rel.replace("/", "_") + ".json"), "w") as fh:
                json.dump(prof, fh)
    if n_bad:
        sys.exit(f"ERROR: {n_bad} files with problems")
    print(">> Integrity check passed.")


if __name__ == "__main__":
    main()
