"""
Quantitative verification that the deltas have been applied correctly.

For an (original, PGW) pair and the deltas of its member, the EXPECTED change
of each field is computed and compared with the change ACTUALLY written to
the GRIB file. Two families of checks:

 A) Exact replica (whole domain): the expected change is recomputed with the
    package formulas and the final ps read from the PGW file. The error must
    stay within the GRIB packing noise, expressed in "packing units"
    (p.u. = quantization step of the message, 2^E · 10^-D). OK if <= 1 p.u.
      T, q, skt, sst, stl1-4, (u, v)

 B) Physical and independent checks:
      * Δzg reached at p_ref versus the GCM target                    [m]
      * ΔRH reached (RH recomputed from the final T, q, p) versus the
        target ΔRH, in the troposphere and outside clipped points     [%]
      * stratospheric q unchanged (above q_taper_top)
      * ΔT of sample columns recomputed with np.interp in ln p
        (independent implementation of the vertical interpolation)    [K]
      * SST mask preserved
"""

from datetime import datetime

import numpy as np
import eccodes as ec

from .apply import PGWOptions
from .constants import G
from .gribio import IFSFile, PARAM, SOIL_DEPTH
from .sst import ocean_only_delta
from .thermo import q_to_rh, rh_to_q
from .vertical import (split_pv, full_level_pressure, geopotential_at_pressure,
                       interp_delta_to_model_levels)

# Tolerances of the checks
TOL_PACK = 1.0          # p.u.
TOL_DZ = 0.25           # m       error of Δzg at p_ref
TOL_DRH = 0.5           # %       RMSE of ΔRH at non-clipped points
TOL_STRAT_Q = 1.0       # p.u.    change of q in the stratosphere
TOL_INDEP_T = 1.0e-3    # K       independent ΔT replica


def _pack_unit(f, name, type_of_level, level):
    """Quantization step of the message: 2^E · 10^-D (binary and decimal
    scale factors of the packing). Falls back to range / 2^bits."""
    gid = f.handles[f.index[(PARAM[name], type_of_level, level)]]
    try:
        e = ec.codes_get(gid, "binaryScaleFactor")
        d = ec.codes_get(gid, "decimalScaleFactor")
        step = (2.0 ** e) * (10.0 ** -d)
        if np.isfinite(step) and step > 0:
            return step
    except Exception:
        pass
    bits = ec.codes_get(gid, "bitsPerValue")
    v = ec.codes_get_values(gid)
    if ec.codes_get(gid, "bitmapPresent"):
        v = v[v != ec.codes_get(gid, "missingValue")]
    rng = float(v.max() - v.min()) if v.size else 0.0
    return max(rng / (2.0 ** bits) if bits else 0.0, 1.0e-12)


def _pack_units_3d(f, name):
    return np.array([_pack_unit(f, name, "hybrid", lev) for lev in f.hybrid_levels(name)])


def _stats(err, unit=None, mask=None):
    e = err if mask is None else err[mask]
    if e.size == 0:
        return {"bias": 0.0, "rmse": 0.0, "maxabs": 0.0, "max_pu": 0.0, "n": 0}
    out = {"bias": float(e.mean()), "rmse": float(np.sqrt((e ** 2).mean())),
           "maxabs": float(np.abs(e).max()), "n": int(e.size)}
    if unit is not None:
        u = unit if mask is None else np.broadcast_to(unit, err.shape)[mask]
        out["max_pu"] = float((np.abs(e) / u).max())
    return out


def _indep_dT(d_ta, plev, pf0, d_tas, ps0, depth, cols):
    """Expected ΔT on columns 'cols' with np.interp (independent of vertical.py)."""
    order = np.argsort(plev)
    lp = np.log(plev[order])
    out = np.empty((pf0.shape[0], cols.size))
    for j, c in enumerate(cols):
        prof = d_ta[order, c]
        dstd = np.interp(np.log(pf0[:, c]), lp, prof)        # constant extrapolation
        if d_tas is not None:
            d_ps = np.interp(np.log(ps0[c]), lp, prof)
            w = np.clip(1.0 - (ps0[c] - pf0[:, c]) / depth, 0.0, 1.0)
            dstd = dstd + (d_tas[c] - d_ps) * w
        out[:, j] = dstd
    return out


def verify_file(src, dst, deltas, opt: PGWOptions, n_cols=300, seed=0):
    """Returns (metrics, extras). metrics: {name: {stats..., ok}}."""
    fa, fb = IFSFile(src), IFSFile(dst)
    try:
        return _verify(fa, fb, deltas, opt, n_cols, seed)
    finally:
        fa.close()
        fb.close()


def _verify(fa, fb, deltas, opt, n_cols, seed):
    vdate, vtime = fa.valid_datetime()
    valid = datetime.strptime(f"{vdate}{vtime:04d}", "%Y%m%d%H%M")
    d = deltas.at(valid, opt.month)
    plev = deltas.plev
    gi = fa.grid_info()
    nj, ni = gi["Nj"], gi["Ni"]
    M = {}

    a, b = split_pv(fa.pv())
    ps0 = np.exp(fa.get2d("lnsp", "hybrid", 1)[0])
    ps1 = np.exp(fb.get2d("lnsp", "hybrid", 1)[0])
    phis = fa.get2d("z", "hybrid", 1)[0]
    t0, t1 = fa.get3d("t"), fb.get3d("t")
    q0, q1 = fa.get3d("q"), fb.get3d("q")
    pf0, pf1 = full_level_pressure(a, b, ps0), full_level_pressure(a, b, ps1)
    anchor, bd = opt.sfc_anchor, opt.sfc_blend_depth

    # ── T ────────────────────────────────────────────────────────────────────
    dta = interp_delta_to_model_levels(d["ta"], plev, pf0,
                                       d["tas"] if anchor else None, ps0, bd)
    dT_exp = dta if opt.perturb_t else np.zeros_like(t0)
    uT = _pack_units_3d(fb, "t")[:, None]
    M["T (replica)"] = _stats((t1 - t0) - dT_exp, uT)
    M["T (replica)"]["ok"] = M["T (replica)"]["max_pu"] <= TOL_PACK

    rng = np.random.default_rng(seed)
    cols = np.sort(rng.choice(t0.shape[1], size=min(n_cols, t0.shape[1]), replace=False))
    if opt.perturb_t:
        ind = _indep_dT(d["ta"], plev, pf0, d["tas"] if anchor else None, ps0, bd, cols)
        e = (t1 - t0)[:, cols] - ind
        k = "ΔT columns (np.interp) [K]"
        M[k] = _stats(e)
        M[k]["ok"] = M[k]["maxabs"] <= TOL_INDEP_T + 2 * uT.max()

    # ── q / RH ───────────────────────────────────────────────────────────────
    rh0 = q_to_rh(q0, t0, pf0)
    rh1 = q_to_rh(q1, t1, pf1)
    dhur = interp_delta_to_model_levels(d["hur"], plev, pf0,
                                        d["hurs"] if anchor else None, ps0, bd) / 100.0
    uq = _pack_units_3d(fb, "q")[:, None]
    if opt.perturb_q:
        lo, hi = opt.rh_min_factor * rh0, np.maximum(rh0, 1.0)
        rh_tgt = rh0 + dhur
        rh_pgw = np.clip(rh_tgt, lo, hi)
        clipped = rh_pgw != rh_tgt
        w_q = np.clip((np.log(pf0) - np.log(opt.q_taper_top)) /
                      (np.log(opt.q_taper_bottom) - np.log(opt.q_taper_top)), 0.0, 1.0)
        q_exp = q0 + w_q * (rh_to_q(rh_pgw, t0 + dT_exp, pf1) - q0)
        M["q (replica)"] = _stats(q1 - q_exp, uq)
        M["q (replica)"]["ok"] = M["q (replica)"]["max_pu"] <= TOL_PACK
        tropo = (pf0 >= opt.q_taper_bottom) & ~clipped
        k = "ΔRH troposphere [%]"
        M[k] = _stats(100 * ((rh1 - rh0) - dhur), mask=tropo)
        M[k]["ok"] = M[k]["rmse"] <= TOL_DRH
        M[k]["frac_clipped"] = float(clipped[pf0 >= opt.q_taper_bottom].mean())
        strat = pf0 <= opt.q_taper_top
        k = "q stratosphere unchanged"
        M[k] = _stats(q1 - q0, uq, mask=strat)
        M[k]["ok"] = M[k]["max_pu"] <= TOL_STRAT_Q
    else:
        M["q (unchanged)"] = _stats(q1 - q0, uq)
        M["q (unchanged)"]["ok"] = M["q (unchanged)"]["max_pu"] <= TOL_PACK
    M["q >= 0"] = {"min": float(q1.min()), "ok": bool(q1.min() >= 0)}

    # ── ps / Δzg ─────────────────────────────────────────────────────────────
    dz_err = None
    if opt.ps_mode == "hydrostatic":
        order = np.argsort(plev)
        lp = np.log(plev[order])
        k = int(np.clip(np.searchsorted(lp, np.log(opt.p_ref)), 1, lp.size - 1))
        w = (np.log(opt.p_ref) - lp[k - 1]) / (lp[k] - lp[k - 1])
        dzg_tgt = d["zg"][order][k - 1] * (1 - w) + d["zg"][order][k] * w
        phi0 = geopotential_at_pressure(a, b, ps0, t0, q0, phis, opt.p_ref)
        phi1 = geopotential_at_pressure(a, b, ps1, t1, q1, phis, opt.p_ref)
        dz_err = (phi1 - phi0) / G - dzg_tgt
        k = f"Δzg({opt.p_ref / 100:.0f} hPa) [m]"
        M[k] = _stats(dz_err)
        M[k]["ok"] = float(np.abs(dz_err).max()) <= TOL_DZ
    elif opt.ps_mode == "gcm":
        M["Δps vs GCM [Pa]"] = _stats((ps1 - ps0) - d["ps"])
        M["Δps vs GCM [Pa]"]["ok"] = M["Δps vs GCM [Pa]"]["maxabs"] <= 2.0
    else:
        M["ps unchanged [Pa]"] = _stats(ps1 - ps0)
        M["ps unchanged [Pa]"]["ok"] = M["ps unchanged [Pa]"]["maxabs"] <= 2.0

    # ── Surface ──────────────────────────────────────────────────────────────
    dts = d["ts"]
    lsm = fa.get2d("lsm")[0]
    sea = lsm < opt.lsm_threshold
    dts_sea, _ = ocean_only_delta(dts.reshape(nj, ni), lsm.reshape(nj, ni), opt.lsm_threshold,
                                  opt.coast_buffer, opt.k_neighbors, opt.idw_power,
                                  opt.sst_smooth_sigma)
    dts_comb = np.where(sea, dts_sea.ravel(), dts)
    skt0, skt1 = fa.get2d("skt")[0], fb.get2d("skt")[0]
    e = (skt1 - skt0) - (dts_comb if opt.perturb_skt else 0.0)
    M["skt (replica)"] = _stats(e, _pack_unit(fb, "skt", "surface", 0))
    M["skt (replica)"]["ok"] = M["skt (replica)"]["max_pu"] <= TOL_PACK
    dsst = None
    if fa.has("sst", "surface", 0):
        (sst0, m0), (sst1, m1) = fa.get2d("sst"), fb.get2d("sst")
        M["SST mask preserved"] = {"ok": bool(np.array_equal(m0, m1))}
        e = (sst1 - sst0) - (dts_comb if opt.perturb_sst else 0.0)
        M["sst (replica)"] = _stats(e, _pack_unit(fb, "sst", "surface", 0), mask=~m0)
        M["sst (replica)"]["ok"] = M["sst (replica)"]["max_pu"] <= TOL_PACK
        dsst = np.where(m0, np.nan, sst1 - sst0)
    dts_ann = deltas.annual_mean("ts")
    for name, depth in SOIL_DEPTH.items():
        levs = [lev for (p, tl, lev) in fa.index if p == PARAM[name] and tl == "depthBelowLandLayer"]
        if not levs:
            continue
        s0 = fa.get2d(name, "depthBelowLandLayer", levs[0])[0]
        s1 = fb.get2d(name, "depthBelowLandLayer", levs[0])[0]
        exp = dts_ann + np.exp(-depth / opt.soil_efold) * (dts - dts_ann)
        e = (s1 - s0) - (exp if opt.perturb_soil else 0.0)
        k = f"{name} (replica)"
        M[k] = _stats(e, _pack_unit(fb, name, "depthBelowLandLayer", levs[0]))
        M[k]["ok"] = M[k]["max_pu"] <= TOL_PACK

    # ── Wind ─────────────────────────────────────────────────────────────────
    if opt.perturb_wind:
        for c, dv in (("u", "ua"), ("v", "va")):
            exp = interp_delta_to_model_levels(d[dv], plev, pf0)
            e = (fb.get3d(c) - fa.get3d(c)) - exp
            k = f"{c} (replica)"
            M[k] = _stats(e, _pack_units_3d(fb, c)[:, None])
            M[k]["ok"] = M[k]["max_pu"] <= TOL_PACK

    # ── Extras for figures and the intermember comparison ────────────────────
    extras = {
        "valid": valid.isoformat(),
        "shape": (nj, ni),
        "p_hPa": pf0.mean(axis=1) / 100.0,
        "prof_dT": (t1 - t0).mean(axis=1),
        "prof_dT_exp": dT_exp.mean(axis=1),
        "prof_dRH": (100 * (rh1 - rh0)).mean(axis=1),
        "prof_dRH_tgt": (100 * dhur).mean(axis=1),
        "prof_dq_rel": (100 * (q1 - q0) / np.maximum(q0, 1e-12)).mean(axis=1),
        "dT3d": (t1 - t0).astype("float32"),
        "dRH3d": (100 * (rh1 - rh0)).astype("float32"),
        "map_dps": ps1 - ps0,
        "map_dT_low": t1[-1] - t0[-1],
        "map_dT_low_err": (t1[-1] - t0[-1]) - dT_exp[-1],
        "map_dsst": dsst,
        "map_dz_err": dz_err,
        "scatter_T": (dT_exp[:, cols].ravel(), (t1 - t0)[:, cols].ravel()),
        "means": {"dps": float((ps1 - ps0).mean()),
                  "dT_low": float((t1 - t0)[-1].mean()),
                  "dT850": float(_near(t1 - t0, pf0, 85000).mean()),
                  "dT500": float(_near(t1 - t0, pf0, 50000).mean()),
                  "dRH850": float(_near(100 * (rh1 - rh0), pf0, 85000).mean()),
                  "dq_rel850": float(_near(100 * (q1 - q0) / np.maximum(q0, 1e-12),
                                           pf0, 85000).mean()),
                  "dsst": float(np.nanmean(dsst)) if dsst is not None else float("nan"),
                  "dskt_land": float((skt1 - skt0)[~sea].mean()) if (~sea).any() else float("nan")},
    }
    return M, extras


def _near(field, pf, p):
    k = np.abs(pf - p).argmin(axis=0)
    return field[k, np.arange(field.shape[1])]
