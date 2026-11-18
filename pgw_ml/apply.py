"""
PGW applied to one IFS file on model levels.

Algorithm (adaptation of step_03_apply_to_era.py of PGW4ERA5, Brogli et al.
2023, https://github.com/Potopoles/PGW4ERA5, with i_reinterp = 0):

 1. Full-level p with the original ps (p = a + b·ps).
 2. Original RH = f(q, T, p)   (IFS mixed water/ice saturation).
 3. Deltas ta, hur (and ua, va) interpolated in ln(p) to the ORIGINAL
    pressures of each level. For ta/hur a continuous surface correction
    toward tas/hurs is added over the lowest 'sfc_blend_depth' Pa
    (anchored at the IFS ps; see README).
 4. T_pgw = T + ΔT ;  RH_pgw = RH + ΔRH ;  u,v_pgw = u,v + Δu,v (rotated).
    RH_pgw is bounded to [rh_min_factor·RH, max(RH, 1)] and the change in q
    is tapered in ln(p) between q_taper_bottom and q_taper_top (q unchanged
    in the stratosphere).
 5. Iterative ps adjustment: ps_pgw is sought such that
        Φ_pgw(p_ref) - Φ_orig(p_ref) = g · Δzg(p_ref)
    recomputing q_pgw = f(RH_pgw, T_pgw, p(ps_pgw)) at each iteration.
        ps += -adj · ps / (Rd · T_pgw,lowest level) · error_Φ
 6. lnsp_pgw = ln(ps_pgw). The surface geopotential is unchanged.
 7. Surface: skt + Δts ("open-sea" Δts over the sea), sst + coast-cleaned
    Δts over the sea, stl1-4 + Δts damped with depth:
        Δst(z) = Δts_annual + exp(-z/2.8 m) · (Δts - Δts_annual)
"""

from dataclasses import dataclass, asdict
from datetime import datetime

import numpy as np

from .constants import G, RD
from .deltas import Deltas
from .gribio import IFSFile, PARAM, SOIL_DEPTH
from .sst import ocean_only_delta
from .thermo import q_to_rh, rh_to_q
from .vertical import (split_pv, full_level_pressure, geopotential_at_pressure,
                       interp_delta_to_model_levels)


@dataclass
class PGWOptions:
    month: str = "auto"              # 1..12 or 'auto' (temporal interpolation)
    perturb_t: bool = True
    perturb_q: bool = True           # keeping RH + ΔRH
    q_taper_bottom: float = 15000.0  # Pa: below (higher p) full change in q
    q_taper_top: float = 10000.0     # Pa: above (lower p) q is not modified
    rh_min_factor: float = 0.5       # RH_pgw >= rh_min_factor * RH_orig
    perturb_wind: bool = False
    ps_mode: str = "hydrostatic"     # hydrostatic | gcm | none
    p_ref: float = 30000.0           # Pa (PGW4ERA5: p_ref_inp = 30000)
    adj_factor: float = 0.95
    thresh_phi: float = 0.15         # m2 s-2
    max_iter: int = 20
    sfc_anchor: bool = True          # anchor ta/hur to tas/hurs at the surface
    sfc_blend_depth: float = 10000.0 # Pa: depth of the surface transition layer
    perturb_skt: bool = True
    perturb_sst: bool = True
    perturb_soil: bool = True
    soil_efold: float = 2.8          # m
    lsm_threshold: float = 0.5
    coast_buffer: int = 2
    k_neighbors: int = 8
    idw_power: float = 2.0
    sst_smooth_sigma: float = 1.0


def _interp_p_ref(delta_pl, plev, p_ref):
    """Delta at p_ref (ln p interpolation) for each column."""
    order = np.argsort(plev)
    lp = np.log(plev[order])
    d = delta_pl[order]
    k = int(np.clip(np.searchsorted(lp, np.log(p_ref)), 1, lp.size - 1))
    w = (np.log(p_ref) - lp[k - 1]) / (lp[k] - lp[k - 1])
    return d[k - 1] * (1 - w) + d[k] * w


def apply_pgw_to_file(path_in, path_out, deltas: Deltas, opt: PGWOptions, log=print):
    f = IFSFile(path_in)
    try:
        return _apply(f, path_out, deltas, opt, log)
    finally:
        f.close()


def _apply(f, path_out, deltas, opt, log):
    diag = {"file": f.path}
    vdate, vtime = f.valid_datetime()
    valid_dt = datetime.strptime(f"{vdate}{vtime:04d}", "%Y%m%d%H%M")
    d = deltas.at(valid_dt, opt.month)
    diag["valid"] = valid_dt.isoformat()
    diag["month_weights"] = {int(k) + 1: round(float(v), 4) for k, v in d["_weights"].items()}
    plev = deltas.plev

    # ── Original state ───────────────────────────────────────────────────────
    a, b = split_pv(f.pv())
    lnsp, _ = f.get2d("lnsp", "hybrid", 1)
    ps0 = np.exp(lnsp)
    phis, _ = f.get2d("z", "hybrid", 1)
    t0 = f.get3d("t")
    q0 = f.get3d("q")
    pf0 = full_level_pressure(a, b, ps0)
    rh0 = q_to_rh(q0, t0, pf0)

    if pf0.min() < plev.min():
        log(f"  [INFO] Model top ({pf0.min():.2f} Pa) above the highest GCM level "
            f"({plev.min():.0f} Pa): constant extrapolation of the delta.")

    # ── Deltas on model levels (original pressures) ──────────────────────────
    anchor = opt.sfc_anchor
    bd = opt.sfc_blend_depth
    dta = interp_delta_to_model_levels(d["ta"], plev, pf0,
                                       d["tas"] if anchor else None, ps0, bd)
    dhur = interp_delta_to_model_levels(d["hur"], plev, pf0,
                                        d["hurs"] if anchor else None, ps0, bd) / 100.0

    t_pgw = t0 + dta if opt.perturb_t else t0.copy()
    if opt.perturb_q:
        # Relative floor (avoids RH <= 0 in very dry areas) and ceiling at
        # saturation (or at the original supersaturation, which the IFS
        # allows with respect to ice).
        rh_pgw = np.clip(rh0 + dhur, opt.rh_min_factor * rh0, np.maximum(rh0, 1.0))
        # Vertical weight of the change in q: 1 in the troposphere, 0 in the
        # stratosphere (stratospheric vapor is not controlled by RH, and
        # keeping RH with the GCM ΔT produces unrealistic changes in q).
        lw = (np.log(pf0) - np.log(opt.q_taper_top)) / \
             (np.log(opt.q_taper_bottom) - np.log(opt.q_taper_top))
        w_q = np.clip(lw, 0.0, 1.0)
    else:
        rh_pgw = None

    def q_for(ps):
        if rh_pgw is None:
            return q0
        q_rh = rh_to_q(rh_pgw, t_pgw, full_level_pressure(a, b, ps))
        return q0 + w_q * (q_rh - q0)

    # ── Surface pressure ─────────────────────────────────────────────────────
    ps = ps0.copy()
    if opt.ps_mode == "gcm":
        ps = ps0 + d["ps"]
        diag["ps_iterations"] = 0
    elif opt.ps_mode == "hydrostatic":
        phi_ref0 = geopotential_at_pressure(a, b, ps0, t0, q0, phis, opt.p_ref)
        dphi_clim = G * _interp_p_ref(d["zg"], plev, opt.p_ref)
        err_max = np.inf
        it = 0
        while err_max > opt.thresh_phi:
            it += 1
            if it > opt.max_iter:
                raise RuntimeError(f"The ps adjustment does not converge ({err_max:.3f} m2/s2 "
                                   f"after {opt.max_iter} iterations)")
            q_it = q_for(ps)
            phi = geopotential_at_pressure(a, b, ps, t_pgw, q_it, phis, opt.p_ref)
            err = (phi - phi_ref0) - dphi_clim
            err_max = float(np.abs(err).max())
            if err_max > opt.thresh_phi:
                ps = ps - opt.adj_factor * ps / (RD * t_pgw[-1]) * err
        diag["ps_iterations"] = it
        diag["phi_err_max"] = err_max
    elif opt.ps_mode != "none":
        raise ValueError(f"Unknown ps_mode: {opt.ps_mode}")

    q_pgw = q_for(ps)

    # ── 3D output ────────────────────────────────────────────────────────────
    if opt.perturb_t:
        f.set3d("t", t_pgw)
    if opt.perturb_q:
        f.set3d("q", q_pgw)
    if opt.ps_mode != "none":
        f.set2d("lnsp", np.log(ps), "hybrid", 1)
    if opt.perturb_wind:
        if not deltas.winds_rotated and f.grid_info()["uv_relative_to_grid"]:
            raise RuntimeError("Grid-relative winds but the deltas are not rotated")
        du = interp_delta_to_model_levels(d["ua"], plev, pf0)
        dv = interp_delta_to_model_levels(d["va"], plev, pf0)
        f.set3d("u", f.get3d("u") + du)
        f.set3d("v", f.get3d("v") + dv)
        diag["du_mean_lowest"] = float(du[-1].mean())

    # ── Surface ──────────────────────────────────────────────────────────────
    gi = f.grid_info()
    nj, ni = gi["Nj"], gi["Ni"]
    dts = d["ts"]
    lsm, _ = f.get2d("lsm")
    sea = lsm < opt.lsm_threshold
    dts_sea, n_clean = ocean_only_delta(
        dts.reshape(nj, ni), lsm.reshape(nj, ni), opt.lsm_threshold,
        opt.coast_buffer, opt.k_neighbors, opt.idw_power, opt.sst_smooth_sigma)
    dts_sea = dts_sea.ravel()
    dts_comb = np.where(sea, dts_sea, dts)

    if opt.perturb_skt:
        skt, _ = f.get2d("skt")
        f.set2d("skt", skt + dts_comb)
    if opt.perturb_sst and f.has("sst", "surface", 0):
        sst, miss = f.get2d("sst")
        f.set2d("sst", np.where(miss, sst, sst + dts_comb), missing=miss)
    if opt.perturb_soil:
        dts_ann = deltas.annual_mean("ts")
        for name, depth in SOIL_DEPTH.items():
            lev = _soil_level(f, name)
            if lev >= 0:
                st, _ = f.get2d(name, "depthBelowLandLayer", lev)
                dst = dts_ann + np.exp(-depth / opt.soil_efold) * (dts - dts_ann)
                f.set2d(name, st + dst, "depthBelowLandLayer", lev)

    f.write(path_out)

    # ── Diagnostics ──────────────────────────────────────────────────────────
    diag.update({
        "dps_mean": float((ps - ps0).mean()),
        "dps_min": float((ps - ps0).min()),
        "dps_max": float((ps - ps0).max()),
        "dT_lowest_mean": float((t_pgw - t0)[-1].mean()),
        "dT_850_mean": float(_mean_near(t_pgw - t0, pf0, 85000.0)),
        "dT_500_mean": float(_mean_near(t_pgw - t0, pf0, 50000.0)),
        "dq_lowest_rel_mean": float(((q_pgw - q0)[-1] / q0[-1]).mean()),
        "dts_sea_mean": float(dts_sea[sea].mean()) if sea.any() else float("nan"),
        "dts_land_mean": float(dts[~sea].mean()) if (~sea).any() else float("nan"),
        "n_open_sea_points": n_clean,
    })
    return diag


def _soil_level(f, name):
    """GRIB level of soil layer 'name' (-1 if not present in the file)."""
    levs = [lev for (p, tl, lev) in f.index if p == PARAM[name] and tl == "depthBelowLandLayer"]
    return levs[0] if levs else -1


def _mean_near(field, pf, p):
    """Domain mean of the field on the model level closest to p."""
    k = np.abs(pf - p).argmin(axis=0)
    return field[k, np.arange(field.shape[1])].mean()


def options_dict(opt):
    return asdict(opt)
