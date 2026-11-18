"""
pgw_ml: pseudo-global warming (PGW) on hybrid model levels for the IFS
initial and boundary conditions used by HARMONIE.

Modules
-------
constants : physical constants (IFS values).
thermo    : saturation vapor pressure (IFS mixed water/ice), RH <-> q, Tv.
vertical  : hybrid pressures, hydrostatic geopotential, ln(p) interpolation of
            the deltas with a surface anchor.
grids     : rotated-pole transform, GCM -> HARMONIE regridding, wind rotation.
gribio    : reading/writing of the IFS GRIB files (mixed GRIB1/GRIB2).
prepare   : deltas of one CMIP6 model -> HARMONIE grid.
ensemble  : scenarios, model discovery and selection, ensemble mean.
deltas    : access to the prepared deltas and temporal selection.
sst       : coastal cleaning of the SST delta (IDW from the open sea).
apply     : full PGW algorithm for one file (including the ps adjustment).
verify    : verification metrics for one original/PGW file pair.
"""
