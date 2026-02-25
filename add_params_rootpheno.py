import xarray as xr
import numpy as np
import os
from utils.paths import *

path_parameter = os.path.join(os.environ["PROJDIR"], "E3SM", "inputdata", "atm", "datm7",
                              "CLM1PT_data", "SPRUCE_data")

#orgfile = 'clm_params_SPRUCE_20231120_spruceroot_eca.nc'
#newfile = 'clm_params_SPRUCE_20231120_spruceroot_eca.nc_rootpheno'
#orgfile = 'clm_params_SPRUCE_20231120_spruceroot.nc_CNP'
#newfile = 'clm_params_SPRUCE_20231120_spruceroot.nc_rootpheno'
orgfile = 'clm_params_SPRUCE_20231120_spruceroot.nc_npcompet'
newfile = 'clm_params_SPRUCE_20231120_spruceroot.nc_rootpheno_npcompet'


hr = xr.open_dataset(os.path.join(path_parameter, orgfile), decode_times=False)

# leaf-root ratio of tamarack has to be edited to achive growth
hr['froot_leaf'][3] = 0.2
##hr['froot_leaf'][11] = 0.15

### reduce the root N to boreal PFT levels in Ben's paper (2019) and should reduce the MR
##hr['frootcn'][2] = 45
##hr['frootcn'][3] = 45
##hr['frootcn'][11] = 45

# The longevity of black spruce and tamarack both appear rather low
# Ruess, R.W., Hendrick, R.L., Burton, A.J., Pregitzer, K.S., Sveinbjornssön, B., Allen, M.F. and Maurer, G.E. (2003), COUPLING FINE ROOT DYNAMICS WITH ECOSYSTEM CARBON CYCLING IN BLACK SPRUCE FORESTS OF INTERIOR ALASKA. Ecological Monographs, 73: 643-662. https://doi.org/10.1890/02-4032
hr["froot_long"][3] = 1.5
hr['froot_long'][11] = 1.5 # from MWM model, Siya Shao 2022 New Phytologist

hr["fcur"][2] = 0.2
hr["fcur"][3] = 0.0
hr["fcur"][11] = 0.0

hr["fcur_root"] = xr.DataArray(
    [
      np.nan, np.nan, 0.5, 0.8, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 0.8, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,
    ], 
    coords={"pft": hr["pft"]},
    dims=["pft"],
    attrs={
        "units": "",
        "long_name": "Fraction of fine root new biomass allocated to display instead of storage",
    },
)

hr["uf_scale"] = -0.0441
hr["uf_tbase"] = xr.DataArray(
    [6.848 + 273.15],
    dims=["allpft"],
    attrs={
        "units": "K",
        "long_name": "Base temperature in uniforc model",
    },
)
hr["crit_onset_uf"] = 66.943
hr["ndays_on"] = xr.DataArray(
    [
        np.nan, np.nan, 30, 30, np.nan, np.nan, 30, 30, 30, np.nan, 30, 30, np.nan, 30, 30, 30, 30, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,
    ],
    coords={"pft": hr["pft"]},
    dims=["pft"],
    attrs={"units": "days", "long_name": "Number of days to complete leaf onset"},
)

hr["gdd_tbase"] = xr.DataArray(
    [
        np.nan, np.nan, np.nan, 279.5, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 279.05, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,
    ],
    coords={"pft": hr["pft"]},
    dims=["pft"],
    attrs={"units": "K", "long_name": "Base temperature for GDD accumulation"},
)

hr["crit_chil1"] = xr.DataArray(
    [
        np.nan, np.nan, np.nan, 9.0, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 33, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,
    ],
    coords={"pft": hr["pft"]},
    dims=["pft"],
    attrs={"units": "", "long_name": "Parameter in alternating model"},
)

hr["crit_chil2"] = xr.DataArray(
    [
        np.nan, np.nan, np.nan, 2112, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 1388, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,
    ],
    coords={"pft": hr["pft"]},
    dims=["pft"],
    attrs={"units": "", "long_name": "Parameter in alternating model"},
)

hr["crit_chil3"] = xr.DataArray(
    [
        np.nan, np.nan, np.nan, -0.04, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, -0.02, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,
    ],
    coords={"pft": hr["pft"]},
    dims=["pft"],
    attrs={"units": "", "long_name": "Parameter in alternating model"},
)

hr["maxday_off"] = xr.DataArray(
    [286],
    coords={"allpfts": hr["allpfts"]},
    dims=["allpfts"],
    attrs={"units": "days", "long_name": "Day of maximum litterfall"},
)

hr["ndays_off"] = xr.DataArray(
    [
        np.nan, np.nan, 48, 15, np.nan, np.nan, 15, 15, 15, np.nan, 15, 15, np.nan, 15, 15, 15, 15, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,
    ],
    coords={"pft": hr["pft"]},
    dims=["pft"],
    attrs={
        "units": "days", "long_name": "Number of days to complete leaf litterfall (for evergreen this is only the relevant parameter not exact)",
    },
)

hr["crit_dayl"] = xr.DataArray(
    [
        np.nan, np.nan, np.nan, 39300, np.nan, np.nan, 36000, 39300, 39300, np.nan, 36000, 39300, np.nan, 36000, 36000, 36000, 36000, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,
    ],
    coords={"pft": hr["pft"]},
    dims=["pft"],
    attrs={"units": "seconds", "long_name": "Critical day length for senescence"},
)

hr["off_pstart"] = xr.DataArray(
    [
        np.nan, np.nan, np.nan, 46800.0, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 54600.0, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,
    ],
    coords={"pft": hr["pft"]},
    dims=["pft"],
    attrs={"units": "", "long_name": "Paramter for deciduous offset"},
)

hr["off_pend"] = xr.DataArray(
    [
        np.nan, np.nan, np.nan, 1750.0, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 1600.0, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,
    ],
    coords={"pft": hr["pft"]},
    dims=["pft"],
    attrs={"units": "", "long_name": "Paramter for deciduous offset"},
)

hr["off_tbase"] = xr.DataArray(
    [
        np.nan, np.nan, np.nan, 294.5, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 290.15, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,
    ],
    coords={"pft": hr["pft"]},
    dims=["pft"],
    attrs={"units": "", "long_name": "Paramter for deciduous offset"},
)

hr["crit_onset_uf_root"] = 42

hr["crit_chil2_root"] = xr.DataArray(
    [
        np.nan, np.nan, np.nan, 1200.0, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 400.0, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,
    ],
    coords={"pft": hr["pft"]},
    dims=["pft"],
    attrs={
        "units": "K",
        "long_name": "Scale for the critical onset gdd of the alternating model",
    },
)

hr["nmin_scale"] = xr.DataArray(
    [1],
    dims=["allpft"],
    attrs={
        "units": "",
        "long_name": "Parameter in soil nutrient control on root onset growth",
    },
)

hr["wt_scale"] = xr.DataArray(
    [100],
    dims=["allpft"],
    attrs={
        "units": "mm",
        "long_name": "Parameter in soil moisture control on root onset growth",
    },
)

hr["ndays_on_root"] = xr.DataArray(
    [30],
    dims=["allpft"],
    attrs={
        "units": "days",
        "long_name": "Number of days to complete root onset",
    },
)

hr["ndays_off_fcur"] = xr.DataArray(
    [138],
    dims=["allpft"],
    attrs={
        "units": "days",
        "long_name": "Number of days to decrease fcur_dyn from 1 to 0",
    },
)

hr["mort_tsoi"] = xr.DataArray(
    [10 + 273.15],
    dims=["allpft"],
    attrs={
        "units": "K",
        "long_name": "Parameter in soil moisture control on root growth and mortality",
    },
)

# Need to make tamarack root more flood tolerant
# also make spruce root more sensitive to temperature
hr["mort_a"] = xr.DataArray(
    [
        np.nan, np.nan, 0.006, 0.00175, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 0.00175, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,
    ],
    coords={"pft": hr["pft"]},
    dims=["pft"],
    attrs={
        "units": "",
        "long_name": "Parameter in soil temperature control on root mortality",
    },
)

hr["mort_psi"] = xr.DataArray(
    [
        np.nan, np.nan, 45, 25, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 45, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,
    ],
    coords={"pft": hr["pft"]},
    dims=["pft"],
    attrs={
        "units": "",
        "long_name": "Parameter in soil moisture control on root growth and mortality",
    },
)

hr["mort_h2o"] = xr.DataArray(
    [
        np.nan, np.nan, 0.6, 0.8, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 0.6, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,
    ],
    coords={"pft": hr["pft"]},
    dims=["pft"],
    attrs={
        "units": "",
        "long_name": "Parameter in soil moisture control on root growth and mortality",
    },
)
hr["mort_b"] = 3 # maximum mortality due to waterlogging
hr["mort_d"] = 0.05

hr["hardiness_root"] = xr.DataArray(
    [
        np.nan, np.nan, 273.15 - 7.5, 273.15 - 7.5, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 273.15 - 7.5, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,
    ],
    coords={"pft": hr["pft"]},
    dims=["pft"],
    attrs={
        "units": "K",
        "long_name": "Base temperature for GDD accumulation for root",
    },
)


encoding = {}
for data_var in hr.data_vars:
    if "_FillValue" in hr[data_var].encoding.keys():
        continue # somehow I cannot drop this line or E3SM throws error
    elif np.any(np.isnan(hr[data_var].values)):
        encoding[data_var] = {"_FillValue": -1e20}
    else:
        encoding[data_var] = {"_FillValue": None}
hr.to_netcdf(
    os.path.join(path_parameter, newfile), encoding=encoding, format="NETCDF3_CLASSIC"
)