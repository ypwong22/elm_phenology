""" Compare the seasonal cycle of air temperature, soil temperature, and soil moisture """
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import itertools as it
import xarray as xr
import matplotlib as mpl
from utils.constants import *
from utils.paths import *
from utils.plotting import *
from utils.analysis import *
from datetime import datetime
from gcc_spruce_visualize import *
from glob import glob
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


hr = xr.open_dataset(os.path.join(path_intrim, "spruce_validation_data.nc"))
# 5 cm; but apparently there's actuall no obsesrvation data for this layer
obs_tsoi_5 = (
    (
        hr["halfhourly_tsoil_hummock"][:, :, 1] * 0.64
        + hr["halfhourly_tsoil_hollow"][:, :, 1] * 0.36
    )
    .resample({"time": "1MS"})
    .mean()
)
obs_tsoi_5 = pd.DataFrame(
    obs_tsoi_5.values, index=obs_tsoi_5["time"].to_index(), columns=hr["chamber"]
)
# 10 cm
obs_tsoi_10 = (
    (
        hr["halfhourly_tsoil_hummock"][:, :, 2] * 0.64
        + hr["halfhourly_tsoil_hollow"][:, :, 2] * 0.36
    )
    .resample({"time": "1MS"})
    .mean()
)
obs_tsoi_10 = pd.DataFrame(
    obs_tsoi_10.values, index=obs_tsoi_10["time"].to_index(), columns=hr["chamber"]
)
# hummock & hollo
obs_swc = (
    hr["halfhourly_swc_hummock"].resample({"time": "1MS"}).mean() * 0.64
    + hr["halfhourly_swc_hollow"].resample({"time": "1MS"}).mean() * 0.36
)
obs_swc = pd.DataFrame(
    obs_swc.values, index=obs_swc["time"].to_index(), columns=hr["chamber"]
)
hr.close()

mpl.rcParams["font.size"] = 14
mpl.rcParams["axes.titlesize"] = 14

# sims_prefix = ['20221212', '20230120', '20230505']  # '20230122', 20230121
# sims_names = ['Default', 'Optim', 'Optim EvgrRoot'] # 'Optim Evgr', 'Optim EvgrRoot'
sims_prefix = ["20221212", "20230120", "20230526", "20230601"]
sims_names = ["Default", "Optim XYS", "Optim Scheme 2 Correct", "Optim EvgrRoot"]
clist = ["#0000ff", "#800080", "#20b2aa", "#ff4040"]
var_list = ["TBOT", "TSOI_3", "TSOI_4", "SWC", "H2OSOI", "NET_NMIN", "BTRAN"]
unit_list = ["$^o$C", "$^o$C", "$^o$C", "", "", "", ""]


fig, axes = plt.subplots(
    nrows=len(var_list),
    ncols=3,
    figsize=(12, 2 * len(var_list)),
    sharex=False,
    sharey=False,
)
fig.subplots_adjust(hspace=0.1, wspace=0.25)

for k, (varname, unit) in enumerate(zip(var_list, unit_list)):
    for j, prefix in enumerate(sims_prefix):
        collection_ts = read_extract_sims_ts(prefix)
        month = pd.DatetimeIndex(
            [
                datetime(m, n, 1)
                for m, n in zip(
                    collection_ts.index.get_level_values(0).year,
                    collection_ts.index.get_level_values(0).month,
                )
            ]
        )

        if varname in ["SWC", "H2OSOI"]:
            var = (
                collection_ts.loc[:, (slice(None), f"{varname}_1", 0, "hummock")] * 0.64
                + collection_ts.loc[
                    :, (slice(None), f"{varname}_1", 0, "hollow")
                ].values
                * 0.36
            ) * 0.5 + (
                collection_ts.loc[:, (slice(None), f"{varname}_5", 0, "hummock")] * 0.64
                + collection_ts.loc[
                    :, (slice(None), f"{varname}_5", 0, "hollow")
                ].values
                * 0.36
            ).values * 0.5
        else:
            var = (
                collection_ts.loc[:, (slice(None), varname, 0, "hummock")] * 0.64
                + collection_ts.loc[:, (slice(None), varname, 0, "hollow")].values
                * 0.36
            )

        var_chamber = var.mean(axis=0)
        t_chamber = (
            collection_ts.loc[:, (slice(None), "TBOT", 0, "hummock")] * 0.64
            + collection_ts.loc[:, (slice(None), "TBOT", 0, "hollow")].values * 0.36
        )
        t_chamber = t_chamber.mean(axis=0) - 273.15

        var.columns = var.columns.droplevel([1, 2, 3])
        var = var.groupby(month).mean()
        if varname == "TBOT":
            var = var - 273.15
        var = var.stack().to_frame()

        var_annual = (
            var.groupby(var.index.get_level_values(0).year).mean().values.reshape(-1)
        )
        var_seasonal = (
            var.groupby(var.index.get_level_values(0).month).mean().values.reshape(-1)
        )
        var_seasonal_std = (
            var.groupby(var.index.get_level_values(0).month).std().values.reshape(-1)
        )

        ax = axes[k, 0]
        ax.plot(range(1, 13), var_seasonal, "-", color=clist[j], label=sims_names[j])
        ax.fill_between(
            range(1, 13),
            var_seasonal - var_seasonal_std,
            var_seasonal + var_seasonal_std,
            color=clist[j],
            alpha=0.2,
        )
        if (
            varname.startswith("SWC_")
            or varname.startswith("H2OSOI_")
            or varname.startswith("TSOI_")
        ):
            layer = str(int(varname.split("_")[1]) - 1)
            varname2 = varname.split("_")[0]
            ax.set_ylabel(f"{varname2}_{layer} {unit}")
        else:
            ax.set_ylabel(f"{varname} {unit}")
        ax.set_xticks(range(1, 13))
        if k == (len(var_list) - 1):
            ax.set_xlabel("Month")
        else:
            ax.set_xticklabels([])
        # 5 degrees cutoff
        if "TSOI" in varname:
            ax.axhline(5, color="k", ls="--", lw=0.5)

        ax = axes[k, 1]
        ax.plot(range(2015, 2021), var_annual, "-", color=clist[j], label=sims_names[j])
        if k == (len(var_list) - 1):
            ax.set_xlabel("Year")
        else:
            ax.set_xticklabels([])

        ax = axes[k, 2]
        ax.plot(t_chamber, var_chamber, "o", color=clist[j], label=sims_names[j])
        if k == (len(var_list) - 1):
            ax.set_xlabel("Chamber mean temperature ($^o$C)")
        else:
            ax.set_xticklabels([])

    if varname in ["TSOI_3", "TSOI_4", "SWC", "H2OSOI"]:
        if varname == "TSOI_3":
            obs = obs_tsoi_5  # apparently there's actuall no obsesrvation data
        if varname == "TSOI_4":
            obs = obs_tsoi_10
        elif varname in ["SWC", "H2OSOI"]:
            obs = obs_swc
        obs_chamber = obs.mean(axis=0).loc[var_chamber.index.get_level_values(0)]
        obs = obs.stack()
        obs_annual = obs.groupby(obs.index.get_level_values(0).year).mean()
        obs_seasonal = obs.groupby(obs.index.get_level_values(0).month).mean()
        obs_seasonal_std = obs.groupby(obs.index.get_level_values(0).month).std()
        ax = axes[k, 0]
        ax.plot(obs_seasonal.index, obs_seasonal, "-", color="k", label="Obs.")
        ax.fill_between(
            obs_seasonal.index,
            obs_seasonal - obs_seasonal_std,
            obs_seasonal + obs_seasonal_std,
            color="k",
            alpha=0.2,
        )
        ax = axes[k, 1]
        ax.plot(obs_annual.index, obs_annual, "-", color="k", label="Obs")
        ax = axes[k, 2]
        ax.plot(t_chamber, obs_chamber, "o", color="k", label="Obs")

ax.legend(loc=(-2, -0.8), ncol=4, columnspacing=1)
fig.savefig(
    os.path.join(path_out, f"plot_env_cycle.png"), dpi=600.0, bbox_inches="tight"
)
plt.close(fig)
