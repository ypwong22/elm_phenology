""" Following previous studies on evergreen needleleaf phenology, test
    10%, 15%, 25%, 50%, 90% of the LAI amplitude during the green-up
    and green-down phase. 

    The seasonal amplitude is the difference between maximum and minimum baseline.

    Moon et al., 2021. Multiscale assessment of land surface phenology from harmonized Landsat 8 and Sentinel-2, PlanetScope, and PhenoCam imagery. Remote Sensing of Environment.

    Liu et al., 2020. Using the red chromatic coordinate to characterize the phenology of forest canopy photosynthesis. Agricultural and Forest Meteorology.

    Richardson et al., 2018. Intercomparison of phenological transition dates derived from the PhenoCam Dataset V1.0 and MODIS satellite remote sensing. Scientific Reports. 
"""
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import itertools as it
from utils.constants import *
from utils.paths import *
from utils.plotting import *
from utils.analysis import *
import xarray as xr
from datetime import datetime
import matplotlib as mpl
from gcc_spruce_visualize import *
from glob import glob
from scipy.signal import savgol_filter
from scipy.stats import linregress


def get_model_dates(variable, pft, read=False):
    if not read:
        pct = 25  # since GCC 25 was employed for GPP; ['15','25','50','90']
        years = range(2015, 2021)

        data = pd.DataFrame(
            np.nan,
            index=pd.MultiIndex.from_product([chamber_list, years]),
            columns=pd.MultiIndex.from_product(
                [sims_names, ["temperature", "CO2", "SOS", "EOS"]]
            ),
        )
        for prefix, model in zip(sims_prefix, sims_names):
            collection_ts = read_extract_sims_ts(prefix)
            tvec = collection_ts.index
            for chamber in chamber_list:
                value = (
                    collection_ts.loc[:, (chamber, variable, pft, "hummock")].values
                    * 0.64
                    + collection_ts.loc[:, (chamber, variable, pft, "hollow")].values
                    * 0.36
                )

                # use savgol filter to smooth the GPP
                if variable == "GPP":
                    try:
                        value = pd.Series(savgol_filter(value, 81, 3), index=tvec.year)
                    except:
                        import pdb

                        pdb.set_trace()
                else:
                    value = pd.Series(value, index=tvec.year)

                start = pd.Series(np.nan, index=np.unique(tvec.year))
                end = pd.Series(np.nan, index=np.unique(tvec.year))

                for y in np.unique(tvec.year):
                    """if y == tvec.year[0]:
                        temp = np.concatenate((value.loc[y], value.loc[y+1][:100]))
                        dayofyear = range(0, 365 + 100)
                    elif y == tvec.year[-1]:
                        temp = np.concatenate((value.loc[y-1][265:], value.loc[y]))
                        dayofyear = range(-100, 365)
                    else:
                        temp = np.concatenate((value.loc[y-1][265:], value.loc[y], value.loc[y+1][:100]))
                        dayofyear = range(-100, 365 + 100)
                    temp = pd.Series(temp, index = dayofyear)

                    ind = np.where((temp > temp.shift(-1)) & (temp > temp.shift(1)))[0][0]
                    val_max = temp.values[ind]

                    ind2 = np.where((temp < temp.shift(-1)) & (temp <= temp.shift(1)))[0][0]
                    val_min = temp.values[ind2]
                    threshold = float(pct) * (val_max - val_min) / 100 + val_min
                    filtered = ((temp - threshold) >= 0)
                    start.loc[y] = dayofyear[np.where(np.diff(np.insert(filtered.values, 0, 0).astype(int)) == 1)[0][-1]]

                    ind2 = np.where((temp <= temp.shift(-1)) & (temp < temp.shift(1)))[0][0]
                    val_min = temp.values[ind2]
                    threshold = float(pct) * (val_max - val_min) / 100 + val_min
                    filtered = ((temp - threshold) >= 0)
                    end.loc[y] = dayofyear[np.where(np.diff(np.insert(filtered.values, 0, 1).astype(int)) == -1)[0][-1]]
                    """

                    dayofyear = range(1, 366)
                    val_max = value.loc[y].max()
                    val_min = value.loc[y].min()
                    threshold = float(pct) * (val_max - val_min) / 100 + val_min
                    filtered = (value.loc[y] - threshold) >= 0
                    try:
                        start.loc[y] = dayofyear[
                            np.where(
                                np.diff(np.insert(filtered.values, 0, 0).astype(int))
                                == 1
                            )[0][-1]
                        ]
                        end.loc[y] = dayofyear[
                            np.where(
                                np.diff(np.insert(filtered.values, 0, 1).astype(int))
                                == -1
                            )[0][-1]
                        ]
                    except:
                        import pdb

                        pdb.set_trace()

                for year, ss, ee in zip(start.index, start, end):
                    data.loc[(chamber, year), (model, "temperature")] = chamber_levels[
                        f"{chamber:02d}"
                    ][0]
                    data.loc[(chamber, year), (model, "CO2")] = chamber_levels[
                        f"{chamber:02d}"
                    ][1]
                    if ss > 240:
                        data.loc[(chamber, year + 1), (model, "SOS")] = ss - 365
                    else:
                        data.loc[(chamber, year), (model, "SOS")] = ss

                    data.loc[(chamber, year), (model, "temperature")] = chamber_levels[
                        f"{chamber:02d}"
                    ][0]
                    data.loc[(chamber, year), (model, "CO2")] = chamber_levels[
                        f"{chamber:02d}"
                    ][1]
                    if ee < 120:
                        data.loc[(chamber, year - 1), (model, "EOS")] = ee + 365
                    else:
                        data.loc[(chamber, year), (model, "EOS")] = ee

        data.to_csv(
            os.path.join(
                path_out, "fit_model_phenology_{}_{}.csv".format(variable, pft)
            )
        )
    else:
        data = pd.read_csv(
            os.path.join(
                path_out, "fit_model_phenology_{}_{}.csv".format(variable, pft)
            ),
            index_col=[1, 2],
            header=[0, 1],
        )
    return data


def get_obs_dates(variable, pft):
    hr = xr.open_dataset(os.path.join(path_intrim, "spruce_validation_data.nc"))
    if pft == 2:
        pft_name = "EN"
    elif pft == 3:
        pft_name = "DN"
    elif pft == 11:
        pft_name = "SH"
    else:
        pft_name = "GR"
    if variable == "TLAI":
        obs_dates = (
            hr["pheno_dates_lai"]
            .loc[:, :, :, pft_name]
            .transpose("chamber", "year", "side")
        )
    elif variable == "GPP":
        sapflow_dates = (
            hr["pheno_dates_sapflow"]
            .loc[:, :, :, pft_name]
            .transpose("chamber", "year", "side")
        )
        gcc_dates = (
            hr["pheno_dates_gcc"]
            .loc[:, :, :, pft_name]
            .transpose("chamber", "year", "side")
        )
        obs_dates = (sapflow_dates, gcc_dates)
    else:
        if pft in [2, 3]:
            obs_dates = {
                "SOS": hr["sos_tree"].transpose("chamber", "year"),
                "EOS": hr["eos_tree"].transpose("chamber", "year"),
            }
        else:
            obs_dates = {
                "SOS": hr["sos_shrub"].transpose("chamber", "year"),
                "EOS": hr["eos_shrub"].transpose("chamber", "year"),
            }
    hr.close()
    return obs_dates


# for slides
read = False
mpl.rcParams["font.size"] = 14
mpl.rcParams["axes.titlesize"] = 14


# sims_prefix = ['20221212', '20230120', '20230122', '20230505']  # 20230121
# sims_names = ['Default', 'Optim', 'Optim Evgr', 'Optim EvgrRoot'] # 'Optim EvgrRoot'
sims_prefix = ["20230526", "20230623", "20230710"]
sims_names = ["Optim Scheme 2 Correct", "Optim EvgrRoot", "EvgrRoot NP"]
clist = ["#0000ff", "#800080", "#20b2aa", "#ff4040"]
co2_levels = {"ambient": [6, 20, 13, 8, 17], "elevated": [19, 11, 4, 16, 10]}
pft_list = [2, 3, 11]
pft_list_names = ["spruce", "larch", "shrub"]
for variable in ["TLAI", "GPP"]:
    for co2 in ["ambient", "elevated"]:
        fig, axes = plt.subplots(
            nrows=2,
            ncols=len(pft_list),
            figsize=(len(pft_list) * 4, 6),
            sharex=True,
            sharey=False,
        )
        fig.subplots_adjust(hspace=0.1, wspace=0.1)
        for i, side in enumerate(["SOS", "EOS"]):
            for j, (pft, pft_name) in enumerate(zip(pft_list, pft_list_names)):
                ax = axes[i, j]

                model_dates = get_model_dates(variable, pft, read)
                if variable == "GPP":
                    sapflow_dates, gcc_dates = get_obs_dates(variable, pft)
                elif variable == "TLAI":
                    obs_dates = get_obs_dates(variable, pft)
                else:
                    root_dates = get_obs_dates(variable, pft)

                if variable == "TLAI":
                    x = np.array(
                        [chamber_levels[f"{x:02d}"][0] for x in co2_levels[co2]]
                    )
                    x = np.broadcast_to(x.reshape(-1, 1), [len(x), obs_dates.shape[1]])
                    y = obs_dates.loc[co2_levels[co2], :, side].values

                    temp = np.isnan(x) | np.isnan(y)
                    x = x[~temp]
                    y = y[~temp]
                    if len(x) > 0:
                        h1 = ax_regress(
                            ax,
                            x,
                            y,
                            display=None,
                            args_pt={
                                "color": "k",
                                "marker": "o",
                                "markersize": 3,
                                "lw": 0,
                            },
                            args_ln={"color": "k", "lw": 1.2},
                            args_ci={"color": "k", "alpha": 0.1},
                        )

                        res = linregress(x, y)
                        if res.pvalue <= 0.05:
                            fontweight = "bold"
                        else:
                            fontweight = "normal"

                        if (i == 1) & (j == 0):
                            ax.text(
                                0.08,
                                0.02,
                                f"{res.slope:.2f}",
                                fontsize=8,
                                fontweight=fontweight,
                                transform=ax.transAxes,
                                color="k",
                            )
                        elif (i == 0) & (j == 0):
                            ax.text(
                                0.08,
                                0.1,
                                f"{res.slope:.2f}",
                                fontsize=8,
                                fontweight=fontweight,
                                transform=ax.transAxes,
                                color="k",
                            )
                        else:
                            ax.text(
                                0.08,
                                0.92,
                                f"{res.slope:.2f}",
                                fontsize=8,
                                fontweight=fontweight,
                                transform=ax.transAxes,
                                color="k",
                            )

                elif variable == "GPP":
                    x = np.array(
                        [chamber_levels[f"{x:02d}"][0] for x in co2_levels[co2]]
                    )
                    x = np.broadcast_to(
                        x.reshape(-1, 1), [len(x), sapflow_dates.shape[1]]
                    )
                    y1 = sapflow_dates.loc[co2_levels[co2], :, side].values
                    y2 = gcc_dates.loc[co2_levels[co2], :, side].values

                    temp = np.isnan(x) | np.isnan(y1)
                    if (~temp).sum() > 0:
                        h1 = ax_regress(
                            ax,
                            x[~temp] - 0.25,
                            y1[~temp],
                            display=None,
                            args_pt={
                                "color": "k",
                                "marker": "o",
                                "markersize": 3,
                                "lw": 0,
                            },
                            args_ln={"color": "k", "lw": 1.2},
                            args_ci={"color": "k", "alpha": 0.1},
                        )

                        res = linregress(x[~temp], y1[~temp])
                        if res.pvalue <= 0.05:
                            fontweight = "bold"
                        else:
                            fontweight = "normal"
                        ax.text(
                            0.08,
                            0.92,
                            f"{res.slope:.2f}",
                            fontsize=8,
                            fontweight=fontweight,
                            transform=ax.transAxes,
                            color="k",
                        )

                    temp = np.isnan(x) | np.isnan(y2)
                    if (~temp).sum() > 0:
                        h2 = ax_regress(
                            ax,
                            x[~temp],
                            y2[~temp],
                            display=None,
                            args_pt={
                                "color": "#756bb1",
                                "marker": "o",
                                "markersize": 3,
                                "lw": 0,
                            },
                            args_ln={"color": "#756bb1", "lw": 1.2},
                            args_ci={"color": "#756bb1", "alpha": 0.1},
                        )

                        res = linregress(x[~temp], y2[~temp])
                        if res.pvalue <= 0.05:
                            fontweight = "bold"
                        else:
                            fontweight = "normal"
                        ax.text(
                            0.08,
                            0.82,
                            f"{res.slope:.2f}",
                            fontsize=8,
                            fontweight=fontweight,
                            transform=ax.transAxes,
                            color="#756bb1",
                        )

                else:
                    if side == "SOS" and co2 == "elevated":
                        (h1,) = ax.plot(
                            chamber_levels["10"][0],
                            root_dates[side].loc[10, :].loc[2019],
                            "o",
                            markersize=3,
                            color="k",
                        )
                        ax.plot(
                            chamber_levels["19"][0],
                            root_dates[side].loc[19, :].loc[2019],
                            "o",
                            markersize=3,
                            color="k",
                        )
                    else:
                        (h1,) = ax.plot(
                            chamber_levels["06"][0],
                            root_dates[side].loc[6, :].loc[2015],
                            "o",
                            markersize=3,
                            color="k",
                        )

                h3 = [None] * len(sims_names)
                for k, model in enumerate(sims_names):
                    if co2 == "ambient":
                        x = np.array(
                            [
                                chamber_levels[f"{x:02d}"][0]
                                for x in model_dates.index.get_level_values(0)
                                if chamber_levels[f"{x:02d}"][1] == 0
                            ]
                        ) + 0.25 * (k + 1)
                        y = model_dates.loc[
                            [
                                x
                                for x in chamber_list
                                if chamber_levels[f"{x:02d}"][1] == 0
                            ],
                            (model, side),
                        ].values
                    else:
                        x = np.array(
                            [
                                chamber_levels[f"{x:02d}"][0]
                                for x in model_dates.index.get_level_values(0)
                                if chamber_levels[f"{x:02d}"][1] == 500
                            ]
                        ) + 0.25 * (k + 1)
                        y = model_dates.loc[
                            [
                                x
                                for x in chamber_list
                                if chamber_levels[f"{x:02d}"][1] == 500
                            ],
                            (model, side),
                        ].values
                    temp = np.isnan(x) | np.isnan(y)
                    if (~temp).sum() > 0:
                        h3[k] = ax_regress(
                            ax,
                            x[~temp],
                            y[~temp],
                            display=None,
                            args_pt={
                                "color": clist[k],
                                "marker": "o",
                                "markersize": 3,
                                "lw": 0,
                            },
                            args_ln={"color": clist[k], "lw": 1.2},
                            args_ci={"color": clist[k], "alpha": 0.1},
                        )

                        res = linregress(x[~temp], y[~temp])
                        if res.pvalue <= 0.05:
                            fontweight = "bold"
                        else:
                            fontweight = "normal"

                        if (variable != "GPP") & (i == 1) & (j == 0):
                            ax.text(
                                0.28 + 0.2 * k,
                                0.02,
                                f"{res.slope:.2f}",
                                fontsize=8,
                                fontweight=fontweight,
                                transform=ax.transAxes,
                                color=clist[k],
                            )
                        elif (variable == "TLAI") & (i == 0) & (j == 0):
                            ax.text(
                                0.28 + 0.2 * k,
                                0.1,
                                f"{res.slope:.2f}",
                                fontsize=8,
                                fontweight=fontweight,
                                transform=ax.transAxes,
                                color=clist[k],
                            )
                        else:
                            ax.text(
                                0.28 + 0.2 * k,
                                0.92,
                                f"{res.slope:.2f}",
                                fontsize=8,
                                fontweight=fontweight,
                                transform=ax.transAxes,
                                color=clist[k],
                            )

                if j == 0:
                    if side == "SOS":
                        ax.set_ylabel("SOS")
                    else:
                        ax.set_ylabel("EOS")
                else:
                    ax.set_yticklabels([])

                if i == 0:
                    ax.set_title(pft_name)

                if side == "SOS":
                    ax.set_ylim([0, 240])
                else:
                    ax.set_ylim([200, 460])
                    ax.set_xlabel(r"$\Delta$ Warming ($^o$C)")

        if variable == "GPP":
            ax.legend(
                [h1, h2] + h3,
                ["Sapflow", "GCC"] + sims_names,
                ncol=5,
                loc=(-2.2, -0.5),
                columnspacing=1,
            )
        else:
            ax.legend(
                [h1] + h3,
                ["Ground obs"] + sims_names,
                ncol=5,
                loc=(-2.2, -0.5),
                columnspacing=1,
            )
        fig.savefig(
            os.path.join(
                path_out,
                "phenology_temperature_sensitivity_{}_{}.png".format(variable, co2),
            ),
            dpi=600.0,
            bbox_inches="tight",
        )
        plt.close(fig)
