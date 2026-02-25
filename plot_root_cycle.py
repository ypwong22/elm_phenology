""" Compare the seasonal cycle of fine root in the new v.s. old root model """
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


def plotter_monthly(var_list, unit_list, minmax_list, filename):
    fig, axes = plt.subplots(
        nrows=len(var_list),
        ncols=3,
        figsize=(14, 2.5 * len(var_list)),
        sharex=True,
        sharey=False,
    )
    fig.subplots_adjust(hspace=0.12, wspace=0.05)

    for k, (varname, unit, minmax) in enumerate(zip(var_list, unit_list, minmax_list)):
        for i, (pft, pftname) in enumerate(
            zip([2, 3, 11], ["spruce", "larch", "shrub"])
        ):
            ax = axes[k, i]
            for j, prefix in enumerate(sims_prefix):
                collection_ts = read_extract_sims_ts(prefix)

                if "/" in varname:
                    num_var = varname.split("/")[0]
                    num = (
                        collection_ts.loc[:, (slice(None), num_var, pft, "hummock")]
                        * 0.64
                        + collection_ts.loc[
                            :, (slice(None), num_var, pft, "hollow")
                        ].values
                        * 0.36
                    )
                    num.columns = num.columns.droplevel([1, 2, 3])
                    num = num.resample("1MS").mean()
                    if num_var in ["AR", "FROOT_MR", "CPOOL_FROOT_GR"]:
                        num = num * 86400

                    denom_var = varname.split("/")[1]
                    denom = (
                        collection_ts.loc[:, (slice(None), denom_var, pft, "hummock")]
                        * 0.64
                        + collection_ts.loc[
                            :, (slice(None), denom_var, pft, "hollow")
                        ].values
                        * 0.36
                    )
                    denom.columns = denom.columns.droplevel([1, 2, 3])

                    denom = denom.resample("1MS").mean()
                    if denom_var in ["GPP", "FROOTC_ALLOC"]:
                        denom = denom * 86400

                    frootc = num / denom.values
                    frootc[denom.values < 1e-10] = np.nan
                    frootc = frootc.resample("1MS").mean()
                else:
                    frootc = (
                        collection_ts.loc[:, (slice(None), varname, pft, "hummock")]
                        * 0.64
                        + collection_ts.loc[
                            :, (slice(None), varname, pft, "hollow")
                        ].values
                        * 0.36
                    )
                    frootc.columns = frootc.columns.droplevel([1, 2, 3])
                    frootc = frootc.resample("1MS").mean()

                    if "day-1" in unit:
                        # convert g s to g day-1
                        frootc = frootc * 86400

                    if unit == "ratio to annual\naverage":
                        # normalize to 1 in annual average
                        frootc = frootc.groupby(frootc.index.year).apply(
                            lambda df: df / df.mean(axis=0)
                        )

                frootc = frootc.stack().to_frame()
                frootc_mean = (
                    frootc.groupby(frootc.index.get_level_values(0).month)
                    .mean()
                    .iloc[:, 0]
                )
                frootc_std = (
                    frootc.groupby(frootc.index.get_level_values(0).month)
                    .std()
                    .iloc[:, 0]
                )

                ax.plot(
                    frootc_mean.index,
                    frootc_mean,
                    "-",
                    color=clist[j],
                    label=sims_names[j],
                )
                ax.fill_between(
                    frootc_mean.index,
                    frootc_mean - frootc_std,
                    frootc_mean + frootc_std,
                    color=clist[j],
                    alpha=0.2,
                )

                if varname in ["TLAI", "FROOTC"]:
                    if pft == 2:
                        if j == 0:
                            ax_inset = inset_axes(
                                ax, width="80%", height="40%", loc="upper right"
                            )
                        ax_inset.plot(
                            frootc_mean.index,
                            frootc_mean,
                            "-",
                            color=clist[j],
                            label=sims_names[j],
                        )
                        ax_inset.set_ylim([0.85, 1.2])
                        ax_inset.set_xticks([])

            if k == 0:
                ax.set_title(pftname)
            elif k == (len(var_list) - 1):
                ax.set_xlabel("Month")

            if len(minmax) > 0:
                ax.set_ylim(minmax)

            if i == 0:
                # if '/' in varname:
                #    varname2 = varname.replace('/', '/\n')
                # else:
                varname2 = varname
                ax.set_ylabel(f"{varname2}\n({unit})")
            else:
                if len(minmax) > 0:
                    ax.set_yticklabels([])
            ax.set_xticks(range(1, 13))
    ax.legend(loc=(-2, -0.6), ncol=4)
    fig.savefig(os.path.join(path_out, filename), dpi=600.0, bbox_inches="tight")
    plt.close(fig)


def plotter_annual(var_list, unit_list, minmax_list, filename):
    fig, axes = plt.subplots(
        nrows=len(var_list),
        ncols=3,
        figsize=(14, 2.2 * len(var_list)),
        sharex=True,
        sharey=False,
    )
    fig.subplots_adjust(hspace=0.12, wspace=0.05)
    for k, (varname, unit, minmax) in enumerate(zip(var_list, unit_list, minmax_list)):
        for i, (pft, pftname) in enumerate(
            zip([2, 3, 11], ["spruce", "larch", "shrub"])
        ):
            ax = axes[k, i]
            for j, prefix in enumerate(sims_prefix):
                collection_ts = read_extract_sims_ts(prefix)

                if "/" in varname:
                    num_var = varname.split("/")[0]
                    num = (
                        collection_ts.loc[:, (slice(None), num_var, pft, "hummock")]
                        * 0.64
                        + collection_ts.loc[
                            :, (slice(None), num_var, pft, "hollow")
                        ].values
                        * 0.36
                    )
                    num.columns = num.columns.droplevel([1, 2, 3])
                    num = num.resample("1YS").mean() * 86400

                    denom_var = varname.split("/")[1]
                    denom = (
                        collection_ts.loc[:, (slice(None), denom_var, pft, "hummock")]
                        * 0.64
                        + collection_ts.loc[
                            :, (slice(None), denom_var, pft, "hollow")
                        ].values
                        * 0.36
                    )
                    denom.columns = denom.columns.droplevel([1, 2, 3])
                    denom = denom.resample("1YS").mean() * 86400

                    frootc = num / denom.values
                    frootc[denom.values < 1e-10] = np.nan
                    frootc = frootc.resample("1YS").mean()
                else:
                    frootc = (
                        collection_ts.loc[:, (slice(None), varname, pft, "hummock")]
                        * 0.64
                        + collection_ts.loc[
                            :, (slice(None), varname, pft, "hollow")
                        ].values
                        * 0.36
                    )
                    frootc.columns = frootc.columns.droplevel([1, 2, 3])
                    frootc = frootc.resample("1YS").mean()

                    if varname in ["GPP", "AR"]:
                        # convert g s to g day-1
                        frootc = frootc * 86400
                    elif varname == "FROOTC":
                        # normalize to 1 in annual average
                        frootc = frootc.groupby(frootc.index.year).apply(
                            lambda df: df / df.mean(axis=0)
                        )

                frootc_mean = frootc.mean(axis=1)
                frootc_std = frootc.std(axis=1)

                ax.plot(
                    frootc_mean.index,
                    frootc_mean,
                    "-",
                    color=clist[j],
                    label=sims_names[j],
                )
                ax.fill_between(
                    frootc_mean.index,
                    frootc_mean - frootc_std,
                    frootc_mean + frootc_std,
                    color=clist[j],
                    alpha=0.2,
                )

            if k == 0:
                ax.set_title(pftname)
            elif k == (len(var_list) - 1):
                ax.set_xlabel("Year")

            if len(minmax) > 0:
                ax.set_ylim(minmax)

            if i == 0:
                if "/" in varname:
                    varname2 = varname.replace("/", "/\n")
                else:
                    varname2 = varname
                ax.set_ylabel(f"{varname2}\n({unit})")
            else:
                if len(minmax) > 0:
                    ax.set_yticklabels([])

    ax.legend(loc=(-2, -0.6), ncol=4)
    fig.savefig(os.path.join(path_out, filename), dpi=600.0, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    mpl.rcParams["font.size"] = 14
    mpl.rcParams["axes.titlesize"] = 14

    # sims_prefix = ['20221212', '20230120', '20230505']  # '20230122', 20230121
    # sims_names = ['Default', 'Optim', 'Optim EvgrRoot'] # 'Optim Evgr', 'Optim EvgrRoot'
    # sims_prefix = ["20221212", "20230120", "20230526", "20230601"]
    # sims_names = ["Default", "Optim XYS", "Optim Scheme 2 Correct", "Optim EvgrRoot"]
    sims_prefix = ["20230526", "20230623", "20230710"]
    sims_names = ["Optim Scheme 2 Correct", "Optim EvgrRoot", "EvgrRoot NP"]
    clist = ["#0000ff", "#800080", "#20b2aa", "#ff4040"]

    var_list = [
        "TLAI",
        "LEAFC",
        "LEAFC_STORAGE",
        "FROOTC",
        "FROOTC_STORAGE",
    ]  # "BGLFR_FROOT",
    unit_list = [
        "ratio to annual\naverage",
        "ratio to annual\naverage",
        "ratio to annual\naverage",
        "ratio to annual\naverage",
        "ratio to annual\naverage",
    ]  # "s-1",
    # minmax_list = [[-0.3, 3.2], [0.5, 1.5], [0.5, 1.5], [-0.3, 3.2], [-0.5e-9, 9e-9], [-0.1e-8, 5e-8]]
    minmax_list = [[], [], [], [], [], []]
    plotter_monthly(var_list, unit_list, minmax_list, "plot_root_cycle_lai.png")

    """
    var_list = ['AR/TOTVEGC', 'LEAF_MR/LEAFC', 'FROOT_MR/FROOTC', 'DOWNREG'] # 'CPOOL_FROOT_GR/FROOTC', 'FROOTC/FROOTC_STORAGE', 
    unit_list = ['g g-1 day-1','g g-1 day-1','g g-1 day-1', ''] # 'g g-1 day-1', 'g g-1'
    minmax_list = [[-1e-4, 0.004], [-1e-8, 1.5e-7], [0, 0.013], [-0.05, 0.75]]
    plotter_monthly(var_list, unit_list, minmax_list, 'plot_root_cycle.png')
    """

    """
    var_list = [
        "ONSET_FLAG",
        "OFFSET_FLAG",
        "ONSET_FLAG_ROOT",
    ]  # 'CPOOL_FROOT_GR/FROOTC', 'FROOTC/FROOTC_STORAGE',
    unit_list = ["", ""]
    minmax_list = [[], []]
    plotter_monthly(var_list, unit_list, minmax_list, "plot_root_cycle_pheno.png")
    """

    """
    # Annual ratio of AR/GPP, CPOOL_FROOT_GR/FROOTC_ALOC
    var_list = ['AR/GPP', 'CPOOL_FROOT_GR/FROOTC_ALLOC']
    unit_list = ['g g-1 day-1', 'g g-1 day-1']
    minmax_list = [[0, 0.125], [-1e-4, 0.019]]
    plotter_annual(var_list, unit_list, minmax_list, f'plot_root_cycle_annual.png')
    """
