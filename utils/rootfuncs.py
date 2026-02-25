import os
from .paths import *
import pandas as pd
import numpy as np
import xarray as xr
from .phenofuncs import _find_solstice
from .constants import *
from .analysis import read_extract_sims_ts
from glob import glob
import itertools as it
import warnings
from datetime import datetime


def get_observation():
    """minirhizotron and ingrowth core data in the unit of gC m-2 grid area day-1"""
    # Convert the g biomass to gC
    cconc = pd.read_csv(os.path.join(path_input, "FRED3_cleaned.csv"))

    picea_rootC = cconc.loc[
        (cconc["Plant taxonomy_Accepted genus_TPL"] == "Picea")
        & (cconc["Plant Taxonomy_Accepted species_TPL"] != "engelmannii"),
        [
            "Plant taxonomy_Accepted genus_TPL",
            "Plant Taxonomy_Accepted species_TPL",
            "Root C content",
        ],
    ].dropna(axis=0)
    picea_rootC = picea_rootC["Root C content"].mean() / 1000  # mg/g

    larch_rootC = cconc.loc[
        (cconc["Plant taxonomy_Accepted genus_TPL"] == "Larix")
        & (cconc["Plant Taxonomy_Accepted species_TPL"] == "decidua"),
        [
            "Plant taxonomy_Accepted genus_TPL",
            "Plant Taxonomy_Accepted species_TPL",
            "Root C content",
        ],
    ].dropna(axis=0)
    larch_rootC = larch_rootC["Root C content"].mean() / 1000.0

    shrub_rootC = cconc.loc[
        (cconc["Plant taxonomy_Accepted family_TPL"] == "Ericaceae"),
        [
            "Plant taxonomy_Accepted genus_TPL",
            "Plant Taxonomy_Accepted species_TPL",
            "Root C content",
        ],
    ].dropna(axis=0)
    shrub_rootC = shrub_rootC["Root C content"].mean() / 1000.0

    # Soren's minirhizotron data
    # minirhizotron = pd.read_csv(os.path.join(path_input, 'soren_root_prod_mort_growth_20230316.csv'))
    minirhizotron = pd.read_csv(
        os.path.join(path_input, "soren_root_prod_mort_growth_20230511.csv")
    )
    minirhizotron["end_date"] = pd.DatetimeIndex(minirhizotron["end_date"])
    minirhizotron["start_date"] = pd.DatetimeIndex(minirhizotron["start_date"])
    minirhizotron = minirhizotron.loc[minirhizotron["pft"] != "sedge"]

    # Take the absolute value of mortality rate
    minirhizotron["m_g_d"] = minirhizotron["m_g_d"].abs()

    # Convert from g biomass to gC
    # pima: 0.36, lala: 0.14
    minirhizotron.loc[minirhizotron["pft"] == "tree", "m_g_d"] *= (
        picea_rootC * 0.36 + larch_rootC * 0.14
    ) / 0.5
    minirhizotron.loc[minirhizotron["pft"] == "tree", "g_g_d"] *= (
        picea_rootC * 0.36 + larch_rootC * 0.14
    ) / 0.5
    minirhizotron.loc[minirhizotron["pft"] == "shrub", "m_g_d"] *= shrub_rootC
    minirhizotron.loc[minirhizotron["pft"] == "shrub", "g_g_d"] *= shrub_rootC

    minirhizotron = minirhizotron.drop(
        [
            "temp",
            "co2",
            "year",
            "time_step",
            "npp_km_d",
            "m_km_d",
            "g_km_d",
            "npp_g_d",
            "tube",
        ],
        axis=1,
    ).set_index(["plot", "topo", "pft", "start_date", "end_date"])
    minirhizotron = minirhizotron.dropna(axis=0, how="all").sort_index(axis=0)

    # 2014-2017 ingrowth core data
    # winter data is near zero
    ingrowth = pd.read_csv(
        os.path.join(path_input, "spruce_root_ingrowth_2014_2017_20200408.csv"),
        header=[0, 1],
        encoding="Windows-1252",
    )
    ingrowth.columns = [
        f"{i} {j}" if not "Unnamed" in j else f"{i}"
        for i, j in zip(
            ingrowth.columns.get_level_values(0), ingrowth.columns.get_level_values(1)
        )
    ]
    ingrowth["start_date yyyy-mm-dd"] = pd.DatetimeIndex(
        ingrowth["start_date yyyy-mm-dd"]
    )
    ingrowth["end_date yyyy-mm-dd"] = pd.DatetimeIndex(ingrowth["end_date yyyy-mm-dd"])
    ingrowth["root_biomass g m-2 day-1"] = ingrowth[
        "root_biomass g/m2/season"
    ] / np.array(
        [
            (d.days + 1)
            for d in (
                ingrowth["end_date yyyy-mm-dd"] - ingrowth["start_date yyyy-mm-dd"]
            )
        ]
    )
    ingrowth = ingrowth.loc[ingrowth["pft"] != "graminoid", :]
    ingrowth.loc[ingrowth["pft"] == "spruce", "root_biomass g m-2 day-1"] = (
        ingrowth.loc[ingrowth["pft"] == "spruce", "root_biomass g m-2 day-1"].values
        * picea_rootC
    )
    ingrowth.loc[ingrowth["pft"] == "shrub", "root_biomass g m-2 day-1"] = (
        ingrowth.loc[ingrowth["pft"] == "shrub", "root_biomass g m-2 day-1"].values
        * shrub_rootC
    )
    ingrowth.loc[ingrowth["pft"] == "larch", "root_biomass g m-2 day-1"] = (
        ingrowth.loc[ingrowth["pft"] == "larch", "root_biomass g m-2 day-1"].values
        * larch_rootC
    )

    season = np.array([d.split("_")[0].lower() for d in ingrowth["season"]])
    ingrowth = ingrowth.loc[season == "summer", :].copy()
    ingrowth = ingrowth.groupby(
        [
            "topog hummock or hollow",
            "plot",
            "start_date yyyy-mm-dd",
            "end_date yyyy-mm-dd",
            "pft",
        ]
    ).sum()[["root_biomass g m-2 day-1"]]

    ingrowth.index.names = ["topo", "plot", "start_date", "end_date", "pft"]

    return minirhizotron, ingrowth


def hh_average(x):
    hum = x.index.get_level_values("topo") == "hummock"
    hol = x.index.get_level_values("topo") == "hollow"
    if (sum(hum) > 0) & (sum(hol) > 0):
        data = 0.64 * x.loc[hum, :].values + 0.36 * x.loc[hol, :].values
    else:
        data = np.full(x.shape[1], np.nan)
    return pd.Series(data.reshape(-1), index=x.columns)


def convert_observation(minirhizotron, ingrowth):
    """Interpolate the minirhizotron data to daily values, and calculate
    (1) annual total growth during the common observation period of all the years & chambers (unit: gC m-2 grid area)
    (2) seasonal cycle in 2018, averaged over all the chambers (pre-normalized to 1 before averaging)

    Intersect the ingrowth core data to uniform dates of the year and sum up (unit: gC m-2 grid area)
    """
    ###############
    # minirhizotron
    ###############
    # average over hollow and hummock
    minirhizotron = (
        minirhizotron.groupby(["start_date", "end_date", "plot", "pft"])
        .apply(hh_average)
        .dropna(how="all")
        .unstack()
        .unstack()
    )
    minirhizotron.columns.names = ["variable", "pft", "plot"]

    # The observation period in 2018 (3/10-10/18) is longer than in 2019 (5/1-9/14) or 2020 (6/19-8/28)
    date_pairs = minirhizotron.index.to_frame(index=False)

    date_latest_start = []
    date_earliest_end = []
    for year in [2015, 2018, 2019, 2020]:
        start = pd.DatetimeIndex(
            date_pairs.loc[
                pd.DatetimeIndex(date_pairs["start_date"]).year == year, "start_date"
            ]
        ).min()
        date_latest_start.append(start.month * 100 + start.day)
        end = pd.DatetimeIndex(
            date_pairs.loc[
                pd.DatetimeIndex(date_pairs["end_date"]).year == year, "end_date"
            ]
        ).max()
        date_earliest_end.append(end.month * 100 + end.day)
    date_latest_start = max(date_latest_start)
    date_earliest_end = min(date_earliest_end)
    print(date_latest_start, date_earliest_end)

    # (1) annual total growth during 6/19-8/28
    annual_minirhizotron = pd.DataFrame(
        0.0, index=[2015, 2018, 2019, 2020], columns=minirhizotron.columns
    )
    for y in [2015, 2018, 2019, 2020]:
        temp = minirhizotron.loc[
            pd.DatetimeIndex(date_pairs["start_date"]).year == y, :
        ]
        for ind, row in temp.iterrows():
            date_start = max(ind[0].month * 100 + ind[0].day, date_latest_start)
            date_end = min(ind[1].month * 100 + ind[1].day, date_earliest_end)
            if date_end > date_start:
                ndays = (
                    datetime(y, int(date_end / 100), np.mod(date_end, 100))
                    - datetime(y, int(date_start / 100), np.mod(date_start, 100))
                ).days + 1
                annual_minirhizotron.loc[y, :] = (
                    annual_minirhizotron.loc[y, :] + row.values * ndays
                )
    # merge chamber 7 and chamber 21 to chamber 7, because both are TAMB
    for v, pft, y in it.product(
        ["m_g_d", "g_g_d"], ["shrub", "tree"], [2015, 2018, 2019, 2020]
    ):
        if ~np.isnan(annual_minirhizotron.loc[y, (v, pft, 21)]):
            if np.isnan(annual_minirhizotron.loc[y, (v, pft, 7)]):
                annual_minirhizotron.loc[y, (v, pft, 7)] = annual_minirhizotron.loc[
                    y, (v, pft, 21)
                ]
            else:
                annual_minirhizotron.loc[y, (v, pft, 7)] = 0.5 * (
                    annual_minirhizotron.loc[y, (v, pft, 7)]
                    + annual_minirhizotron.loc[y, (v, pft, 21)]
                )
    annual_minirhizotron = annual_minirhizotron.drop(21, axis=1, level=2)

    # (2) normalized seasonal cycle in 2015 and 2018
    minirhizotron_cycle = minirhizotron.stack().stack().reset_index()
    minirhizotron_cycle["year"] = [t.year for t in minirhizotron_cycle["start_date"]]
    minirhizotron_cycle["month"] = [t.month for t in minirhizotron_cycle["start_date"]]
    minirhizotron_cycle = minirhizotron_cycle.drop(
        ["start_date", "end_date"], axis=1
    ).set_index(["year", "month", "plot", "pft"])
    minirhizotron_cycle_mean = (
        minirhizotron_cycle.groupby(["month", "pft"]).mean().unstack()
    )
    minirhizotron_cycle_std = (
        minirhizotron_cycle.groupby(["month", "pft"]).std().unstack()
    )
    minirhizotron_cycle_std = minirhizotron_cycle_std / minirhizotron_cycle_mean.sum(
        axis=0, skipna=False
    )
    minirhizotron_cycle_mean = minirhizotron_cycle_mean / minirhizotron_cycle_mean.sum(
        axis=0, skipna=False
    )

    ###############
    # ingrowth
    ###############
    # average over hollow and hummock
    ingrowth = (
        ingrowth.groupby(["start_date", "end_date", "plot", "pft"])
        .apply(hh_average)
        .unstack()
        .unstack()
    )
    ingrowth = ingrowth.loc[:, "root_biomass g m-2 day-1"]

    # the period is 6/10 - 9/22
    date_latest_start = (
        ingrowth.index.get_level_values("start_date").month * 100
        + ingrowth.index.get_level_values("start_date").day
    ).max()
    date_earliest_end = (
        ingrowth.index.get_level_values("end_date").month * 100
        + ingrowth.index.get_level_values("end_date").day
    ).min()

    ndays = (
        datetime(2015, int(date_earliest_end / 100), np.mod(date_earliest_end, 100))
        - datetime(2015, int(date_latest_start / 100), np.mod(date_latest_start, 100))
    ).days + 1
    ingrowth_2014_2017_mean = (ingrowth * ndays).mean(axis=0).unstack()
    ingrowth_2014_2017_std = (ingrowth * ndays).std(axis=0).unstack()

    return (
        annual_minirhizotron,
        minirhizotron_cycle_mean,
        minirhizotron_cycle_std,
        ingrowth_2014_2017_mean,
        ingrowth_2014_2017_std,
    )


def convert_sims(prefix):
    collection_ts = read_extract_sims_ts(prefix)

    # (1) annual total growth during 6/19-8/28
    annual_minirhizotron = pd.DataFrame(
        np.nan,
        index=[2015, 2018, 2019, 2020],
        columns=pd.MultiIndex.from_product(
            [["m_g_d", "g_g_d"], ["shrub", "tree"], chamber_list_complete],
            names=["variable", "pft", "plot"],
        ),
    )
    for cha in chamber_list_complete:
        for y in [2015, 2018, 2019, 2020]:
            start = datetime(y, 6, 19)
            end = datetime(y, 8, 28)
            filt = (collection_ts.index >= start) & (collection_ts.index <= end)

            temp = (
                collection_ts.loc[filt, (cha, "FROOTC_TO_LITTER", 2)] * 0.36
                + collection_ts.loc[filt, (cha, "FROOTC_TO_LITTER", 3)] * 0.14
            )
            temp = temp["hummock"] * 0.64 + temp["hollow"] * 0.36
            annual_minirhizotron.loc[y, ("m_g_d", "tree", cha)] = temp.sum() * 86400

            if 'FROOTC_ALLOC' in collection_ts.columns.levels[1]:
                temp = (
                    0.36 * collection_ts.loc[filt, (cha, "FROOTC_ALLOC", 2)]
                    + 0.14 * collection_ts.loc[filt, (cha, "FROOTC_ALLOC", 3)]
                )
            else:
                temp = (
                    0.36 * collection_ts.loc[filt, (cha, "CPOOL_TO_FROOTC", 2)]
                    + 0.14 * collection_ts.loc[filt, (cha, "CPOOL_TO_FROOTC", 3)]
                ) + (
                    0.36 * collection_ts.loc[filt, (cha, "FROOTC_XFER_TO_FROOTC", 2)]
                    + 0.14 * collection_ts.loc[filt, (cha, "FROOTC_XFER_TO_FROOTC", 3)]
                ).values
            temp = temp["hummock"] * 0.64 + temp["hollow"] * 0.36
            annual_minirhizotron.loc[y, ("g_g_d", "tree", cha)] = temp.sum() * 86400

            temp = collection_ts.loc[filt, (cha, "FROOTC_TO_LITTER", 11)] * 0.25
            temp = temp["hummock"] * 0.64 + temp["hollow"] * 0.36
            annual_minirhizotron.loc[y, ("m_g_d", "shrub", cha)] = temp.sum() * 86400

            if 'FROOTC_ALLOC' in collection_ts.columns.levels[1]:
                temp = 0.25 * collection_ts.loc[filt, (cha, "FROOTC_ALLOC", 11)]
            else:
                temp = 0.25 * collection_ts.loc[filt, (cha, "CPOOL_TO_FROOTC", 11)] + \
                       0.25 * collection_ts.loc[filt, (cha, "FROOTC_XFER_TO_FROOTC", 11)]
            temp = temp["hummock"] * 0.64 + temp["hollow"] * 0.36
            annual_minirhizotron.loc[y, ("g_g_d", "shrub", cha)] = temp.sum() * 86400

    minirhizotron_cycle = pd.DataFrame(
        0.0,
        index=pd.MultiIndex.from_tuples(
            [
                (datetime(2015, 5, 26), datetime(2015, 6, 2)),
                (datetime(2015, 6, 2), datetime(2015, 6, 11)),
                (datetime(2015, 6, 11), datetime(2015, 6, 15)),
                (datetime(2015, 6, 15), datetime(2015, 7, 1)),
                (datetime(2015, 7, 1), datetime(2015, 7, 6)),
                (datetime(2015, 7, 6), datetime(2015, 7, 13)),
                (datetime(2015, 7, 13), datetime(2015, 7, 27)),
                (datetime(2015, 7, 27), datetime(2015, 8, 4)),
                (datetime(2015, 8, 4), datetime(2015, 8, 25)),
                (datetime(2015, 8, 25), datetime(2015, 9, 10)),
                (datetime(2015, 9, 10), datetime(2015, 11, 25)),
                (datetime(2015, 11, 25), datetime(2015, 12, 3)),
                (datetime(2018, 3, 10), datetime(2018, 4, 4)),
                (datetime(2018, 4, 4), datetime(2018, 5, 3)),
                (datetime(2018, 5, 3), datetime(2018, 6, 13)),
                (datetime(2018, 6, 13), datetime(2018, 7, 22)),
                (datetime(2018, 7, 22), datetime(2018, 7, 28)),
                (datetime(2018, 7, 28), datetime(2018, 10, 18)),
            ],
            names=["start_date", "end_date"],
        ),
        columns=pd.MultiIndex.from_product(
            [["m_g_d", "g_g_d"], ["shrub", "tree"], chamber_list_complete],
            names=["variable", "pft", "plot"],
        ),
    )
    for start, end in minirhizotron_cycle.index:
        filt = (collection_ts.index >= start) & (collection_ts.index <= end)

        for cha in chamber_list_complete:
            temp = (
                collection_ts.loc[filt, (cha, "FROOTC_TO_LITTER", 2)] * 0.36
                + collection_ts.loc[filt, (cha, "FROOTC_TO_LITTER", 3)] * 0.14
            )
            temp = temp["hummock"] * 0.64 + temp["hollow"] * 0.36
            minirhizotron_cycle.loc[(start, end), ("m_g_d", "tree", cha)] = (
                temp.mean() * 86400
            )

            if 'FROOTC_ALLOC' in collection_ts.columns.levels[1]:
                temp2 = (
                    0.36 * collection_ts.loc[filt, (cha, "FROOTC_ALLOC", 2)]
                    + 0.14 * collection_ts.loc[filt, (cha, "FROOTC_ALLOC", 3)]
                )
            else:
                temp2 = (
                    0.36 * collection_ts.loc[filt, (cha, "CPOOL_TO_FROOTC", 2)]
                    + 0.14 * collection_ts.loc[filt, (cha, "CPOOL_TO_FROOTC", 3)]
                ) + (
                    0.36 * collection_ts.loc[filt, (cha, "FROOTC_XFER_TO_FROOTC", 2)]
                    + 0.14 * collection_ts.loc[filt, (cha, "FROOTC_XFER_TO_FROOTC", 3)]
                )
            temp2 = temp2["hummock"] * 0.64 + temp2["hollow"] * 0.36
            minirhizotron_cycle.loc[(start, end), ("g_g_d", "tree", cha)] = (
                temp2.mean() # + temp.mean()
            ) * 86400

            temp = collection_ts.loc[filt, (cha, "FROOTC_TO_LITTER", 11)] * 0.25
            temp = temp["hummock"] * 0.64 + temp["hollow"] * 0.36
            minirhizotron_cycle.loc[(start, end), ("m_g_d", "shrub", cha)] = (
                temp.mean() * 86400
            )

            if 'FROOTC_ALLOC' in collection_ts.columns.levels[1]:
                temp2 = 0.25 * collection_ts.loc[filt, (cha, "FROOTC_ALLOC", 11)]
            else:
                temp2 = 0.25 * collection_ts.loc[filt, (cha, "CPOOL_TO_FROOTC", 11)] + \
                    0.25 * collection_ts.loc[filt, (cha, "FROOTC_XFER_TO_FROOTC", 11)]
            temp2 = temp2["hummock"] * 0.64 + temp2["hollow"] * 0.36
            minirhizotron_cycle.loc[(start, end), ("g_g_d", "shrub", cha)] = (
                temp2.mean() # + temp.mean()
            ) * 86400

    minirhizotron_cycle = minirhizotron_cycle.stack().stack().reset_index()
    minirhizotron_cycle["year"] = [t.year for t in minirhizotron_cycle["start_date"]]
    minirhizotron_cycle["month"] = [t.month for t in minirhizotron_cycle["start_date"]]
    minirhizotron_cycle = minirhizotron_cycle.drop(["start_date", "end_date"], axis=1)
    minirhizotron_cycle = minirhizotron_cycle.set_index(
        ["year", "month", "plot", "pft"]
    )
    minirhizotron_cycle_mean = (
        minirhizotron_cycle.groupby(["month", "pft"]).mean().unstack()
    )
    minirhizotron_cycle_std = (
        minirhizotron_cycle.groupby(["month", "pft"]).std().unstack()
    )

    minirhizotron_cycle_std = minirhizotron_cycle_std / minirhizotron_cycle_mean.sum(
        axis=0, skipna=False
    )
    minirhizotron_cycle_mean = minirhizotron_cycle_mean / minirhizotron_cycle_mean.sum(
        axis=0, skipna=False
    )

    ingrowth_2014_2017_mean = pd.DataFrame(
        0.0, index=["larch", "shrub", "spruce"], columns=chamber_list_complete
    )
    ingrowth_2014_2017_std = pd.DataFrame(
        0.0, index=["larch", "shrub", "spruce"], columns=chamber_list_complete
    )
    for cha in chamber_list_complete:
        ingrowth = {"spruce": [], "larch": [], "shrub": []}
        for y in range(2014, 2018):
            start = datetime(y, 6, 10)
            end = datetime(y, 9, 22)
            filt = (collection_ts.index >= start) & (collection_ts.index <= end)

            for name, frac, pft in zip(
                ["spruce", "larch", "shrub"], [0.36, 0.14, 0.25], [2, 3, 11]
            ):
                if 'FROOTC_ALLOC' in collection_ts.columns.levels[1]:
                    temp = frac * collection_ts.loc[filt, (cha, "FROOTC_ALLOC", pft)]
                else:
                    temp = frac * collection_ts.loc[filt, (cha, "CPOOL_TO_FROOTC", pft)] + \
                        frac * collection_ts.loc[filt, (cha, "FROOTC_XFER_TO_FROOTC", pft)].values
                temp = temp["hummock"] * 0.64 + temp["hollow"] * 0.36
                ingrowth[name].append(temp.sum() * 86400)
        for name in ["spruce", "larch", "shrub"]:
            ingrowth_2014_2017_mean.loc[name, cha] = np.mean(ingrowth[name])
            ingrowth_2014_2017_std.loc[name, cha] = np.std(ingrowth[name])

    return (
        annual_minirhizotron,
        minirhizotron_cycle_mean,
        minirhizotron_cycle_std,
        ingrowth_2014_2017_mean,
        ingrowth_2014_2017_std,
    )
