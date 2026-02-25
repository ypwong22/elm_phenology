import numpy as np
import os
import pandas as pd
from datetime import datetime, timedelta
from .analysis import *
from scipy.optimize import minimize, basinhopping, differential_evolution
import itertools as it


lat, lon, tzn, elv = "47.563 N", "93.438 W", "US/Central", 430


def _find_solstice(year_list):
    # find last winter solstice
    solstice = pd.Series(index=year_list)
    for year in year_list:
        last_year = pd.DatetimeIndex(
            [
                datetime(year - 1, 12, 1) + timedelta(days=i)
                for i in range((datetime(year, 1, 1) - datetime(year - 1, 1, 1)).days)
            ]
        )

        dl_prev = 1e20
        for ind, yy, mm, dd in zip(
            range(len(last_year)), last_year.year, last_year.month, last_year.day
        ):
            dl = daylength(yy, mm, dd, lat, lon, elv, tzn)
            if dl >= dl_prev:
                break
            else:
                dl_prev = dl
        solstice[year] = last_year[ind]
    return solstice


def prepare_inputs():
    pheno_obs = read_leaf_sos()
    observations = {}
    for pft in ["EN", "DN", "SH"]:
        # Because ELM starts accumulation on winter solstice,
        # roll the dates 12.21 -> 1.1: add 11 days
        pheno_obs[pft] = pheno_obs[pft] + 11
        observations[pft] = pheno_obs[pft].stack().to_frame("doy")
        observations[pft].index.names = ["year", "site_id"]
        observations[pft] = observations[pft].reset_index()

    tsoi_collect, _ = read_obs_tsoi_daily()
    soil_layer_list = ["2m", "10cm"]
    for lyr in soil_layer_list:
        temp = tsoi_collect[lyr]
        temp.loc["2016-02-29", :] = 0.5 * (
            temp.loc["2016-02-28", :].values + temp.loc["2016-03-01", :].values
        )
        temp.loc["2020-02-29", :] = 0.5 * (
            temp.loc["2020-02-28", :].values + temp.loc["2020-03-01", :].values
        )
        tsoi_collect[lyr] = pd.DataFrame(
            temp.iloc[:-11, :].values, index=temp.index[11:], columns=temp.columns
        )

    predictors = {}
    for lyr in soil_layer_list:
        temp = tsoi_collect[lyr].loc[:, pheno_obs[pft].columns]
        temp = temp.sort_index(axis=0)

        temp = temp.stack().to_frame("temperature")
        temp.index.names = ["time", "site_id"]
        temp = temp.reset_index()

        temp["year"] = pd.DatetimeIndex(temp["time"]).year
        temp["doy"] = pd.DatetimeIndex(temp["time"]).dayofyear

        # shift back 11 days to get the correct daylength
        doy_list_correct = (
            pd.DatetimeIndex(temp["time"]) - timedelta(days=11)
        ).dayofyear
        temp["daylength"] = [
            daylength_simple(doy, lat=47.563) for doy in doy_list_correct
        ]

        predictors[lyr] = temp.drop("time", axis=1)

    return observations, predictors
