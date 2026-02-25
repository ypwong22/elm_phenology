# https://gist.github.com/anttilipp/ed3ab35258c7636d87de6499475301ce
import numpy as np
import pandas as pd
import os
from glob import glob
import xarray as xr
from .constants import *
from .paths import *
import warnings
from typing import Union, List
from scipy.stats import linregress, t
from datetime import datetime


def get_treatment_string(chamber):
    chamber_levels = {
        "07": ["TAMB", 0],
        "14": ["TAMB", 0],
        "21": ["TAMB", 0],
        "06": [0, 0],
        "19": [0, 500],
        "11": [2.25, 500],
        "20": [2.25, 0],
        "04": [4.5, 500],
        "13": [4.5, 0],
        "08": [6.75, 0],
        "16": [6.75, 500],
        "10": [9, 500],
        "17": [9, 0],
    }
    temperature, co2 = chamber_levels[f"{chamber:02g}"]
    if temperature == "TAMB":
        treatment = "TAMB"
    else:
        treatment = f"T{temperature:.2f}"
    if co2 > 0:
        treatment = treatment + "CO2"
    return treatment


def get_mossfrac(year, treatment):
    mossfrac = pd.read_excel(
        os.path.join(
            os.environ["HOME"], "Git", "phenology_elm", "Sphagnum_fraction.xlsx"
        ),
        index_col=0,
        skiprows=1,
        engine="openpyxl",
    ).drop(["plot", "Temp", "CO2"], axis=1)
    mossfrac[2015] = mossfrac[2016]
    return mossfrac.loc[treatment, year]


def read_mortality():
    """Return the data from SPRUCE_S1_Minirhizotron data_2012_For Yaoping.xls"""
    dates = pd.DatetimeIndex(
        [
            datetime(2012, 5, 16),
            datetime(2012, 5, 21),
            datetime(2012, 5, 28),
            datetime(2012, 6, 5),
            datetime(2012, 6, 12),
            datetime(2012, 6, 18),
            datetime(2012, 7, 3),
            datetime(2012, 7, 9),
            datetime(2012, 7, 17),
            datetime(2012, 7, 24),
            datetime(2012, 7, 30),
            datetime(2012, 8, 17),
            datetime(2012, 9, 1),
            datetime(2012, 9, 15),
        ]
    )
    #
    mortality = np.array(
        [
            0,
            33.4986,
            104.1712,
            27.1749,
            45.6811,
            30.8191,
            93.8463,
            49.2897,
            61.8272,
            69.4396,
            50.9765,
            158.3639,
            176.7845,
            99.4316,
        ]
    )

    days = (dates[1:] - dates[:-1]).days

    mortality = mortality[1:] / days

    date_list = []
    mort_list = []
    for i in range(len(days)):
        date_list.extend(
            list(pd.date_range(dates[i], dates[i + 1] - timedelta(days=1)))
        )
        mort_list.extend([mortality[i]] * days[i])

    mort = pd.Series(mort_list, index=date_list)
    mort_eos = mort.index[np.where(mort.cumsum() >= mort.cumsum()[-1] * 0.75)[0]][0]

    return mort, mort_eos


def daylength_simple(dayOfYear, lat):
    """
    Computes the length of the day (the time between sunrise and
    sunset) given the day of the year and latitude of the location.
    Function uses the Brock model for the computations.
    For more information see, for example,
    Forsythe et al., "A model comparison for daylength as a
    function of latitude and day of year", Ecological Modelling,
    1995.
    Parameters
    ----------
    dayOfYear : int
        The day of the year. 1 corresponds to 1st of January
        and 365 to 31st December (on a non-leap year).
    lat : float
        Latitude of the location in degrees. Positive values
        for north and negative for south.
    Returns
    -------
    d : float
        Daylength in hours.
    """
    latInRad = np.deg2rad(lat)
    declinationOfEarth = 23.45 * np.sin(np.deg2rad(360.0 * (283.0 + dayOfYear) / 365.0))
    if -np.tan(latInRad) * np.tan(np.deg2rad(declinationOfEarth)) <= -1.0:
        return 24.0
    elif -np.tan(latInRad) * np.tan(np.deg2rad(declinationOfEarth)) >= 1.0:
        return 0.0
    else:
        hourAngle = np.rad2deg(
            np.arccos(-np.tan(latInRad) * np.tan(np.deg2rad(declinationOfEarth)))
        )
        return 2.0 * hourAngle / 15.0


def daylength(year, month, day, lat, lon, elv, tzn):
    # https://stackoverflow.com/questions/54485777/finding-twilight-times-with-skyfield
    from skyfield import api, almanac
    from datetime import datetime, timedelta
    import pytz
    from skyfield.nutationlib import iau2000b

    """Build a function of time that returns the daylength.

    The function that this returns will expect a single argument that is a
    :class:`~skyfield.timelib.Time` and will return ``True`` if the sun is up
    or twilight has started, else ``False``.
    """
    DAYLENGTH_CENTER_HORIZON = 0.0
    DAYLENGTH_TOP_HORIZON = 0.26667
    DAYLENGTH_TOP_HORIZON_APPARENTLY = 0.8333
    DAYLENGTH_CIVIL_TWILIGHT = 6.0
    DAYLENGTH_NAUTICAL_TWILIGHT = 12.0
    DAYLENGTH_ASTRONOMICAL_TWILIGHT = 18.0

    planets = api.load("de421.bsp")
    topos = api.Topos(lat, lon, elevation_m=elv)
    ts = api.load.timescale()
    tz = pytz.timezone(tzn)

    sun = planets["sun"]
    topos_at = (planets["earth"] + topos).at

    t0 = ts.utc(datetime(year, month, day, tzinfo=tz))
    t1 = ts.utc(tz.normalize(datetime(year, month, day, tzinfo=tz) + timedelta(1)))

    def is_sun_up_at(t):
        """Return `True` if the sun has risen by time `t`."""
        t._nutation_angles = iau2000b(t.tt)
        return (
            topos_at(t).observe(sun).apparent().altaz()[0].degrees
            > -DAYLENGTH_TOP_HORIZON_APPARENTLY
        )

    is_sun_up_at.rough_period = 0.5  # twice a day

    center_time, _ = almanac.find_discrete(t0, t1, is_sun_up_at)
    up, down = center_time.utc_iso()
    daylength = datetime.strptime(down, "%Y-%m-%dT%H:%M:%SZ") - datetime.strptime(
        up, "%Y-%m-%dT%H:%M:%SZ"
    )
    return daylength.total_seconds()


def kge(simulations, evaluation):
    """
    Stolen from the hydroeval package

    Original Kling-Gupta Efficiency (KGE) and its three components
    (r, α, β) as per `Gupta et al., 2009
    <https://doi.org/10.1016/j.jhydrol.2009.08.003>`_.

    Note, all four values KGE, r, α, β are returned, in this order.

    :Calculation Details:
        .. math::
           E_{\\text{KGE}} = 1 - \\sqrt{[r - 1]^2 + [\\alpha - 1]^2
           + [\\beta - 1]^2}
        .. math::
           r = \\frac{\\text{cov}(e, s)}{\\sigma({e}) \\cdot \\sigma(s)}
        .. math::
           \\alpha = \\frac{\\sigma(s)}{\\sigma(e)}
        .. math::
           \\beta = \\frac{\\mu(s)}{\\mu(e)}

        where *e* is the *evaluation* series, *s* is (one of) the
        *simulations* series, *cov* is the covariance, *σ* is the
        standard deviation, and *μ* is the arithmetic mean.

    """
    # calculate error in timing and dynamics r
    # (Pearson's correlation coefficient)
    sim_mean = np.mean(simulations, axis=0, dtype=np.float64)
    obs_mean = np.mean(evaluation, dtype=np.float64)

    r_num = np.sum(
        (simulations - sim_mean) * (evaluation - obs_mean), axis=0, dtype=np.float64
    )
    r_den = np.sqrt(
        np.sum((simulations - sim_mean) ** 2, axis=0, dtype=np.float64)
        * np.sum((evaluation - obs_mean) ** 2, dtype=np.float64)
    )
    r = r_num / r_den
    # calculate error in spread of flow alpha
    alpha = np.std(simulations, axis=0) / np.std(evaluation, dtype=np.float64)
    # calculate error in volume beta (bias of mean discharge)
    beta = np.sum(simulations, axis=0, dtype=np.float64) / np.sum(
        evaluation, dtype=np.float64
    )
    # calculate the Kling-Gupta Efficiency KGE
    kge_ = 1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)

    return np.vstack((kge_, r, alpha, beta))


def extract_sims(prefix, var_list={"pft": [], "col": [], "const": []}, ensemble_id = None):
    pft_list = [2, 3, 11, 12]
    hol_add = 17

    collection_ts = {}
    collection_const = pd.DataFrame(
        np.nan, index=["hummock", "hollow"], columns=var_list["const"]
    )

    for plot in chamber_list_complete:
        if ensemble_id is None:
            tvec = pd.date_range("2015-01-01", "2023-12-31", freq="1D")
            tvec = tvec[(tvec.month != 2) | (tvec.day != 29)]
            path_data = os.path.join(os.environ["E3SM_ROOT"],
                "output", f"{prefix}_US-SPR_ICB20TRCNPRDCTCBC",
                "spruce_treatments", f'plot{plot:02d}_US-SPR_ICB20TRCNPRDCTCBC', 'run')
        else:
            tvec = pd.date_range("2015-01-01", "2021-12-31", freq="1D")
            tvec = tvec[(tvec.month != 2) | (tvec.day != 29)]
            path_data = os.path.join(os.environ["E3SM_ROOT"], "output", "UQ", 
                                     f"{prefix}_US-SPR_ICB20TRCNPRDCTCBC", f"g{ensemble_id:05g}",
                                     chamber_list_complete_dict[f'P{plot:02d}'])

        flist_pft = sorted(glob(os.path.join(path_data, "*.h2.*.nc")))[:-1]
        flist_col = sorted(glob(os.path.join(path_data, "*.h1.*.nc")))[:-1]
        flist_const = sorted(glob(os.path.join(path_data, "*.h0.*.nc")))[:-1]

        if ensemble_id is None:
            print(f'Plot {plot}', chamber_list_complete_dict[f'P{plot:02d}'], 
                  len(flist_pft), len(flist_col), len(flist_const))
        else:
            print(f'Plot {plot}', f"g{ensemble_id:05g}", 
                  chamber_list_complete_dict[f'P{plot:02d}'], 
                  len(flist_pft), len(flist_col), len(flist_const))
 
        var_list_pft = var_list["pft"]
        hr = xr.open_mfdataset(flist_pft, decode_times=False)
        for var in var_list_pft:
            for pft in pft_list:
                if var == 'TOTVEGC':
                    # need to subtract out CPOOL to be comparable to obs
                    collection_ts[(plot, var, pft, "hummock")] = hr[var][:, pft].values - \
                        hr['CPOOL'][:, pft].values
                    collection_ts[(plot, var, pft, "hollow")] = \
                        hr[var][:, pft + hol_add].values - \
                        hr['CPOOL'][:, pft + hol_add].values
                elif 'ROOTFR' in var:
                    # root fraction weighted variable
                    var_ = '_'.join(var.split('_')[:-1])

                    rootfr = hr["ROOTFR"][:, :, pft].values
                    nlev = np.max(np.where(rootfr[0, :] > 0)[0])
                    nlev = nlev + 1 # the indexation needs to be 1+ the last index

                    if var_ in ['TSOI','H2OSOI','SMINN_vr','SOLUTIONP_vr',
                                'SMIN_NH4_vr', 'SMIN_NO3_vr', 
                                'LITR1C_vr','LITR2C_vr','LITR3C_vr',
                                'LITR1N_vr','LITR2N_vr','LITR3N_vr',
                                'LITR1P_vr','LITR2P_vr','LITR3P_vr']:
                        # column variables
                        hr2 = xr.open_mfdataset(flist_col)
                        collection_ts[(plot, var, pft, "hummock")] = np.sum(
                            hr2[var_][:,:nlev,0].values * rootfr[:,:nlev], axis=1
                        )
                        collection_ts[(plot, var, pft, "hollow")] = np.sum(
                            hr2[var_][:,:nlev,1].values * rootfr[:,:nlev], axis=1
                        )
                        hr2.close()

                        if var_ == 'TSOI':
                            collection_ts[(plot, var, pft, "hummock")] = \
                                collection_ts[(plot, var, pft, "hummock")] - 273.15
                            collection_ts[(plot, var, pft, "hollow")] = \
                                collection_ts[(plot, var, pft, "hollow")] - 273.15
                    else:
                        # pft-specific variables, (time, levdcmp, pft)
                        collection_ts[(plot, var, pft, "hummock")] = np.sum(
                            hr[var_][:, :nlev, pft].values * rootfr[:,:nlev], axis=1
                        )
                        collection_ts[(plot, var, pft, "hollow")] = np.sum(
                            hr[var_][:, :nlev, pft+hol_add].values * rootfr[:,:nlev], axis=1
                        )
                else:
                    collection_ts[(plot, var, pft, "hummock")] = hr[var][:, pft].values
                    collection_ts[(plot, var, pft, "hollow")] = hr[var][:, pft + hol_add].values
        hr.close()

        var_list_col = var_list["col"]
        hr = xr.open_mfdataset(flist_col, decode_times=False)
        for var in var_list_col:
            if 'TSOI_' in var or 'H2OSOI_' in var or 'SMINN_' in var or 'SOLUTIONP_' in var:
                # layer thickness weighted variable

                LEVGRND = np.array([0.007100635, 0.027925, 0.06225858, 0.1188651, 0.2121934,
                                    0.3660658, 0.6197585, 1.038027, 1.727635, 2.864607, 4.739157,
                                    7.829766, 12.92532, 21.32647, 35.17762])
                LEVGRND_I = np.append(np.insert(
                    (LEVGRND[1:] + LEVGRND[:-1])*0.5, 0, 0
                ), LEVGRND[-1] + 0.5 * (LEVGRND[-1] - LEVGRND[-2]))
                THICKNESS = np.diff(LEVGRND_I)

                depth = float(var.split('_')[1]) / 100.
                maxlayer = np.where(LEVGRND_I < depth)[0][-1]

                if 'TSOI_' in var:
                    thisvar = 'TSOI'
                elif 'H2OSOI_' in var:
                    thisvar = 'H2OSOI'
                elif 'SMINN_' in var:
                    thisvar = 'SMINN_vr'
                elif 'SMIN_NH4_' in var:
                    thisvar = 'SMIN_NH4_vr'
                elif 'SMIN_NO3_' in var:
                    thisvar = 'SMIN_NO3_vr'
                elif 'SOLUTIONP_' in var:
                    thisvar = 'SOLUTIONP_vr'

                data = np.zeros([hr[thisvar].shape[0], 2])
                for i in range(maxlayer):
                    data = data + hr[thisvar][:, i, :] * THICKNESS[i]
                last_depth = min(THICKNESS[i], depth - LEVGRND_I[maxlayer])
                data = data + hr[thisvar][:, maxlayer, :] * last_depth
                data = data / depth

                collection_ts[(plot, var, 0, "hummock")] = data.values[:, 0]
                collection_ts[(plot, var, 0, "hollow")] = data.values[:, 1]

                if 'TSOI' in var:
                    collection_ts[(plot, var, 0, "hummock")] -= 273.15
                    collection_ts[(plot, var, 0, "hollow")] -= 273.15

            elif var == "SMP_MAX":
                collection_ts[(plot, var, 0, "hummock")] = np.nanmax(
                    hr["SMP"][:, :, 0].values, axis=1
                )  # mm
                collection_ts[(plot, var, 0, "hollow")] = np.nanmax(
                    hr["SMP"][:, :, 1].values, axis=1
                )

            elif var == "ZWT":
                hr2 = xr.open_mfdataset(flist_col)
                collection_ts[(plot, var, 0, "hummock")] = (
                    0.15 - hr2["ZWT"][:, 0].values
                )
                collection_ts[(plot, var, 0, "hollow")] = (
                    hr2["H2OSFC"][:, 1] / 1000.0 - hr2["ZWT"][:, 1]
                )
                hr2.close()

            else:
                collection_ts[(plot, var, 0, "hummock")] = hr[var][:, 0].values
                collection_ts[(plot, var, 0, "hollow")] = hr[var][:, 1].values
                if var == 'TBOT':
                    collection_ts[(plot, var, 0, "hummock")] -= 273.15
                    collection_ts[(plot, var, 0, "hollow")] -= 273.15
        hr.close()

        # some time-constant variables are in the h0 files
        var_list_const = var_list["const"]
        hr = xr.open_mfdataset(flist_const, decode_times=False)
        for var in var_list_const:
            if var == "SUCSAT":
                collection_const.loc[:, "SUCSAT"] = pd.Series(
                    hr["SUCSAT"][2, :].values, index=["hummock", "hollow"]
                )
            elif var == "WATSAT":
                collection_const.loc[:, "WATSAT"] = pd.Series(
                    hr["WATSAT"][2, :].values, index=["hummock", "hollow"]
                )
            elif var == "BSW":
                collection_const.loc[:, "BSW"] = pd.Series(
                    hr["BSW"][2, :].values, index=["hummock", "hollow"]
                )
            else:
                raise Exception("Not implemented")
        hr.close()

    collection_ts = pd.DataFrame(collection_ts)
    collection_ts.index = tvec
    collection_ts.columns.names = ["plot", "variable", "pft", "topo"]

    return collection_ts, collection_const


def extract_xys(var_list={"pft": [], "col": [], "const": []}):
    tvec = pd.date_range("2015-01-01", "2020-12-31", freq="1D")
    tvec = tvec[(tvec.month != 2) | (tvec.day != 29)]

    pft_list = [2, 3, 11, 12]
    hol_add = 17

    collection_ts = {}
    collection_const = pd.DataFrame(
        np.nan, index=["hummock", "hollow"], columns=var_list["const"]
    )

    count = 0
    for plot, plot_name in zip(chamber_list_complete, chamber_list_names_complete):
        count = count + 1

        if count >= 2:
            continue

        path_data = os.path.join(
            os.environ["SCRATCH"],
            "../y9s",
            f"20230414_spruceroot_{plot_name}_US-SPR_ICB20TRCNPRDCTCBC",
            "run",
        )
        print(path_data)
        flist_pft = sorted(glob(os.path.join(path_data, "*.h3.*.nc")))[:-1, :]
        flist_col = sorted(glob(os.path.join(path_data, "*.h2.*.nc")))[:-1, :]
        flist_const = sorted(glob(os.path.join(path_data, "*.h0.*.nc")))

        # skip 2021
        flist_pft = [p for p in flist_pft if "2021" not in p]
        flist_col = [p for p in flist_col if "2021" not in p]
        flist_const = [p for p in flist_const if "2021" not in p]

        var_list_pft = var_list["pft"]
        hr = xr.open_mfdataset(flist_pft, decode_times=False)
        for var in var_list_pft:
            for pft in pft_list:
                if var == "TSOI_AVG":
                    rootfr = hr["ROOTFR"][:, :, pft].values
                    hr2 = xr.open_mfdataset(flist_col)
                    collection_ts[(plot, var, pft, "hummock")] = np.sum(
                        (hr2["TSOI"][:, :, 0].values - 273.15) * rootfr, axis=1
                    )
                    collection_ts[(plot, var, pft, "hollow")] = np.sum(
                        (hr2["TSOI"][:, :, 1].values - 273.15) * rootfr, axis=1
                    )
                    hr2.close()
                elif var == "SWC_AVG":
                    rootfr = hr["ROOTFR"][:, :, pft].values
                    hr2 = xr.open_dataset(flist_const[0])
                    dzsoi = hr2["DZSOI"].values
                    hr2.close()
                    hr2 = xr.open_mfdataset(flist_col)
                    collection_ts[(plot, var, pft, "hummock")] = np.sum(
                        hr2["SOILLIQ"][:, :, 0].values
                        / dzsoi[:, 0].reshape(1, -1)
                        / 1000
                        * rootfr,
                        axis=1,
                    )
                    collection_ts[(plot, var, pft, "hollow")] = np.sum(
                        hr2["SOILLIQ"][:, :, 1].values
                        / dzsoi[:, 1].reshape(1, -1)
                        / 1000
                        * rootfr,
                        axis=1,
                    )
                    hr2.close()
                elif var == "ZWT":
                    hr2 = xr.open_mfdataset(flist_col)
                    collection_ts[(plot, var, pft, "hummock")] = (
                        0.3 - hr2["ZWT"][:, 0].values
                    )
                    collection_ts[(plot, var, pft, "hollow")] = (
                        hr2["H2OSFC"][:, 1] / 1000.0 - hr2["ZWT"][:, 1]
                    )
                    hr2.close()
                else:
                    collection_ts[(plot, var, pft, "hummock")] = hr[var][:, pft].values
                    collection_ts[(plot, var, pft, "hollow")] = hr[var][
                        :, pft + hol_add
                    ].values
        hr.close()

        var_list_col = var_list["col"]
        hr = xr.open_mfdataset(flist_col, decode_times=False)
        for var in var_list_col:
            if var.startswith("TSOI_"):
                layer = int(var.split("_")[1])
                collection_ts[(plot, var, 0, "hummock")] = (
                    hr["TSOI"][:, layer - 1, 0].values - 273.15
                )
                collection_ts[(plot, var, 0, "hollow")] = (
                    hr["TSOI"][:, layer - 1, 1].values - 273.15
                )
            elif var.startswith("SWC_"):
                layer = int(var.split("_")[1])
                hr2 = xr.open_dataset(flist_const[0])
                dzsoi = hr2["DZSOI"].values
                hr2.close()
                collection_ts[(plot, var, 0, "hummock")] = (
                    hr["SOILLIQ"][:, layer - 1, 0].values / dzsoi[layer - 1, 0] / 1000
                )
                collection_ts[(plot, var, 0, "hollow")] = (
                    hr["SOILLIQ"][:, layer - 1, 1].values / dzsoi[layer - 1, 1] / 1000
                )
            elif var.startswith("H2OSOI_"):
                layer = int(var.split("_")[1])
                collection_ts[(plot, var, 0, "hummock")] = hr["H2OSOI"][
                    :, layer - 1, 0
                ].values
                collection_ts[(plot, var, 0, "hollow")] = hr["H2OSOI"][
                    :, layer - 1, 1
                ].values
            elif var == "SMP_MAX":
                collection_ts[(plot, var, 0, "hummock")] = np.nanmax(
                    hr["SMP"][:, :, 0].values, axis=1
                )  # mm
                collection_ts[(plot, var, 0, "hollow")] = np.nanmax(
                    hr["SMP"][:, :, 1].values, axis=1
                )
            else:
                collection_ts[(plot, var, 0, "hummock")] = hr[var][:, 0].values
                collection_ts[(plot, var, 0, "hollow")] = hr[var][:, 1].values
        hr.close()

        # some time-constant variables are in the h0 files
        var_list_const = var_list["const"]
        hr = xr.open_mfdataset(flist_const, decode_times=False)
        for var in var_list_const:
            if var == "SUCSAT":
                collection_const.loc[:, "SUCSAT"] = pd.Series(
                    hr["SUCSAT"][2, :].values, index=["hummock", "hollow"]
                )
            elif var == "WATSAT":
                collection_const.loc[:, "WATSAT"] = pd.Series(
                    hr["WATSAT"][2, :].values, index=["hummock", "hollow"]
                )
            elif var == "BSW":
                collection_const.loc[:, "BSW"] = pd.Series(
                    hr["BSW"][2, :].values, index=["hummock", "hollow"]
                )
            else:
                raise Exception("Not implemented")
        hr.close()

    collection_ts = pd.DataFrame(collection_ts)
    collection_ts.index = tvec
    collection_ts.columns.names = ["plot", "variable", "pft", "topo"]

    return collection_ts, collection_const


def read_extract_sims_ts(prefix):
    collection_ts = pd.read_csv(
        os.path.join(path_out, "extract", prefix, "analysis_ts.csv"),
        index_col=0,
        header=[0, 1, 2, 3],
        parse_dates=True,
    )
    indices = collection_ts.columns.to_list()
    indices = [(int(i), j, int(k), l) for i, j, k, l in indices]
    collection_ts.columns = pd.MultiIndex.from_tuples(
        indices, names=collection_ts.columns.names
    )
    return collection_ts


def read_sims_tair_daily():
    prefix = "20231113"  # identical for any
    collection_ts = read_extract_sims_ts(prefix)
    temperature = (
        0.64 * collection_ts.loc[:, (slice(None), "TBOT", 0, "hummock")]
        + 0.36 * collection_ts.loc[:, (slice(None), "TBOT", 0, "hollow")].values
    )
    temperature.columns = temperature.columns.droplevel([1, 2, 3])
    return temperature


def read_sims_tair_annual():
    temperature = read_sims_tair_daily()
    temperature = temperature.groupby(temperature.index.year).mean()
    return temperature


def read_obs_tair_annual():
    obs_data = pd.read_excel(
        os.path.join(path_input, "SPRUCE C Budget Summary 28Apr2022EXP.xlsx"),
        sheet_name="DataForPythonRead",
        skiprows=1,
        engine="openpyxl",
    )
    obs_data = obs_data.set_index(["Year", "Plot"]).sort_index(axis=0)
    obs_data = obs_data.sort_index()
    t2m_obs = obs_data.loc[:, "Mean Annual Temp. at 2 m"].unstack()
    t2m_obs.columns = [int(p[1:]) for p in t2m_obs.columns]
    t2m_obs = t2m_obs.loc[:, chamber_list]
    t2m_obs.loc[2015, :] = np.nan
    t2m_obs = t2m_obs.drop(2021, axis=0)
    t2m_obs = t2m_obs.sort_index(axis=0)
    return t2m_obs


def read_leaf_sos():
    pheno_obs = {}
    for var in ["EN", "DN", "SH"]:
        hr = xr.open_dataset(os.path.join(path_intrim, "spruce_validation_data.nc"))
        temp = hr["pheno_dates_lai"].loc[:, "SOS", :, var]
        temp = pd.DataFrame(
            temp.values,
            index=temp["year"],
            columns=["%02d" % i for i in temp["chamber"]],
        )
        pheno_obs[var] = (
            temp.dropna(axis=1, how="all").dropna(axis=0, how="all").sort_index(axis=1)
        )
        hr.close()
    return pheno_obs


def read_leaf_eos():
    pheno_obs = {}
    for var in ["EN", "DN", "SH"]:
        hr = xr.open_dataset(os.path.join(path_intrim, "spruce_validation_data.nc"))
        temp = hr["pheno_dates_lai"].loc[:, "EOS", :, var]
        temp = pd.DataFrame(
            temp.values,
            index=temp["year"],
            columns=["%02d" % i for i in temp["chamber"]],
        )
        pheno_obs[var] = (
            temp.dropna(axis=1, how="all").dropna(axis=0, how="all").sort_index(axis=1)
        )
        hr.close()
    return pheno_obs


def read_obs_tsoi_daily():
    tvec = pd.date_range("2015-01-01", "2021-12-31", freq="1D")
    tvec = tvec[~((tvec.month == 2) & (tvec.day == 29))]

    annt2m = {}
    tsoi_collect = {}  # '2m', '10cm'
    for fid in chamber_levels.keys():
        env = pd.read_csv(
            os.path.join(
                path_input,
                "WEW_Complete_Environ_20220518",
                "WEW PLOT_{}_Complete_Environ_20220518.csv".format(fid),
            )
        )
        env = env.loc[(env["Year"] >= 2015) & (env["Year"] <= 2021), :]
        env.index = pd.DatetimeIndex(env["TIMESTAMP"])
        env = env.replace(to_replace="^\s+", value=np.nan, regex=True).sort_index()
        env = env.loc[~env.index.duplicated(keep="first"), :]

        # These do not match ELM input, use ELM input
        tseries = (env["TA_2_0__1"].astype(float) + env["TA_2_0__2"].astype(float)) / 2
        ##tmax      = tseries.groupby(env.index.year * 1000 + env.index.month * 100 + env.index.day).max()
        ##tmin      = tseries.groupby(env.index.year * 1000 + env.index.month * 100 + env.index.day).min()
        ##tavg      = tseries.groupby(env.index.year * 1000 + env.index.month * 100 + env.index.day).mean()
        annt2m[fid] = tseries.groupby(env.index.year).mean()
        tseries = tseries.resample("1d").mean()
        tseries = tseries.loc[~((tseries.index.month == 2) & (tseries.index.day == 29))]

        # Average hollow & hummock @ 10cm below surface
        temp = (
            env.loc[:, ["TS_ 10__A3", "TS_ 10__B3", "TS_ 10__C3"]].mean(axis=1) * 0.64
            + env.loc[:, ["TS_Hummock_A2", "TS_Hummock_B2"]].mean(axis=1).values * 0.36
        )
        temp = temp.astype(float)
        # has NaNs if keep hourly level
        temp = temp.resample("1d").mean()
        temp = temp.loc[~((temp.index.month == 2) & (temp.index.day == 29))]

        tsoi = pd.DataFrame(np.nan, index=tvec, columns=["2m", "10cm"])
        tsoi.loc[temp.index, "10cm"] = temp.values

        # check: tseries.loc[(tsoi.index[0] - timedelta(days = 119, hours = 23, minutes = 30)):(tsoi.index[0])].mean()
        tair = tseries.rolling("21d").mean()
        tsoi.loc[tair.index, "2m"] = tair.values

        # fill NaNs
        for col in tsoi.columns:
            tsoi.loc[:, col] = tsoi.loc[:, col].interpolate(
                limit=5, limit_direction="both"
            )

        # skip NaNs in the beginning
        tsoi = tsoi.loc[tsoi.index >= datetime(2015, 11, 1), :]

        narows = np.where(tsoi.isna().any(axis=1))[0]
        if len(narows) > 0:
            print(fid, tsoi["10cm"][narows])
            import pdb

            pdb.set_trace()
            raise Exception("check")

        tsoi = dict(tsoi.dropna(axis=0, how="any"))

        for k in tsoi.keys():
            if k not in tsoi_collect:
                tsoi_collect[k] = pd.DataFrame({fid: tsoi[k]})
            else:
                tsoi_collect[k][fid] = tsoi[k]
    annt2m = pd.DataFrame(annt2m)
    annt2m = annt2m.sort_index(axis=1)
    for k in tsoi_collect.keys():
        tsoi_collect[k] = tsoi_collect[k].sort_index(axis=1)

    return tsoi_collect, annt2m


########################################
# Given folder path, get the simulated values matched to the observed values
########################################
def get_sim_carbonfluxes(year_range, runroot, growing_season, extra_pft_vars = [], 
                         extra_col_vars = []):
    warnings.filterwarnings("ignore")

    mossfrac = pd.read_excel(
        os.path.join(os.environ['HOME'], 'Git', 'phenology_elm', "Sphagnum_fraction.xlsx"),
        index_col=0, skiprows=1, engine="openpyxl"
    ).drop(["plot", "Temp", "CO2"], axis=1)
    mossfrac[2015] = mossfrac[2016]

    var_list = ['Tair', 'AGBiomass_Spruce', 'AGBiomass_Tamarack', 'AGBiomass_Shrub',
                'AGNPPtoBiomass_Spruce', 'AGNPPtoBiomass_Tamarack', 'AGNPPtoBiomass_Shrub',
                'AGNPP_Spruce', 'AGNPP_Tamarack', 'AGNPP_Shrub', 'NPP_moss',
                'BGNPP_TreeShrub', 'BGtoAG_TreeShrub', 'NPP', 'HR', 'NEE']

    grid_to_plot = {"T0.00": "P06", "T2.25": "P20", "T4.50": "P13",
        "T6.75": "P08", "T9.00": "P17", "T0.00CO2": "P19", "T2.25CO2": "P11",
        "T4.50CO2": "P04", "T6.75CO2": "P16", "T9.00CO2": "P10", "TAMB": "P07"}
    plot_to_grid = dict([(b,a) for a,b in grid_to_plot.items()])

    pft_stride = 17
    plot_list = ['P04', 'P06', 'P07', 'P08', 'P10', 'P11', 'P13', 'P16', 'P17', 'P19', 'P20']

    collect = pd.DataFrame(np.nan,
        index = pd.MultiIndex.from_product([['hummock', 'hollow', 'average'],
                                            plot_list,
                                            year_range], names = ['column', 'plot', 'year']),
        columns = var_list + extra_col_vars + [f'{v}_Spruce' for v in extra_pft_vars] + \
             [f'{v}_Tamarack' for v in extra_pft_vars] + [f'{v}_Shrub' for v in extra_pft_vars])

    for plot in plot_list:
        if not "UQ" in runroot:
            temp = plot.replace("P", '')
            rundir = os.path.join(runroot, f'plot{temp}_US-SPR_ICB20TRCNPRDCTCBC', 'run')
            if not os.path.exists(rundir):
                # try using descriptive label
                rundir = glob(os.path.join(runroot, 
                    f'*{chamber_list_complete_dict[plot]}_US-SPR_ICB20TRCNPRDCTCBC', 'run'))[0]
        else:
            temp = chamber_list_complete_dict[plot]
            rundir = os.path.join(runroot, temp)

        flist_pft = sorted(glob(rundir + "/*.h2.*.nc"))
        flist_pft = [f for f in flist_pft if \
                     int(f.split('/')[-1].split('.')[-2].split('-')[0]) in year_range]

        hr = xr.open_mfdataset(flist_pft)
        flist_col = sorted(glob(rundir + "/*.h1.*.nc"))
        flist_col = [f for f in flist_col if \
                     int(f.split('/')[-1].split('.')[-2].split('-')[0]) in year_range]

        hr2 = xr.open_mfdataset(flist_col)

        if growing_season:
            filter = (hr['time'].to_index().month >= 5) & (hr['time'].to_index().month <= 10)

        ##################################################################
        # PFT_specific variables in VAR_LIST
        ##################################################################
        # hummock: 0.64, hollow: 0.36
        # pima: 0.36, lala: 0.14
        # temporary fix until better values become available
        if growing_season:
            temp = hr['AGNPP'][filter, :].resample({'time': '1Y'}).mean() * 365 * 86400
        else:
            temp = hr['AGNPP'][:-1, :].resample({'time': '1Y'}).mean() * 365 * 86400
        # convert gC/m2/s to gC/m2/year; otherwise gC m-2
        for col, add in zip(['hummock', 'hollow'], [0, pft_stride]):
            collect.loc[(col, plot), f'AGNPP_Spruce'] = temp[:, 2 + add] * 0.36
            collect.loc[(col, plot), f'AGNPP_Tamarack'] = temp[:, 3 + add] * 0.14
            collect.loc[(col, plot), f'AGNPP_Shrub'] = temp[:, 11 + add] * 0.25

        if not 'y9s' in runroot:
            # temporary fix until better values become available
            if growing_season:
                temp = hr['TOTVEGC_ABG'][filter, :].resample({'time': '1Y'}).mean()
            else:
                temp = hr['TOTVEGC_ABG'][:-1, :].resample({'time': '1Y'}).mean()
            temp = temp.values
            for col, add in zip(['hummock', 'hollow'], [0, pft_stride]):
                collect.loc[(col, plot), f'AGBiomass_Spruce'] = temp[:, 2 + add] * 0.36
                collect.loc[(col, plot), f'AGBiomass_Tamarack'] = temp[:, 3 + add] * 0.14
                collect.loc[(col, plot), f'AGBiomass_Shrub'] = temp[:, 11 + add] * 0.25

        for col, add in zip(['hummock', 'hollow'], [0, pft_stride]):
            for pft in ['Spruce', 'Tamarack', 'Shrub']:
                collect.loc[(col, plot), f'AGNPPtoBiomass_{pft}'] = \
                    collect.loc[(col, plot), f'AGNPP_{pft}'].values / \
                    collect.loc[(col, plot), f'AGBiomass_{pft}'].values

        # BGNPP_TreeShrub is purely fine root
        for col, add in zip(['hummock', 'hollow'], [0, pft_stride]):
            # temporary fix until better values become available
            if growing_season:
                temp = (hr['FROOTC_ALLOC'])[filter, :].resample({'time': '1Y'}).mean()
            else:
                temp = (hr['FROOTC_ALLOC'])[:-1, :].resample({'time': '1Y'}).mean()
            # convert gC/m2/s to gC/m2/year; otherwise gC m-2
            temp = temp.values * 365 * 86400
            collect.loc[(col, plot), 'BGNPP_TreeShrub'] = \
                temp[:, 2 + add] * 0.36 + temp[:, 3 + add] * 0.14 + temp[:, 11 + add] * 0.25

        for col, add in zip(['hummock', 'hollow'], [0, pft_stride]):
            collect.loc[(col, plot), 'BGtoAG_TreeShrub'] = \
                collect.loc[(col, plot), 'BGNPP_TreeShrub'].values / \
                (collect.loc[(col, plot), 'AGNPP_Spruce'] + \
                collect.loc[(col, plot), 'AGNPP_Tamarack'] + \
                collect.loc[(col, plot), 'AGNPP_Shrub']).values

        if growing_season:
            temp = hr['NPP'][filter, :].resample({'time': '1Y'}).mean()
        else:
            temp = hr['NPP'][:-1, :].resample({'time': '1Y'}).mean()
        # convert gC/m2/s to gC/m2/year
        temp = temp * 365 * 86400
        for col, add in zip(['hummock', 'hollow'], [0, pft_stride]):
            collect.loc[(col, plot), 'NPP_moss'] = temp[:, 12 + add] * \
                mossfrac.loc[plot_to_grid[plot], :].loc[year_range] / 100.

        #if growing_season:
        #    temp = hr['GPP'][filter, :].resample({'time': '1Y'}).mean()
        #else:
        #    temp = hr['GPP'][:-1, :].resample({'time': '1Y'}).mean()
        ## convert gC/m2/s to gC/m2/year
        #temp = temp * 365 * 86400
        #for col, add in zip(['hummock', 'hollow'], [0, pft_stride]):
        #    collect.loc[(col, plot), 'GPP_moss'] = temp[:, 12 + add] * \
        #        mossfrac.loc[plot_to_grid[plot], :].loc[year_range] / 100.

        ##################################################################
        # PFT variables in extra_pft_vars
        ##################################################################
        for var in extra_pft_vars:
            # temporary fix until better values become available
            if 'LEAFC_ALLOC_TO_TOTVEGC_ABG' in var and not 'y9s' in runroot:
                if growing_season:
                    temp = hr['LEAFC_ALLOC'][filter, :].resample({'time': '1Y'}).mean() / \
                           hr['TOTVEGC_ABG'][filter, :].resample({'time': '1Y'}).mean() * \
                           365 * 86400
                else:
                    temp = hr['LEAFC_ALLOC'][:-1, :].resample({'time': '1Y'}).mean() / \
                           hr['TOTVEGC_ABG'][:-1, :].resample({'time': '1Y'}).mean() * \
                           365 * 86400
            if var == 'AGNPP':
                # convert gC/m2/s to gC/m2/year; otherwise gC m-2
                temp = temp.values * 365 * 86400
            else:
                temp = temp.values
            for col, add in zip(['hummock', 'hollow'], [0, pft_stride]):
                collect.loc[(col, plot), f'{var}_Spruce'] = temp[:, 2 + add] * 0.36
                collect.loc[(col, plot), f'{var}_Tamarack'] = temp[:, 3 + add] * 0.14
                collect.loc[(col, plot), f'{var}_Shrub'] = temp[:, 11 + add] * 0.25

        ##################################################################
        # Column variables in VAR_LIST
        ##################################################################
        col_list = ['NPP','TBOT', 'NEE', 'HR']
        for colvar in col_list: # 'FCH4'
            if growing_season:
                if colvar == 'NPP':
                    temp = (hr2['AGNPP'] + hr2['FROOTC_ALLOC'].values)[filter, :].resample({'time': '1Y'}).mean().values
                else:
                    temp = hr2[colvar][filter, :].resample({'time': '1Y'}).mean().values
            else:
                if colvar == 'NPP':
                    temp = (hr2['AGNPP'] + hr2['FROOTC_ALLOC'].values)[:-1, :].resample({'time': '1Y'}).mean().values
                else:
                    temp = hr2[colvar][:-1, :].resample({'time': '1Y'}).mean().values
            if colvar in ['NPP','NEE', 'HR']:
                # convert gC/m2/s to gC/m2/year
                temp = temp * 365 * 86400
            for num, col in enumerate(['hummock', 'hollow']):
                if colvar == 'FCH4':
                    collect.loc[(col, plot), 'CH4'] = temp[:, num]
                elif colvar == 'TBOT':
                    collect.loc[(col, plot), 'Tair'] = temp[:, num] - 273.15
                else:
                    collect.loc[(col, plot), colvar] = temp[:, num]

        ##################################################################
        # Column variables in extra_col_vars
        ##################################################################
        for colvar in extra_col_vars:
            if growing_season:
                temp = hr2[colvar][filter, :].resample({'time': '1Y'}).mean().values
            else:
                temp = hr2[colvar][:-1, :].resample({'time': '1Y'}).mean().values
            for num, col in enumerate(['hummock', 'hollow']):
                collect.loc[(col, plot), colvar] = temp[:, num]

        hr.close()
        hr2.close()

    temp = (collect.loc['hummock', :] * 0.64 + collect.loc['hollow' , :] * 0.36)
    for ind, row in temp.iterrows():
        collect.loc[('average', *ind), :] = row.values

    warnings.filterwarnings("default")

    return collect


def vert_interp(
    target_nodes: Union[List[float], np.ndarray],
    input_nodes: Union[List[float], np.ndarray],
    input_data: np.ndarray,
    target_single_level: bool = False,
    target_interfaces: Union[List[float], np.ndarray, None] = None,
    input_interfaces: Union[List[float], np.ndarray, None] = None,
) -> np.ndarray:
    """
    Linearly interpolate soil moisture/soil temperature from input_nodes to target_nodes. 
    If the target depths are single-level, returns weighted average based on the distance 
        between the input nodes and target node. 
    If the target depths are defined by bounds (target_interfaces != None), returns weighted
        average based on the overlapping lengths between the target_interfaces and 
        input_interface. 

    Parameters:
    -----------
    target_nodes : Union[List[float], np.ndarray]
        List or numpy array of target node depths in meters.
    
    input_nodes : Union[List[float], np.ndarray]
        List or numpy array of input node depths in meters.

    input_data : np.ndarray
        2D numpy array of input data with shape (time, len(input_nodes)).

    target_single_level : bool
        Indicates whether the target nodes are single level. If true, target_interface and input_interfaces are un-used. 

    Returns:
    --------
    np.ndarray
        Processed data as a 2D numpy array with shape (time, len(target_nodes)).
    """
    # unifying data types
    target_nodes = np.array(target_nodes)
    input_nodes = np.array(input_nodes)

    # sanity checks
    if not input_data.shape[1] == len(input_nodes):
        raise Exception('Mismatch between specified inputs depths and available data')
    if not target_single_level:
        if (target_interfaces is None or input_interfaces is None):
            raise Exception('Must specify depth bounds if target is not single level')
        if not ((len(target_interfaces) - len(target_nodes)) == 1):
            raise Exception('Number of soil layers mismatched between target interface and node depths')
        if not ((len(input_interfaces) - len(input_nodes)) == 1):
            raise Exception('Number of soil layers mismatched between input interface and node depths')
        # unifying data types
        target_interfaces = np.array(target_interfaces)
        input_interfaces = np.array(input_interfaces)

    # actual calculations
    output_data = np.full([input_data.shape[0], len(target_nodes)], np.nan)
    if target_single_level:
        for i, d in enumerate(target_nodes):
            if d < input_nodes[0]:
                output_data[:, i] = input_data[:, 0]
            elif d > input_nodes[-1]:
                output_data[:, i] = input_data[:, -1]
            else:
                d_matched = np.where(np.isclose(input_nodes, d))[0]
                if len(d_matched) > 1:
                    raise Exception('Input nodes have duplicate values')
                elif len(d_matched) == 1:
                    # just apply the nearest node
                    output_data[:, i] = input_data[:, d_matched[0]]
                else:
                    # interpolate between two nearby nodes
                    d_up = np.where(input_nodes < d)[0][-1]
                    d_down = np.where(input_nodes > d)[0][0]
                    f1 = (input_nodes[d_down] - d) / (input_nodes[d_down] - input_nodes[d_up]) 
                    f2 = (d - input_nodes[d_up]) / (input_nodes[d_down] - input_nodes[d_up])
                    output_data[:, i] = input_data[:, d_up] * f1 + input_data[:, d_down] * f2
    else:
        for i, d1 in enumerate(target_interfaces[:-1]):
            d2 = target_interfaces[i+1]
            if d2 <= input_interfaces[1]:
                output_data[:, i] = input_data[:, 0]
            elif d1 >= input_interfaces[-2]:
                output_data[:, i] = input_data[:, -1]
            else:
                output_data[:, i] = 0.
                sum_weight = 0.
                for j, dd1 in enumerate(input_interfaces[:-1]):
                    dd2 = input_interfaces[j+1]
                    if (dd2 <= d1) or (dd1 >= d2):
                        continue
                    else:
                        if (dd1 >= d1):
                            if (dd2 <= d2):
                                sum_weight += (dd2 - dd1)
                                output_data[:, i] += input_data[:, j] * (dd2 - dd1)
                            else:
                                sum_weight += (d2 - dd1)
                                output_data[:, i] += input_data[:, j] * (d2 - dd1)
                        else:
                            if (dd2 <= d2):
                                sum_weight += (dd2 - d1)
                                output_data[:, i] += input_data[:, j] * (dd2 - d1)
                            else:
                                sum_weight = 1.
                                output_data[:, i] = input_data[:, j]
                                break
                output_data[:, i] /= sum_weight
                # print(i, d1, d2, sum_weight)

    return output_data


def uq_get_obs(VAR_LIST):
    """ Get the observational slope & value at ambient chamber """
    plot_list = [f'P{p:02g}' for p in chamber_list_complete]

    obs_data = pd.read_csv(os.path.join(os.environ['PROJDIR'], 'ELM_Phenology', 'output', 'extract',
                                        'extract_obs_productivity.csv'), index_col = [0, 1])
    t2m_obs = obs_data.loc[:, 'Tair']
    # re-order
    obs_varname = ['AGBiomass_Spruce', 'AGBiomass_Tamarack', 'AGBiomass_Shrub',
                   'AGNPPtoBiomass_Spruce', 'AGNPPtoBiomass_Tamarack', 'AGNPPtoBiomass_Shrub',
                   'AGNPP_Spruce', 'AGNPP_Tamarack', 'AGNPP_Shrub', 'NPP_moss',
                   'BGNPP_TreeShrub', 'BGtoAG_TreeShrub', 'NPP', 'HR', 'NEE']
    obs_data = obs_data.loc[:, obs_varname]


    collection = pd.DataFrame(np.nan,
                              index = pd.MultiIndex.from_product([VAR_LIST, ['amb', 'elev']]),
                              columns = ['mean', 'mean_std', 'slope', 'slope_std'])
    collection.index.names = ['Variable', 'CO2']

    for varname in VAR_LIST:
        for co2 in ['amb','elev']:
            if co2 == 'amb':
                filt = obs_data.index.get_level_values(0).isin([plot_list[0]] + plot_list[1::2])
            else:
                filt = obs_data.index.get_level_values(0).isin(plot_list[2::2])

            if varname != 'TOTSOMC':
                obs_temp = obs_data.loc[filt, varname]
                obs_T    = t2m_obs.loc[filt]

                filt2 = ~np.isnan(obs_T) & ~np.isnan(obs_temp)
                obs_temp = obs_temp.values[filt2]
                obs_T = obs_T.values[filt2]

                res = linregress(obs_T, obs_temp)

                ts = abs(t.ppf(0.05, len(obs_T) - 2))

                collection.loc[(varname, co2), 'slope'] = res.slope
                collection.loc[(varname, co2), 'slope_std'] = ts * res.stderr

                # use the average of all the chambers to be compatible with simulated results
                # obs_temp = obs_data.loc[obs_data.index.get_level_values(0) == plot_list[0], varname] # T0
                collection.loc[(varname, co2), 'mean'] = obs_temp.mean()
                # ---- use random variable theory to estimate this uncertainty
                # print(len(obs_temp)) = 25 to 30, because aggregated over chambers
                collection.loc[(varname, co2), 'mean_std'] = obs_temp.std() / len(obs_temp)
            else:
                collection.loc[(varname, co2), 'mean'] = 200000
                collection.loc[(varname, co2), 'mean_std'] = 500000

    # replace the growth uncertainty with the +/- SD in Paul's excel spreadsheet
    # (assume proportional % uncertainty)
    collection.loc['AGNPP_Spruce', 'mean_std'] = 53.5/90.5 * collection.loc['AGNPP_Spruce', 
                                                                            'mean'].values
    collection.loc['AGNPP_Tamarack', 'mean_std'] = 32/73 * collection.loc['AGNPP_Tamarack', 
                                                                          'mean'].values
    collection.loc['AGNPP_Shrub', 'mean_std'] = 35/92.1 * collection.loc['AGNPP_Shrub', 
                                                                          'mean'].values
    collection.loc['NPP_moss', 'mean_std'] = 67/208 * collection.loc['NPP_moss', 
                                                                     'mean'].values
    collection.loc['BGNPP_TreeShrub', 'mean_std'] = 4.8/3.4 * collection.loc['BGNPP_TreeShrub', 'mean'].values
    collection.loc['HR', 'mean_std'] = -53/283 * collection.loc['HR', 'mean'].values

    #obs_data2.loc[:, 'NPP'] = \
    #    (obs_data2.loc[:, 'AGNPP_Spruce'] + obs_data2.loc[:, 'AGNPP_Tamarack'] + \
    #    obs_data2.loc[:, 'AGNPP_Shrub'] + obs_data2.loc[:, 'BGNPP_TreeShrub'] + \
    #    obs_data2.loc[:, 'NPP_moss']).values
    collection.loc['NPP', 'mean_std'] = np.sqrt( \
        collection.loc['AGNPP_Spruce', 'mean_std']**2 + \
        collection.loc['AGNPP_Tamarack', 'mean_std']**2 + \
        collection.loc['AGNPP_Shrub', 'mean_std']**2 + \
        collection.loc['BGNPP_TreeShrub', 'mean_std']**2 + \
        collection.loc['NPP_moss', 'mean_std']**2
    ).values

    ### replace the ingrowth core uncertainty with that estimated from the 2014 data
    ##collection.loc['BGNPP_TreeShrub', 'mean_std'] = 61.72

    # multiply the numbers by -1 to be consistent with model definition
    collection.loc['HR', 'mean'] = -collection.loc['HR', 'mean'].values
    collection.loc['HR', 'slope'] = -collection.loc['HR', 'slope'].values
    collection.loc['NEE', 'mean'] = -collection.loc['NEE', 'mean'].values
    collection.loc['NEE', 'slope'] = -collection.loc['NEE', 'slope'].values

    return collection


def uq_get_sim(prefix, VAR_LIST):
    plot_list = [f'P{p:02g}' for p in chamber_list_complete]

    sim_data = pd.read_csv(
        os.path.join(os.environ['PROJDIR'], 'ELM_Phenology', 'output', "extract", prefix, 
                     'extract_ts_productivity.csv'), index_col = [0,1,2], header = 0
    )

    sim_tair = sim_data.loc['average', 'Tair']
    sim_data = sim_data.loc['average', VAR_LIST]
    ##sim_data['HR'] = - sim_data['HR'] # I adjusted Paul's value to be consistent with model instead
    ##sim_data['NEE'] = - sim_data['NEE']

    collection = pd.DataFrame(np.nan,
                              index = pd.MultiIndex.from_product([VAR_LIST, ['amb', 'elev']]),
                              columns = ['mean', 'mean_std', 'slope', 'slope_std'])
    collection.index.names = ['Variable', 'CO2']

    for varname in VAR_LIST:
        for co2 in ['amb','elev']:
            if co2 == 'amb':
                filt = sim_data.index.get_level_values(0).isin([plot_list[0]] + plot_list[1::2])
            else:
                filt = sim_data.index.get_level_values(0).isin(plot_list[2::2])

            sim_temp = sim_data.loc[filt, varname]

            res = linregress(sim_tair.loc[filt], sim_temp)

            ts = abs(t.ppf(0.05, len(sim_tair.loc[filt]) - 2))

            collection.loc[(varname, co2), 'slope'] = res.slope
            collection.loc[(varname, co2), 'slope_std'] = ts * res.stderr

            collection.loc[(varname, co2), 'mean'] = sim_temp.mean()
            collection.loc[(varname, co2), 'mean_std'] = sim_temp.mean()

    return collection


def get_obs_agnpp():
    """ Get the observational chamber-wise slope & value """
    plot_list = [f'P{p:02g}' for p in chamber_list_complete]

    obs_data = pd.read_csv(os.path.join(os.environ['PROJDIR'], 'ELM_Phenology', 'output', 'extract',
                                        'extract_obs_productivity.csv'), index_col = [0, 1])
    t2m_obs = obs_data.loc[:, 'Tair']
    obs_varname = ['AGNPP_Spruce', 'AGNPP_Tamarack', 'AGNPP_Shrub', 'NPP_moss']
    obs_data = obs_data.loc[:, obs_varname]

    collection = pd.DataFrame(np.nan, index = plot_list, 
                    columns = pd.MultiIndex.from_product([obs_varname, ['mean','mean_std','slope','slope_std']]))

    for varname in obs_varname:
        for plot in plot_list:
            filt = obs_data.index.get_level_values(0) == plot

            obs_temp = obs_data.loc[filt, varname]
            obs_T    = t2m_obs.loc[filt]

            filt2 = ~np.isnan(obs_T) & ~np.isnan(obs_temp)
            obs_temp = obs_temp.values[filt2]
            obs_T = obs_T.values[filt2]

            if len(obs_temp) >= 5:
                res = linregress(obs_T, obs_temp)
                ts = abs(t.ppf(0.05, len(obs_T) - 2))

                collection.loc[plot, (varname, 'slope')] = res.slope
                collection.loc[plot, (varname, 'slope_std')] = res.stderr * ts
                collection.loc[plot, (varname, 'mean')] = obs_temp.mean()
                collection.loc[plot, (varname, 'mean_std')] = obs_temp.std() / len(obs_temp)

    # replace the growth uncertainty with the +/- SD in Paul's excel spreadsheet
    # (assume proportional % uncertainty)
    collection.loc[:, ('AGNPP_Spruce', 'mean_std')] = 53.5/90.5 * \
        collection.loc[:, ('AGNPP_Spruce', 'mean')].values
    collection.loc[:, ('AGNPP_Tamarack', 'mean_std')] = 32/73 * \
        collection.loc[:, ('AGNPP_Tamarack', 'mean')].values
    collection.loc[:, ('AGNPP_Shrub', 'mean_std')] = 35/92.1 * \
        collection.loc[:, ('AGNPP_Shrub', 'mean')].values
    collection.loc[:, ('NPP_moss', 'mean_std')] = 67/208 * \
        collection.loc[:, ('NPP_moss', 'mean')].values

    return collection


def get_dissolved_nutrients(DEPTH):
    # Observed dissolved N & P at multiple depths
    data = pd.read_csv(
        os.path.join(os.environ['PROJDIR'], 'ELM_Phenology', 'input',
                    'SPRUCE_plot_porewater_chemistry_release_20240617.csv'),
        na_values=[-9999, -8888]
    )
    data['DATE'] = data['DATE'].apply(lambda x: datetime.strptime(str(x), '%Y%m%d'))
    data = data.loc[(pd.DatetimeIndex(data['DATE']).year >= 2015) &
                    (~data['PLOT'].isin([2,5,9,14,15,21])),
                    ['PLOT', 'DEPTH', 'DATE', 'NH4', 'NO3', 'SRP', 'TN', 'TP']]
    data['PLOT'] = [f'{x:02d}' for x in data['PLOT']]
    data['dT'] = [chamber_levels_complete[x][0] for x in data['PLOT']]
    data['CO2'] = [chamber_levels_complete[x][1] for x in data['PLOT']]

    data['NH4+NO3'] = data['NH4'] + data['NO3']

    data = data.loc[np.isclose(data['DEPTH'], DEPTH), :]
    data = data.sort_values(by = 'DATE')

    if DEPTH == 0.3:
        # apparently there is some outlier
        data.loc[data['PLOT'].isin(['10', '19']) & (data['NH4'] > 1.5)] = np.nan
    
    return data
