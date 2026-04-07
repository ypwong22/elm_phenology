"""
Find out the link between aboveground and belowground start of date of spring growth. 

For evergreen tree, reduce the cumulative forcing threshold of the Uniforc model, which
    fits the best. 

For deciduous tree and shrub, the temperature threshold for GDD is too high (~6 degrees),
    reduce the temperature threshold to ...


It is not feasible to adjust the GDD threshold (requires negative values)
   or the temperature threshold (requires < -7 degC) within reasonable ranges, to model
   roots out timing. 

In some years, if forcing threshold is never reached, let the root start growting 
    anyways when leaf is out.
"""
from pyPhenology import utils
import numpy as np
import pandas as pd
import os
from utils.constants import *
from utils.paths import *
from utils.analysis import *
from utils.phenofuncs import *
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.dates as mdates
from matplotlib.ticker import MultipleLocator, AutoMinorLocator


dtFmt = mdates.DateFormatter("%m-%d")  # define the formatting

lyr_select = "2m"
observations, predictors = prepare_inputs()
predictors = predictors[lyr_select]


def check_root(year, month, date, params, model_name, pft, ax):
    Model = utils.load_model(model_name)
    if model_name == "Alternating":
        params["threshold"] = -7.5
        # thres_list = np.linspace(-15, 15, 21)
        thres_list = np.linspace(250, 1750, 21)
    elif model_name == "Uniforc":
        thres_list = np.linspace(30, 90, 21)
    else:
        raise Exception("Not implemented")
    mean_out = {}
    low_out = {}
    up_out = {}
    for t in thres_list:
        params = params.copy()
        if model_name == "Alternating":
            # params["threshold"] = t
            params["b"] = t
        elif model_name == "Uniforc":
            params["F"] = t
        else:
            raise Exception("Not implemented")
        model = Model(parameters=params)
        # note the direct prediction is days since winter solstice
        dayofyear = model.predict(observations[pft], predictors) - 11
        # average & std over the chambers and years
        mean_out[t] = datetime(2017, 12, 31) + timedelta(days=dayofyear.mean())
        low_out[t] = timedelta(days=dayofyear.std())
        up_out[t] = timedelta(days=dayofyear.std())
    mean_out = pd.Series(mean_out)
    low_out = pd.Series(low_out)
    up_out = pd.Series(up_out)
    ax.plot(thres_list, mean_out, "ob", label="Chamber average")
    ax.errorbar(
        thres_list,
        mean_out,
        np.vstack((low_out.values.reshape(1, -1), up_out.values.reshape(1, -1))),
        color="b",
    )
    ax.axhline(datetime(year, month, date), lw=2, color="r", label="Observed root out")
    ax.yaxis.set_major_formatter(dtFmt)

    # (add a leaf out reference)
    temp = observations[pft]["doy"]
    doy_leafout = timedelta(days=temp.mean())
    doy_leafout_upper = timedelta(days=temp.mean() + temp.std())
    doy_leafout_lower = timedelta(days=temp.mean() - temp.std())
    ax.axhline(
        datetime(2017, 12, 31) + doy_leafout,
        ls="-",
        color="k",
        label="Observed leaf out",
    )
    ax.axhline(datetime(2017, 12, 31) + doy_leafout_upper, ls="--", color="k")
    ax.axhline(datetime(2017, 12, 31) + doy_leafout_lower, ls="--", color="k")

    # grid line
    ax.xaxis.set_minor_locator(AutoMinorLocator(5))
    ax.grid(axis="x", which="major", color="magenta", linestyle=":", lw=0.5)
    ax.grid(axis="x", which="minor", color="grey", linestyle=":", lw=0.5)

    return ax


fig, axes = plt.subplots(1, 3, figsize=(12, 5), sharey=True)
fig.subplots_adjust(wspace=0.02)

# Evergreen: check the cumulative forcing threshold
params = {
    "F": 66.94329645213861,
    "b": -0.04412234589076469,
    "c": 6.848386620467728,
    "t1": 1,
}
ax = axes[0]
check_root(2018, 4, 4, params, "Uniforc", "EN", ax)
ax.text(-0.05, 1.05, "a", transform=ax.transAxes, fontweight="bold")
ax.set_ylabel("Predicted date of root out")
ax.set_xlabel("Cumulative forcing threshold")

# Deciduous tree: vary the threshold for growing degree days
params = {"a": 9, "b": 2112, "c": -0.04, "threshold": 279.5 - 273.15, "t1": 1}
ax = axes[1]
check_root(2018, 4, 4, params, "Alternating", "DN", ax)
ax.text(-0.05, 1.05, "b", transform=ax.transAxes, fontweight="bold")
ax.set_xlabel("Slope of chilling day curve")
ax.legend(loc="upper left")

# Deciduous shrub
params = {"a": 33, "b": 1388, "c": -0.02, "threshold": 279.05 - 273.15, "t1": 1}
ax = axes[2]
check_root(2018, 3, 10, params, "Alternating", "SH", ax)
ax.text(-0.05, 1.05, "c", transform=ax.transAxes, fontweight="bold")
ax.set_ylabel("")
ax.set_xlabel("Slope of chilling day curve")

fig.savefig(
    os.path.join(path_out, "fit_ground_phenology_chil.png"),
    dpi=600.0,
    bbox_inches="tight",
)
plt.close(fig)
