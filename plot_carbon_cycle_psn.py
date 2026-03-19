import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from utils.constants import *
from utils.paths import *
from utils.plotting import *
from utils.analysis import *
from tqdm import tqdm


data_collection = {}
for prefix, name in zip(["20230526", "20230623"], ["Old", "New"]):
    data_collection[name] = read_extract_sims_ts(prefix)

# Sensitivity of PSNSUN & PSNSHA to air temperature
# Apparently no sensitivity
fig, axes = plt.subplots(2, 3, figsize=(12, 12))
for i, pft in enumerate([2, 3, 11]):
    for j, var in enumerate(["PSNSUN", "PSNSHA"]):
        ax = axes[j, i]

        for name, color in zip(["Old", "New"], ["b", "r"]):
            temp0 = data_collection[name]
            temp = (
                temp0.loc[:, (slice(None), "TBOT", 0, "hummock")] * 0.64
                + temp0.loc[:, (slice(None), "TBOT", 0, "hollow")].values * 0.36
            )

            # subset to July
            # temp = temp.loc[temp.index.month == 7, :]
            temp_mean = temp.resample("1Y").mean()

            if name == "Old":
                prefix = "20230526"
            else:
                prefix = "20230623"
            temp2_mean = pd.DataFrame(
                np.nan, index=temp_mean.index, columns=temp_mean.columns.levels[0]
            )
            for plot in tqdm(chamber_list_complete):
                casename = f"{prefix}_plot{plot:02g}_US-SPR_ICB20TRCNPRDCTCBC"
                hr = xr.open_mfdataset(
                    [
                        os.path.join(
                            path_run,
                            casename,
                            "run",
                            f"{casename}.clm2.h0.2015-02-01-00000.nc",
                        ),
                        os.path.join(
                            path_run,
                            casename,
                            "run",
                            f"{casename}.clm2.h0.2020-02-01-00000.nc",
                        ),
                    ]
                )
                temp2 = (
                    (hr[var][:-1, 0] * 0.64 + hr[var][:-1, 1] * 0.36)
                    .resample(time="1Y")
                    .mean()
                )
                temp2_mean.loc[:, plot] = temp2.values  # original value is monthly
                hr.close()

            ax.plot(
                temp_mean.values.reshape(-1),
                temp2_mean.values.reshape(-1),
                "o",
                color=color,
                label=name,
            )

        ax.set_xlabel("TBOT")
        ax.set_ylabel(var)
        if i == 0:
            ax.legend()

plt.savefig(
    os.path.join(path_out, f"plot_carbon_cycle_compare_psn_temperature.png"),
    dpi=600.0,
    bbox_inches="tight",
)
