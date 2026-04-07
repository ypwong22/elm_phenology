"""
Critical daylength model for leaves senescing
"""

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
from scipy.stats import linregress


plot_list = ["04", "06", "07", "08", "10", "11", "13", "16", "17", "19", "20", "21"]

pheno_obs = pd.read_excel(
    os.path.join(path_input, "SPRUCE_budburst_summary.xlsx"),
    engine="openpyxl",
    sheet_name="EN",
)
pheno_obs = (
    pheno_obs.loc[:, ["year", "leaves senescing"]].set_index("year", drop=True).dropna()
)
crit_dayl_all = []
for i in range(len(pheno_obs)):
    fall_date = datetime(pheno_obs.index[i] - 1, 12, 31) + timedelta(
        float(pheno_obs.iloc[i])
    )
    crit_dayl = daylength(
        fall_date.year, fall_date.month, fall_date.day, lat, lon, elv, tzn
    )
    print("Critical day length = %f seconds" % crit_dayl)
    crit_dayl_all.append(crit_dayl)
print("Average critical day length = %f seconds" % np.mean(crit_dayl_all))
print("Minimum critical day length = %f seconds" % np.min(crit_dayl_all))
print("Minimum critical day length = %f seconds" % np.max(crit_dayl_all))
print("Std of day length = %f seconds" % np.std(crit_dayl_all))
