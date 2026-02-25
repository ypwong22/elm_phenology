"""

Test these fitting equations

https://academic.oup.com/treephys/article/34/1/87/1619388#23767752

Miller et al. 2022 A methodological framework to optimize models predicting critical dates of xylem phenology based on dendrometer data

* For our comparison we selected a set of seven previously used deterministic growth functions, presented in the literature to model intra-annual xylem growth dynamics based on dendrometer time series (three-parameter logistic model, three-parameter Gompertz, four-parameter Gompertz, Weibull 1, Weibull 2, six-parameter double sigmoid function, seven-parameter double sigmoid function; for details on the models see Table A1) as well as SCAMs (e.g. Sprengel et al., 2018; Stangler et al., 2021). The models were fitted to the cleaned daily mean values of dendrometer data using the R packages minpack.lm (Elzhov et al., 2010), drc (Ritz et al., 2015) and scam (Pya, 2021). For SCAMs, the basis dimension determining model flexibility was chosen between 2 and 20 based on AIC selection. In addition, for the deterministic growth models we conducted two model selections for the individual tree models, one based on AIC and another based on mean absolute error (MAE), to test whether models chosen by those selection criteria reflect tree-ring phenology better. Based on the selection criterion for each tree one of the seven deterministic models was chosen.

* From the fitted model curves, growth onset and cessation were derived as the points in time when 5% and 95% of the total radial change in the considered period were achieved, respectively (relative thresholds) (see Fig. 1a). In a second approach, onset and cessation were defined as the points in time when the growth rate, estimated by the first derivative of the fitted models, exceeds and deceeds a value of 5 µm d-1, respectively (absolute thresholds) (see Fig. 1b). These threshold values were arbitrarily defined as 5 µm d-1 by Duchesne et al. (2012) and as the 5% and 95% percentile of the total radial change in the considered period by Henhappl (1965). To assess the suitability of different thresholds, we also performed calculations with absolute thresholds ranging from 1 to 15 µm d-1 as well as relative thresholds ranging from 1% to 15% for growth onset and from 85% to 99% for growth cessation.

"""



