import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import math
import Interactions as Interactions
import glmBuild as glmBuild
import Plots as Plots
import Validation as V

#Loading in data from sev and freq csv files
sev = pd.read_csv("data/freMTPL2sev.csv")
freq = pd.read_csv("data/freMTPL2freq.csv")

#print(sev.head())
#print(freq.head())

exposureList = ["DrivAge", "VehAge", "Density", "BonusMalus"]

#for _ in exposureList:
#   result = freqPlotsPercentile(freq, _)

discreteList = ['VehPower', 'Region', 'VehGas', 'VehBrand', 'Area']

'''
for _ in discreteList:
    result = Plots.discretePlots(freq, _)

    '''

#Interactions.overDispersion(freq)
Interactions.interactionTerms(freq)

freqPivot, exposurePivot = Interactions.heatMap(freq, "DrivAge", "BonusMalus", q=4)

#print(freqPivot)
#print(exposurePivot)

model4, freq = glmBuild.glmFit6(freq)
#freq["predicted"] = model4.predict(freq, offset=np.log(freq["Exposure"]))
#liftSummary = V.liftChart(model4, freq, offsetCol="Exposure", claimCol="ClaimNb", q=10, saveName="lift_chart_model4")
#giniCoef = V.giniCoefficient(model4, freq, offsetCol="Exposure", claimCol="ClaimNb", saveName="lorenz_curve_model4")
#print(f"Gini coefficient: {giniCoef:.4f}")