import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statistics
import math

def interactionTerms(freq):
    '''
    Choosing some interaction terms that make sense, I will be testing (VehGas x VehAge), (Region x VehPower)
    (DrivAve x BonusMalus), ()
    '''

    freq["VehGas_VehAge"] = freq["VehGas"].astype(str) + "_" + freq["VehAge"].astype(str)
    freq["Region_VehPower"] = freq["Region"].astype(str) + "_" + freq["VehPower"].astype(str)
    freq["DrivAge_BonusMalus"] = freq["DrivAge"].astype(str) + "_" + freq["BonusMalus"].astype(str)
    return freq


def overDispersion(freq):
    freqMean = freq["ClaimNb"].mean()
    freqVar = freq["ClaimNb"].var()
    claimNbCount = freq["ClaimNb"].count()
    claimNbCount0 = freq[freq["ClaimNb"] == 0]["ClaimNb"].count()
    print("Claim Frequency Mean: ", freqMean)
    print("claim Frequency Variance: ", freqVar)
    print("Percent of Claims that are 0: ", claimNbCount0 / claimNbCount)
    '''
    Because the percent of claims that are 0 is approximately a Poisson Pr(X=0) = exp(-0.053) (since lambda = mean = 0.053), 
    we can assume the standard Poisson GLM is a reasonable choice for a frequency model
    
    '''

def heatMap(freq, colName1, colName2, q=4):
    bin1 = f"{colName1}_Bin_Coarse"
    bin2 = f"{colName2}_Bin_Coarse"

    freq[bin1] = pd.qcut(freq[colName1], q=q, duplicates='drop')
    freq[bin2] = pd.qcut(freq[colName2], q=q, duplicates='drop')

    grouped = freq.groupby([bin1, bin2], observed=True).agg(
        ClaimNb=("ClaimNb", "sum"), Exposure=("Exposure", "sum")).reset_index()

    grouped["Frequency"] = grouped["ClaimNb"] / grouped["Exposure"]
    freqPivot = grouped.pivot(index=bin1, columns=bin2, values="Frequency")
    exposurePivot = grouped.pivot(index=bin1, columns=bin2, values="Exposure")

    plt.figure(figsize=(8, 6))
    sns.heatmap(freqPivot, annot=True, fmt=".3f", cmap="viridis")
    plt.title(f"Claim Frequency: {colName1} x {colName2}")
    plt.tight_layout()
    plt.savefig(f"ExposureGraphs/{colName1}_{colName2}_Frequency_Heatmap.png", dpi=300, bbox_inches="tight")
    plt.close()

    return freqPivot, exposurePivot