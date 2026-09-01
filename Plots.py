import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import math

def pairPlots(freq):

    fig1 = sns.pairplot(freq, corner = True)
    fig1.savefig("freqPairplot.png", dpi=300, bbox_inches="tight")


def freqPlotsPercentile(freq, colName):
    newColName = f"{colName}_Bin"

    # Check if the column is numeric
    if np.issubdtype(freq[colName].dtype, np.number):
        freq[newColName] = pd.qcut(freq[colName], q=10, duplicates='drop')
    else:
        print(f"Column '{colName}' is not numeric. Using it as a categorical variable.")
        freq[newColName] = freq[colName]  # Treat as categorical

    summaryExp = freq.groupby(newColName, observed=False)["Exposure"].sum().reset_index()
    summaryClaim = freq.groupby(newColName, observed=False)["ClaimNb"].sum().reset_index()
    summary = summaryExp.merge(summaryClaim, on=newColName)
    summary["Frequency"] = summary["ClaimNb"] / summary["Exposure"]

    plt.figure(figsize=(8, 8))
    sns.barplot(data=summary, x=newColName, y="Exposure", palette="viridis")
    plt.title(f"Total Exposure by {newColName} Bucket")
    plt.tick_params(axis='x', rotation=90)
    plt.tight_layout()
    plt.savefig(f"ExposureGraphs/{colName}_Exposure.png", dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(8, 8))
    sns.barplot(data=summary, x=newColName, y="Frequency", palette="viridis")
    plt.title(f"Claim Frequency by {newColName} Bucket")
    plt.tick_params(axis='x', rotation=90)
    plt.tight_layout()
    plt.savefig(f"ExposureGraphs/{colName}_Frequency.png", dpi=300, bbox_inches="tight")
    plt.close()

    return summary

def discretePlots(freq, colName):
    summaryExp = freq.groupby(colName, observed = False)["Exposure"].sum().reset_index()
    summaryClaim = freq.groupby(colName, observed = False)["ClaimNb"].sum().reset_index()
    summary = summaryExp.merge(summaryClaim, on=colName)
    summary["Frequency"] = summary["ClaimNb"] / summary["Exposure"]
    
    plt.figure(figsize=(8, 8))
    sns.barplot(data=summary, x=colName, y="Exposure", palette="viridis")
    plt.title(f"Total Exposure by {colName}")
    plt.tick_params(axis='x', rotation=90)
    plt.tight_layout()
    plt.savefig(f"ExposureGraphs/{colName}_Exposure.png", dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(8, 8))
    sns.barplot(data=summary, x=colName, y="Frequency", palette="viridis")
    plt.title(f"Claim Frequency by {colName}")
    plt.tick_params(axis='x', rotation=90)
    plt.tight_layout()
    plt.savefig(f"ExposureGraphs/{colName}_Frequency.png", dpi=300, bbox_inches="tight")
    plt.close()

def plotInteractionTerms(freq):
    '''
    Generate frequency plots for interaction terms.
    '''
    interactionCols = ["VehGas_VehAge", "Region_VehPower", "DrivAge_BonusMalus"]
    for col in interactionCols:
        freqPlotsPercentile(freq, col)