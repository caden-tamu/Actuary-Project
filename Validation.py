import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
import seaborn as sns
from sklearn.model_selection import train_test_split


def liftChart(model, test, offsetCol="Exposure", claimCol="ClaimNb", q=10, saveName = 'LiftChart'):

    if "predicted" not in test.columns:
        raise KeyError("The 'predicted' column is missing. Ensure you generate predictions before calling liftChart.")

    offset = np.log(test[offsetCol])
    test = test.copy() 
    
    test["predicted_freq"] = test["predicted"] / test[offsetCol]
    test["predicted_decile"] = pd.qcut(test["predicted_freq"], q=q, labels=False, duplicates='drop')

    lift_summary = test.groupby("predicted_decile", observed=True).agg(ActualClaims=(claimCol, "sum"), PredictedClaims=("predicted", "sum"),
        Exposure=(offsetCol, "sum")).reset_index()

    lift_summary["ActualFrequency"] = lift_summary["ActualClaims"] / lift_summary["Exposure"]
    lift_summary["PredictedFrequency"] = lift_summary["PredictedClaims"] / lift_summary["Exposure"]

    plt.figure(figsize=(10, 6))
    plt.plot(lift_summary["predicted_decile"], lift_summary["ActualFrequency"], marker='o', label="Actual")
    plt.plot(lift_summary["predicted_decile"], lift_summary["PredictedFrequency"], marker='o', label="Predicted")
    plt.xlabel("Predicted Risk Decile (0 = lowest, 9 = highest)")
    plt.ylabel("Claim Frequency")
    plt.title("Lift Chart: Actual vs Predicted Frequency by Decile")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"ExposureGraphs/{saveName}.png", dpi=300, bbox_inches="tight")
    plt.close()

    return lift_summary
