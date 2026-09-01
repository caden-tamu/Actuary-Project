import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
import seaborn as sns
from sklearn.model_selection import train_test_split

def glmFitLinear(freq):

    '''
    The first model assumes that the relationship between the continuous variable DrivAge and the response variable ClaimNb
    is linear. This assumption may not hold true in practice, as the relationship could be more complex. However, this model
    serves as a baseline to compare against more flexible models that can capture non-linear relationships. The AIC and deviance
    values from this model will provide a benchmark for evaluating the performance of subsequent models.

    AIC: 231116.477
    Deviance: 1.7572e+05
    
    '''

    train, test = train_test_split(freq, test_size=0.2, random_state=42)
    trainOffset = np.log(train['Exposure'])
    testOffset = np.log(test['Exposure'])

    model = smf.glm(formula='ClaimNb ~ BonusMalus + DrivAge', data=train, family=sm.families.Poisson(), offset=trainOffset).fit()
    print(model.summary())
    print(model.aic)

def glmFit2(freq):

    '''
    In this model, I chose to replace the continuous variable DrivAge with a categorical variable DrivAgeBin. The DrivAge was previously
    forced into a linear assumption that did not match the peak around age 50. Binning the variable allows for a more flexible model that
    can capture the non-linear relationship between age and claim frequency. The bins are created using quantiles to ensure that each bin 
    has a similar number of observations, which helps in stabilizing the estimates. The nonlinear relationship lowered the AIC and deviance,
    so it fits the model better. 

    AIC: 230843.934
    Deviance: 1.7544e+05
    
    '''

    train, test = train_test_split(freq, test_size=0.2, random_state=42)
    trainOffset = np.log(train['Exposure'])
    testOffset = np.log(test['Exposure'])

    train["DrivAgeBin"] = pd.qcut(train["DrivAge"], q=5, duplicates='drop')
    train_bins = pd.qcut(train["DrivAge"], q=5, duplicates='drop', retbins=True)[1]
    train["DrivAgeBin"] = pd.cut(train["DrivAge"], bins=train_bins, include_lowest=True)
    test["DrivAgeBin"] = pd.cut(test["DrivAge"], bins=train_bins, include_lowest=True)

    model = smf.glm(formula='ClaimNb ~ BonusMalus + C(DrivAgeBin)', data=train, family=sm.families.Poisson(), offset=trainOffset).fit()
    print(model.summary())
    print(model.aic)

