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


def glmFit3(freq):

    '''
    In this new fit, I chose to include the Vehicle Age in the dataset. The AIC and deviance values decreased, indicating that the 
    model fit improved with the inclusion of this variable. I first tested the model with just a linear dependence for the Vehicle Age,
    but the AIC and deviance values were higher than the model with DrivAgeBin. I then binned the Vehicle Age variable and was able to reduce the 
    AIC and Deviance to 230,843.934 and 1.7544e+05 respectively. This suggests that the relationship between Vehicle Age and ClaimNb is also non-linear, 
    and binning the variable allows for a more flexible model that can capture this relationship.
    
    '''


    train, test = train_test_split(freq, test_size=0.2, random_state=42)
    trainOffset = np.log(train['Exposure'])
    testOffset = np.log(test['Exposure'])

    train["DrivAgeBin"] = pd.qcut(train["DrivAge"], q=5, duplicates='drop')
    train_bins = pd.qcut(train["DrivAge"], q=5, duplicates='drop', retbins=True)[1]
    train["DrivAgeBin"] = pd.cut(train["DrivAge"], bins=train_bins, include_lowest=True)
    test["DrivAgeBin"] = pd.cut(test["DrivAge"], bins=train_bins, include_lowest=True)

    model = smf.glm(formula='ClaimNb ~ BonusMalus + DrivAge + VehAge', data=train, family=sm.families.Poisson(), offset=trainOffset).fit()
    #print(model.summary())
    #print(model.aic)


    train["VehAgeBin"] = pd.qcut(train["VehAge"], q=5, duplicates='drop')
    train_bins = pd.qcut(train["VehAge"], q=5, duplicates='drop', retbins=True)[1]
    train["VehAgeBin"] = pd.cut(train["VehAge"], bins=train_bins, include_lowest=True)
    test["VehAgeBin"] = pd.cut(test["VehAge"], bins=train_bins, include_lowest=True)

    model = smf.glm(formula='ClaimNb ~ BonusMalus + C(DrivAgeBin) + C(VehAgeBin)', data=train, family=sm.families.Poisson(), offset=trainOffset).fit()
    print(model.summary())
    print(model.aic)



def glmFit4(freq):

    '''
    In this fit I added the Region categorical variable to the dataset. The AIC and deviance values decreased, 
    indicating that the model fit improved with the inclusion of this variable. However, many of the variables have
    high p-values, suggesting that they may not be statistically significant predictors of the response variable. I plan
    to make broader region tiers to avoid overtfitting and determine if the model fit improves.
    '''


    train, test = train_test_split(freq, test_size=0.2, random_state=42)
    trainOffset = np.log(train['Exposure'])
    testOffset = np.log(test['Exposure'])

    train["DrivAgeBin"] = pd.qcut(train["DrivAge"], q=5, duplicates='drop')
    train_bins = pd.qcut(train["DrivAge"], q=5, duplicates='drop', retbins=True)[1]
    train["DrivAgeBin"] = pd.cut(train["DrivAge"], bins=train_bins, include_lowest=True)
    test["DrivAgeBin"] = pd.cut(test["DrivAge"], bins=train_bins, include_lowest=True)
    freq["DrivAgeBin"] = pd.cut(freq["DrivAge"], bins=train_bins, include_lowest=True)  # Add to freq


    train["VehAgeBin"] = pd.qcut(train["VehAge"], q=5, duplicates='drop')
    train_bins = pd.qcut(train["VehAge"], q=5, duplicates='drop', retbins=True)[1]
    train["VehAgeBin"] = pd.cut(train["VehAge"], bins=train_bins, include_lowest=True)
    test["VehAgeBin"] = pd.cut(test["VehAge"], bins=train_bins, include_lowest=True)
    freq["VehAgeBin"] = pd.cut(freq["VehAge"], bins=train_bins, include_lowest=True)  # Add to freq


    #Adding region to the model to see if it improves the AIC and deviance
    model = smf.glm(formula='ClaimNb ~ BonusMalus + C(DrivAgeBin) + C(VehAgeBin) + C(Region)', data=train, family=sm.families.Poisson(), offset=trainOffset).fit()
    print(model.summary())
    print(model.aic)

    return model, freq