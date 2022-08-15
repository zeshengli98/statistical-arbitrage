import pandas as pd
import numpy as np
import yfinance as yf
import statsmodels.api as sm
import matplotlib.pyplot as plt
from statsmodels.tsa.api import adfuller

def OLS(y, x):
    '''
    parameters:
    :param y: independent variable, dataframe or array-like
    :param x: dependent variables, dataframe or array-like
    :return:
    '''

    x = np.array(x)
    y = np.array(y)
    model = sm.OLS(y, sm.add_constant(x)).fit()

    residuals = model.resid
    
    ## OLS params
    c, beta = model.params
    
    ## OLS params sd
    c_sd, beta_sd = model.bse

    # OLS t-statistics
    c_t, beta_t = model.tvalues

    summary =