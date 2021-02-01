import logging, os, pickle
import requests, glob
from datetime import datetime
from bs4 import BeautifulSoup
from datetime import datetime
import pandas as pd
import time
from datetime import date
import urllib.request
from selenium.webdriver import Chrome
import json, re, sys, h5py
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.keys import Keys
import datetime as dt
import pymysql
import matplotlib.pyplot as plt
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from pykrx import stock
import warnings
warnings.filterwarnings(action='ignore')
import shutil
from matplotlib.pyplot import cm
import numpy as np
import scipy.stats as stats
from scipy import stats
from statsmodels.tsa.stattools import coint, adfuller

def find_cointegrated_pairs(data):
    n = data.shape[1]
    score_matrix = np.zeros((n, n))
    pvalue_matrix = np.ones((n, n))
    keys = data.keys()
    pairs = []
    for i in range(n):
        for j in range(i+1, n):
            S1 = data[keys[i]]
            S2 = data[keys[j]]
            result = coint(S1, S2)
            score = result[0]
            pvalue = result[1]
            score_matrix[i, j] = score
            pvalue_matrix[i, j] = pvalue
            if pvalue < 0.02:
                pairs.append((keys[i], keys[j]))
    return score_matrix, pvalue_matrix, pairs
"""
# Heatmap to show the p-values of the cointegration test
# between each pair of stocks
data = test1_ex[['069080', '131370', '203650', '251270','080580', '087600', '208710','011930', '108320', '322000']]
scores, pvalues, pairs = find_cointegrated_pairs(data)
import seaborn
m = [0,0.2,0.4,0.6,0.8,1]
plt.figure(figsize=(14,10))
seaborn.heatmap(pvalues, xticklabels=data.columns, 
                yticklabels=data.columns, cmap='RdYlGn_r', 
                mask = (pvalues >= 0.98))
plt.show()
print(pairs)
"""

def cointeg_test(scs, all_pr, comp,non_stat):
    dct = {}
    errs = []
    for sc in scs:
        try :
            all_sc = list(comp[lambda x : x['sector']==sc].code.values)
            sc_codes = list(set(all_sc).intersection(set(all_pr.columns)))
            sc_codes = list(set(sc_codes).intersection(set(non_stat)))
            sub = all_pr[sc_codes]
            scores, pvalues, pairs = find_cointegrated_pairs(sub)
            dct[sc] = pairs
        except :
            errs.append(sc)
    return dct, errs

def stationarity_test(pr, cutoff=0.01):
    # H_0 in adfuller is unit root exists (non-stationary)
    # We must observe significant p-value to convince ourselves that the series is stationary
    non_stat = []
    yes_stat = []
    errs = []
    for cd in list(pr.columns):
        try :
            sub_df = pr[[cd]].dropna(axis=0)
            pvalue = adfuller(sub_df[cd])[1]
            if pvalue < cutoff:
                yes_stat.append(cd)
            else :
                non_stat.append(cd)
        except:
            errs.append(cd)
    return non_stat, yes_stat, errs