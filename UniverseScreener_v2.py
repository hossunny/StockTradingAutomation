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


def SectorFundaC1(dt, fltr_by = None):
    conn = pymysql.connect(host='localhost',user='root',password='tlqkfdk2',db='INVESTAR',charset='utf8')
    ftrd_ls, sector_dict = FilteringCondition(dt, eqty=0.1, volm=0.1, sctr=5, by=fltr_by)
    cn = conn.cursor()
    cn.execute("select max(date) from daily_price where code='005930'")
    last_update = cn.fetchone()[0].strftime("%Y-%m-%d")
    end = str(int(dt[:4])+2)+'-02-28'
    if end > last_update :
        end = last_update
    df = ldr.GetPrice(str(int(dt[:4])+1)+'-03-31', end, ftrd_ls, item='adjprice', colname='code')
    df = df.dropna(axis=1,how='any')
    df = GetExpectedReturn(df,True)
    tmp = pd.DataFrame(index = list(df.columns), columns=['sector'])
    tmp['sector'] = tmp.index.map(sector_dict)
    sctr_ls = list(tmp['sector'].value_counts()[lambda x : x>=5].index)
    over_ls = []
    total = pd.DataFrame(index = sctr_ls, columns=['C1','C1(%)','C1(T)','C1List','SectorList'])
    for sc in sctr_ls :
        sc_idx = list(tmp[lambda x : x['sector']==sc].index)
        sub_df = df[sc_idx]
        cnt = 0
        c1_ls = []
        for c in sub_df.columns :
            if len(sub_df[lambda x : x[c]>=0.3])>0 or (sub_df[c].values[-1]>=0.2) :
                if len(sub_df[lambda x : x[c]<=-0.23])==0 :
                    cnt += 1
                    c1_ls.append(c)
        total.loc[sc,'C1'] = cnt
        total.loc[sc,'C1(%)'] = cnt / len(sc_idx)
        total.loc[sc,'C1(T)'] = len(sc_idx)
        total.loc[sc,'C1List'] = c1_ls
        total.loc[sc,'SectorList'] = sc_idx
    funda = SummaryFunda(dt)
    funda['sector'] = funda.index.map(sector_dict)
    total.sort_values(by=['C1(%)'],ascending=False,inplace=True)
    return total, df, funda

def DistComparison(total, funda):
    fds = list(funda.columns[:-1])
    cols = ['C1Number','C1(%)']
    for fd in fds :
        cols.append(fd+'-Diff')
        cols.append('Median-'+fd)
        cols.append('Mean-'+fd)
        cols.append('MedianRest-'+fd)
        cols.append('MeanRest-'+fd)
    rst = pd.DataFrame(index = list(total.index), columns = cols)
    for sc in list(total.index):
        tmp_fd = funda[funda.index.isin(total.loc[sc,'C1List'])]
        if len(tmp_fd)==0:
            rst.loc[sc,:] = '-1'
            continue
        rst.loc[sc,'C1Number'] = total.loc[sc,'C1(T)']
        rst.loc[sc,'C1(%)'] = total.loc[sc,'C1(%)']
        rest_ls = list(set(total.loc[sc,'SectorList']) - set(total.loc[sc,'C1List']))
        tmp_rest = funda[funda.index.isin(rest_ls)]
        for fd in fds :
            c1 = list(tmp_fd[fd].values)
            notc1 = list(tmp_rest[fd].values)
            if len(tmp_rest)==0:
                rst.loc[sc,fd+'-Diff'] = 'N'
                rst.loc[sc,'Median-'+fd] = tmp_fd[fd].median()
                rst.loc[sc,'Mean-'+fd] = tmp_fd[fd].mean()
                rst.loc[sc,'MedianRest-'+fd] = 0.0
                rst.loc[sc,'MeanRest-'+fd] = 0.0
            else :
                _, pv = stats.mannwhitneyu(c1, notc1)
                if pv <= 0.05 :
                    rst.loc[sc,fd+'-Diff'] = 'Y'
                    rst.loc[sc,'Median-'+fd] = tmp_fd[fd].median()
                    rst.loc[sc,'Mean-'+fd] = tmp_fd[fd].mean()
                    rst.loc[sc,'MedianRest-'+fd] = tmp_rest[fd].median()
                    rst.loc[sc,'MeanRest-'+fd] = tmp_rest[fd].mean()
                else :
                    rst.loc[sc,fd+'-Diff'] = 'N'
                    rst.loc[sc,'Median-'+fd] = tmp_fd[fd].median()
                    rst.loc[sc,'Mean-'+fd] = tmp_fd[fd].mean()
                    rst.loc[sc,'MedianRest-'+fd] = tmp_rest[fd].median()
                    rst.loc[sc,'MeanRest-'+fd] = tmp_rest[fd].mean()
    return rst


