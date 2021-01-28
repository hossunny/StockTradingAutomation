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

def SummaryFunda_v2(dt, funda_ls=['PBR','PCR','POR','PSR','PER','EPS','BPS','ROE','ROA','시가총액']):
    """Paradox of Simpson"""
    #rst1 = SummaryDataFrame('2016-12','2018-02-15',term=10,funda_ls=['PBR','PCR','POR','PSR','PER','EPS','BPS','ROE','ROA','시가총액'])
    with open("./TradingDates.pickle","rb") as fr :
        td_days = pickle.load(fr)
    conn = pymysql.connect(host='localhost',user='root', password='tlqkfdk2',db='INVESTAR',charset='utf8')
    comp_info = pd.read_sql("select code, sector from company_info",conn)
    code_ls = list(comp_info.code.values)
    #filtered_ls = Filtering(dt, conn, by=['PBR','PCR','POR'])
    print("Initial Filtered Univ : {}".format(len(code_ls)))
    print("EX : {}".format(code_ls[:5]))
    total = pd.DataFrame(index = code_ls, columns = funda_ls)
    fn_df = pd.read_sql(f"select code, itm, value from finance_info_copy where code in {tuple(code_ls)} and date='{dt}' and itm in {tuple(funda_ls)}",conn)
    fn_df = fn_df[lambda x : x['value']!=-999.9]
    for idx, row in fn_df.iterrows():
        if row.itm in ['PBR','PER','PCR','POR','PSR']:
            total.loc[row.code, row.itm] = float(row.value) / 100000000
        elif row.itm in ['EPS','BPS']:
            total.loc[row.code, row.itm] = float(row.value) * 100000000
        else :
            total.loc[row.code, row.itm] = float(row.value)
    
    total = total.dropna(axis=0,how='any').dropna(axis=1,how='any')
    return total.astype(float)

def Labeling(df, qtl=5):
    """Summary df should be inserted"""
    rst = df.copy()
    for c in df.columns :
        tmp_dict = {}
        for i, e in enumerate(list(df.groupby(pd.qcut(df[c],qtl)).agg(['mean']).index)):
            tmp_dict[e] = i+1
        rst[c] = pd.qcut(df[c],qtl).map(tmp_dict)
    return rst


def SectorAnalysis(dt):
    conn = pymysql.connect(host='localhost',user='root',password='tlqkfdk2',db='INVESTAR',charset='utf8')
    comp_info = pd.read_sql("select code, sector from company_info",conn)
    code_ls = list(comp_info.code.values)
    sector_dict = {}
    for i in range(len(comp_info)):
        sector_dict[comp_info.loc[i,'code']] = comp_info.loc[i,'sector']
    cn = conn.cursor()
    cn.execute("select max(date) from daily_price where code='005930'")
    last_update = cn.fetchone()[0].strftime("%Y-%m-%d")
    
    if dt[5:7] == '09':
        start = dt[:4]+'-11-20'
        end = str(int(dt[:4])+1)+'-03-10'
    elif dt[5:7] == '03':
        start = dt[:4]+'-05-20'
        end = dt[:4]+'-08-10'
    elif dt[5:7] == '06':
        start = dt[:4]+'-08-20'
        end = dt[:4]+'-11-10'
    elif dt[5:7] == '12':
        end = str(int(dt[:4])+2)+'-02-28'
        start = str(int(dt[:4])+1)+'-03-31'
    else :
        raise ValueError("Date is not correct.")
    
    if end > last_update :
        end = last_update
    funda = SummaryFunda_v2(dt)
    funda['sector'] = funda.index.map(sector_dict)
    exist_ls = list(funda.index)
    df = ldr.GetPrice(start, end, exist_ls, item='adjprice', colname='code')
    df = df.dropna(axis=1,how='any')
    df = GetExpectedReturn(df,True)
    tmp = pd.DataFrame(index = list(df.columns), columns=['sector'])
    tmp['sector'] = tmp.index.map(sector_dict)
    sctr_ls = list(tmp['sector'].value_counts()[lambda x : x>=20].index)
    rst = pd.DataFrame(columns=['FD-Q','C1','C1(%)','C1(T)','C21-GMean','C21-Mean','C22-GMean','C22-Mean'])
    
    for sc in sctr_ls :
        sc_idx = list(tmp[lambda x : x['sector']==sc].index)
        sub_funda = Labeling(funda[funda.index.isin(sc_idx)].drop(['sector'],axis=1), qtl=3)
        #total_top = pd.DataFrame(index = [sc], columns=['FD-Q','C1','C1(%)','C1(T)','C21-GMean','C21-Mean','C22-GMean','C22-Mean'])
        for fd in funda.columns[:-1]:
            #total_mid = pd.DataFrame(index = [sc], columns=['FD-Q','C1','C1(%)','C1(T)','C21-GMean','C21-Mean','C22-GMean','C22-Mean'])
            for ith in [1,2,3]:
                total = pd.DataFrame(index = [sc], columns=['FD-Q','C1','C1(%)','C1(T)','C21-GMean','C21-Mean','C22-GMean','C22-Mean'])
                total.loc[sc,'FD-Q'] = fd+'-'+str(ith)
                sc_idx_ith = list(sub_funda[lambda x : x[fd]==ith].index)
                sub_df = df[sc_idx_ith]
                cnt = 0
                c1_ls = []
                gmeans = []
                gmeans_m = []
                means = []
                means_m = []
                for cl in sub_df.columns :
                    if len(sub_df[lambda x : x[cl]>=0.3])>0 or (sub_df[cl].values[-1]>=0.2) :
                        if len(sub_df[lambda x : x[cl]<=-0.23])==0 :
                            cnt += 1
                            c1_ls.append(cl)
                    #tmp_df = (sub_df[lambda x : x[cl]>0]+1)
                    tmp_df = sub_df[lambda x : x[cl]>0]
                    gmeans.append(np.exp(np.log(tmp_df.T.prod(axis=1))/tmp_df.T.notna().sum(1)).values[0])
                    means.append(tmp_df[cl].mean())
                    if tmp_df[cl].mean == np.nan:
                        print(cl)

                    #tmp_df_m = (sub_df[lambda x : x[cl]<0]+1)
                    tmp_df_m = sub_df[lambda x : x[cl]<0]*(-1)
                    gmeans_m.append(np.exp(np.log(tmp_df_m.T.prod(axis=1))/tmp_df_m.T.notna().sum(1)).values[0])
                    means_m.append(tmp_df_m[cl].mean())
                
                total.loc[sc,'C1'] = cnt
                total.loc[sc,'C1(%)'] = cnt / len(sc_idx_ith)
                total.loc[sc,'C1(T)'] = len(sc_idx_ith)

                gmean_df = pd.DataFrame(gmeans).T
                total.loc[sc,'C21-GMean'] = np.exp(np.log(gmean_df.prod(axis=1))/gmean_df.notna().sum(1)).values[0]
                total.loc[sc,'C21-Mean'] = pd.Series(means).mean()

                gmean_m_df = pd.DataFrame(gmeans_m).T
                total.loc[sc,'C22-GMean'] = np.exp(np.log(gmean_m_df.prod(axis=1))/gmean_m_df.notna().sum(1)).values[0]
                total.loc[sc,'C22-Mean'] = pd.Series(means_m).mean()
                #total_mid = pd.concat([total_mid, total])
                rst = pd.concat([rst, total])
        
        
    rst.sort_values(by=['C1(%)','C21-GMean','C22-GMean'],ascending=False,inplace=True)
    return rst