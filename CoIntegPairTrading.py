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


##########################################

def AnnualSectorDist_v5(fn_label, pr, dt, sc, initial=True, doplot=False):
    """sc_df_ls = [sc2016,sc2017,sc2018,sc2019]"""
    rst = pd.DataFrame(columns=['Date','Funda','C31-Best','C31-Count','Dominant'])
    year = int(dt[:4])
    if year == 2019 :
        start = str(year+1)+'-03-31'
        end = str(year+1)+'-12-31'
    else :
        start = str(year+1)+'-03-31'
        end = str(year+2)+'-02-28'
    for fd in ['PBR','PCR','POR','PSR','PER','EPS','BPS','ROE','ROA','시가총액']:
        tmp = pd.DataFrame(index=[sc],columns=['Date','Funda','C31-Best','C31-Count','Dominant'])
        mer_total = pd.DataFrame()
        if initial and doplot:
            plt.figure(figsize=(16,12))
            plt.title("Total Expected Return in sector {} with {}".format(sc,fd))
            plt.xlabel(f"{fd} with 3 quantile")
            plt.ylabel('Expected Return Rate')
        for ith in ['1','2','3']:
            sc_idx = list(fn_label[lambda x : x[fd]==int(ith)].index)
            sub_pr = pr[sc_idx]
            sub_pr = sub_pr.dropna(axis=1,how='any')
            er = GetExpectedReturn_v2(sub_pr,initial)
            am_pr = pd.DataFrame(er.T.mean(), columns=[fd+'-'+ith])
            mer_total = pd.concat([mer_total, am_pr],axis=1).dropna(axis=0,how='any')
            if initial and doplot:
                plt.plot(list(sub_pr.index), am_pr[fd+'-'+ith].values, linestyle='--', label='{}-{}-Quantile'.format(fd,ith))
        if initial and doplot:
            plt.legend(loc='upper left')
            plt.grid(True)
            plt.show()
        x = y = z = 0
        for d in list(mer_total.index):
            idx = mer_total.loc[d].idxmax()
            if idx == fd+'-1': x+=1
            elif idx == fd+'-2' : y+=1
            elif idx == fd+'-3' : z+=1
            else : raise ValueError("Can't be !!!")
        var = {x:'1', y:'2', z:'3'}
        tmp.loc[sc,'C31-Best'] = var.get(max(var))
        tmp.loc[sc,'C31-Count'] = [x,y,z]
        tmp.loc[sc,'Date'] = dt
        tmp.loc[sc,'Funda'] = fd
        tmp_ls = [x,y,z]
        tmp_ls = [x/sum(tmp_ls), y/sum(tmp_ls), z/sum(tmp_ls)]
        flag = 'N'
        for e in tmp_ls :
            if e >= 0.65 :
                flag='Y'
        tmp.loc[sc,'Dominant'] = flag
        
        rst = pd.concat([rst,tmp])
                
    return rst

def Cointegrated_Pairs(data, cutoff=0.01):
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
            if pvalue < cutoff:
                pairs.append((keys[i], keys[j]))
    return score_matrix, pvalue_matrix, pairs

def Stationarity_Test_Ticker(pr, cutoff=0.01):
    # H_0 in adfuller is unit root exists (non-stationary)
    # We must observe significant p-value to convince ourselves that the series is stationary
    assert pr.shape[1] == 1
    cd = pr.columns[0]
    try :
        sub_df = pr[[cd]].dropna(axis=0)
        pvalue = adfuller(sub_df[cd])[1]
        if pvalue < cutoff:
            return False
        else :
            return True
    except Exception as e:
        print(e)
        return False

def SummaryFunda(dt, codes, funda_ls=['PBR','PCR','POR','PSR','PER','EPS','BPS','ROE','ROA','시가총액']):
    """Paradox of Simpson"""
    #rst1 = SummaryDataFrame('2016-12','2018-02-15',term=10,funda_ls=['PBR','PCR','POR','PSR','PER','EPS','BPS','ROE','ROA','시가총액'])
    with open("./TradingDates.pickle","rb") as fr :
        td_days = pickle.load(fr)
    conn = pymysql.connect(host='localhost',user='root', password='tlqkfdk2',db='INVESTAR',charset='utf8')
    
    #filtered_ls = Filtering(dt, conn, by=['PBR','PCR','POR'])
    #print("Initial Filtered Univ : {}".format(len(filtered_ls)))
    #print("EX : {}".format(filtered_ls[:5]))
    total = pd.DataFrame(index = codes, columns = funda_ls)
    fn_df = pd.read_sql(f"select code, itm, value from finance_info_copy where code in {tuple(codes)} and date='{dt}' and itm in {tuple(funda_ls)}",conn)
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

def Picker_v1(dt='2019-12', sc='반도체 제조업',doplot=False):
    """Step0 : Default Setting """
    print("============== Step0 ================")
    year = int(dt[:4])
    start = str(year+1)+'-03-31'
    if year != 2019:
        end = str(year+2)+'-02-28'
    else :
        end = '2020-12-31'
    conn = pymysql.connect(host='localhost',user='root', password='tlqkfdk2',db='INVESTAR',charset='utf8')
    sc_ls = list(pd.read_sql("select code, sector from company_info", conn)[lambda x : x['sector']==sc].code.values)
    pr = ldr.GetPrice(start, end, sc_ls, 'adjprice', 'code')
    pr.dropna(axis=1, how='all',inplace=True)
    
    """Step1 : Filtering """
    print("============== Step1 ================")
    basic_fltr, _ = FilteringBySector(dt, sc, by=None)
    
    """Step2 : Non-Stationarity """
    print("============== Step2 ================")
    non_stat = []
    for cd in basic_fltr :
        try :
            if Stationarity_Test_Ticker(pr[[cd]], cutoff=0.05):
                non_stat.append(cd)
        except :
            pass
    if len(non_stat)==0:
        raise ValueError("None of stocks is non-stationary...")
    else :
        print("*Non-Stationarity Filtered number of comps : {}".format(len(non_stat)))
    
    """Step3 : Funda Quantile """
    print("============== Step3 ================")
    fn = SummaryFunda('2019-12',non_stat)
    print("Funda-existence Filtered number of comps : {}".format(len(fn)))
    fn_label = Labeling(fn, 3)
    
    """Step4 : Funda to Best ER """
    print("============== Step4 ================")
    rst_init_Y = AnnualSectorDist_v5(fn_label, pr, dt, sc, initial=True, doplot=doplot)
    rst_init_N = AnnualSectorDist_v5(fn_label, pr, dt, sc, initial=False)
    
    """Step5 : CoInteg Pairs """
    print("============== Step5 ================")
    _,_, pairs = Cointegrated_Pairs(pr[list(fn.index)].dropna(axis=1,how='any'),cutoff=0.05) #빼야하나..
    
    return rst_init_Y, rst_init_N, pairs, fn, fn_label

def Labeling(df, qtl=5):
    """Summary df should be inserted"""
    rst = df.copy()
    for c in df.columns :
        tmp_dict = {}
        for i, e in enumerate(list(df.groupby(pd.qcut(df[c],qtl)).agg(['mean']).index)):
            tmp_dict[e] = i+1
        rst[c] = pd.qcut(df[c],qtl).map(tmp_dict)
    return rst

def counting(sub_df, plimit=0.9):
    ls_ls = list(sub_df['C31-Count'].values)
    q1 = q2 = q3 = 0
    for ls in ls_ls :
        q1 += ls[0]
        q2 += ls[1]
        q3 += ls[2]
    q11 = q1 / (q1+q2+q3)
    q22 = q2 / (q1+q2+q3)
    q33 = q3 / (q1+q2+q3)
    rst = 'None'
    if q11+q22 >= plimit :
        rst = ['1','2']
    elif q11+q33 >= plimit :
        rst = ['1','3']
    elif q22+q33 >= plimit :
        rst = ['2','3']
    else :
        if q11 >= 0.6 :
            rst = ['1']
        elif q22 >= 0.6 :
            rst = ['2']
        elif q33 >= 0.6 :
            rst = ['3']
        else : pass
    
    return [q1, q2, q3], rst

def zscore(series):
    return (series - series.mean()) / np.std(series)

def Picker_v2(sc='반도체 제조업', cutoff=0.05, doplot=False):
    """Step0 : Default Setting """
    print("============== Step0 ================")
    dates = ['2016-12','2017-12','2018-12','2019-12']
    conn = pymysql.connect(host='localhost',user='root', password='tlqkfdk2',db='INVESTAR',charset='utf8')
    sc_ls = list(pd.read_sql("select code, sector from company_info", conn)[lambda x : x['sector']==sc].code.values)
    all_pr = ldr.GetPrice('2017-03-31','2020-12-31',sc_ls,'adjprice','code')
    all_pr.dropna(axis=1, how='all',inplace=True)
    
    for dt in dates :
        year = int(dt[:4])
        start = str(year+1)+'-03-31'
        if year != 2019:
            end = str(year+2)+'-02-28'
        else :
            end = '2020-12-31'
        
        pr = all_pr[(all_pr.index>=start)&(all_pr.index<=end)]
        #pr = ldr.GetPrice(start, end, sc_ls, 'adjprice', 'code')
        #pr.dropna(axis=1, how='all',inplace=True)
    
        """Step1 : Filtering """
        print("============== Step1 ================")
        basic_fltr, _ = FilteringBySector(dt, sc, by=None)
        basic_fltr = list(set(sc_ls).intersection(set(basic_fltr)))
    
    print("Final Basic Filtered number of comps : {}".format(len(basic_fltr)))
    """Step2 : Non-Stationarity """
    print("============== Step2 ================")
    non_stat = []
    for cd in basic_fltr :
        try :
            if Stationarity_Test_Ticker(all_pr[[cd]], cutoff=cutoff):
                non_stat.append(cd)
        except :
            pass
    if len(non_stat)==0:
        raise ValueError("None of stocks is non-stationary...")
    else :
        print("*Non-Stationarity Filtered number of comps : {}".format(len(non_stat)))
    
    rst_total = pd.DataFrame()
    for dt in dates :
        year = int(dt[:4])
        start = str(year+1)+'-03-31'
        if year != 2019:
            end = str(year+2)+'-02-28'
        else :
            end = '2020-12-31'
        pr = all_pr[(all_pr.index>=start)&(all_pr.index<=end)]
        
        """Step3 : Funda Quantile """
        print("============== Step3 ================")
        fn = SummaryFunda(dt,non_stat)
        print("Funda-existence Filtered number of comps : {}".format(len(fn)))
        fn_label = Labeling(fn, 3)
        non_stat = list(set(non_stat).intersection(set(fn_label.index)))

        """Step4 : Funda to Best ER """
        print("============== Step4 ================")
        rst_init_Y = AnnualSectorDist_v5(fn_label, pr, dt, sc, initial=True, doplot=doplot)
        rst_total = pd.concat([rst_total, rst_init_Y])
        #rst_init_N = AnnualSectorDist_v5(fn_label, pr, dt, sc, initial=False)

    """Step5 : CoInteg Pairs """
    print("============== Step5 ================")
    _,_, pairs = Cointegrated_Pairs(pr[non_stat].dropna(axis=1,how='any'),cutoff=cutoff) #빼야하나..
    
    return rst_total, pairs, fn, fn_label.astype(str), non_stat

def FundaPriority(fd_pattern, last_label, plimit):
    given_ls = []
    for fd in list(last_label.columns):
        tmp = counting(fd_pattern[lambda x : x['Funda']==fd], plimit=plimit)[1]
        if tmp != 'None':
            given_ls.append(tmp)
        else :
            given_ls.append(['-1'])
    assert last_label.shape[1] == len(given_ls)
    rst = {}
    for idx, row in last_label.iterrows():
        cnt=0
        for i in range(len(given_ls)):
            if str(row[i]) in given_ls[i]:
                cnt+=1
        rst[idx] = cnt
    return rst

