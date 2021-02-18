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
import pyautogui
from statsmodels.tsa.stattools import coint, adfuller
from sklearn.linear_model import LinearRegression
import Loader
ldr = Loader.Loader()
conn = pymysql.connect(host='localhost',user='root',
                                   password='tlqkfdk2',db='INVESTAR',charset='utf8')


def Correlation_v2(log_pr, cutoff=0.05):
    rst = pd.DataFrame(columns=['corr','A','B'])
    codes = list(log_pr.columns)
    for i in range(len(codes)):
        for j in range(len(codes)):
            if i<j:
                corr, pvalue = stats.pearsonr(log_pr[codes[i]], log_pr[codes[j]])
                if pvalue <= cutoff:
                    tmp = pd.DataFrame(columns=['corr','A','B'])
                    tmp.loc[0,'corr'] = corr
                    tmp.loc[0,'A'] = codes[i]
                    tmp.loc[0,'B'] = codes[j]
                    rst = pd.concat([rst, tmp])
    return rst.sort_values(by=['corr'],ascending=False).reset_index(drop=True)


def PartialCorrelation_v2(df, x,y,kospi):
    x = np.array(df[x]).reshape(-1,1)
    y = np.array(df[y]).reshape(-1,1)
    start = df.index[0]
    end = df.index[-1]
    ksp = kospi[(kospi.index>=start)&(kospi.index<=end)]
    z = np.array(ksp['close']).reshape(-1,1)

    # Remove Market Factor by KOSPI
    reg1 = LinearRegression().fit(z, x)
    x_res = x - reg1.predict(z)
    reg2 = LinearRegression().fit(z, y)
    y_res = y - reg2.predict(z)

    corr, pvalue = stats.pearsonr(x_res.reshape(-1,), y_res.reshape(-1,))
    
    return corr, pvalue

def Filtering(dt, sc_codes, by=None):
    conn = pymysql.connect(host='localhost',user='root',
                                   password='tlqkfdk2',db='INVESTAR',charset='utf8')
    code_ls = list(pd.read_sql("select code from company_info",conn).code.values)
    """Basic Filtering"""
    # 자본 총계가 하위 50% or 연평균(?합?) 거래량이 하위 30%
    tdf = pd.read_sql(f"select * from finance_info_copy where date='{dt}' and itm='자본총계'",conn)
    equity_half = tdf['value'].quantile(q=0.1, interpolation='nearest')
    equity_half_ls = set(tdf[lambda x : x['value']<=equity_half].code.values)
    code_ls = list(set(code_ls) - equity_half_ls)

    df = pd.read_sql(f"select code, date, volume from daily_price where date between '{dt[:4]+'-01-01'}' and '{dt[:4]+'-12-31'}'",conn)
    volume_30 = df.groupby(by='code').mean()['volume'].quantile(q=0.1,interpolation='nearest')
    volume_30_ls = set(df[lambda x : x['volume']<=volume_30].code.values)
    code_ls = list(set(code_ls) - volume_30_ls)

    if by == None :
        return list(set(code_ls).intersection(sc_codes))
    else :
        for e in by :
            if e == 'PBR':
                df = pd.read_sql(f"select * from finance_info_copy where date='{dt}' and itm='PBR'",conn)
                code_ls = list(set(code_ls) - set(df[lambda x : x['value']<0].code.values))
            elif e == 'PCR':
                df = pd.read_sql(f"select * from finance_info_copy where date='{dt}' and itm='PCR'",conn)
                code_ls = list(set(code_ls) - set(df[lambda x : x['value']<0].code.values))
                df = pd.read_sql(f"select * from finance_info_copy where date='{dt}' and itm='영업활동현금흐름'",conn)
                code_ls = list(set(code_ls) - set(df[lambda x : x['value']<0].code.values))
                small_cfc = []
                for cd in list(set(df.code.values)):
                    try :
                        if df[(df.code==cd)]['value'].values[0] <= tdf[(tdf.code==cd)]['value'].values[0] * 0.01:
                            small_cfc.append(cd)
                    except :
                        pass
                code_ls = list(set(code_ls) - set(small_cfc))
            elif e == 'POR':
                df = pd.read_sql(f"select * from finance_info_copy where date='{dt}' and itm='POR'",conn)
                code_ls = list(set(code_ls) - set(df[lambda x : x['value']<0].code.values))
                df = pd.read_sql(f"select * from finance_info_copy where date='{dt}' and itm='영업이익'",conn)
                code_ls = list(set(code_ls) - set(df[lambda x : x['value']<0].code.values))
                small_profit = []
                for cd in list(set(df.code.values)):
                    try :
                        if df[(df.code==cd)]['value'].values[0] <= tdf[(tdf.code==cd)]['value'].values[0] * 0.01:
                            small_profit.append(Cd)
                    except :
                        pass
                code_ls = list(set(code_ls) - set(small_profit))
            else :
                raise ValueError("Can't be !!")
        return list(set(code_ls).intersection(sc_codes))

def Filtering_v2(dt, sc_codes, by=None):
    conn = pymysql.connect(host='localhost',user='root',
                                   password='tlqkfdk2',db='INVESTAR',charset='utf8')
    code_ls = list(pd.read_sql("select code from company_info",conn).code.values)
    """Basic Filtering"""
    # 자본 총계가 하위 50% or 연평균(?합?) 거래량이 하위 30%
    tdf = pd.read_sql(f"select * from finance_info_copy where date='{dt}' and itm='자본총계'",conn)
    equity_half = tdf['value'].quantile(q=0.2, interpolation='nearest')
    equity_half_ls = set(tdf[lambda x : x['value']<=equity_half].code.values)
    code_ls = list(set(code_ls) - equity_half_ls)

    df = pd.read_sql(f"select code, date, volume from daily_price where date between '{dt[:4]+'-01-01'}' and '{dt[:4]+'-12-31'}'",conn)
    volume_30 = df.groupby(by='code').mean()['volume'].quantile(q=0.2,interpolation='nearest')
    volume_30_ls = set(df[lambda x : x['volume']<=volume_30].code.values)
    code_ls = list(set(code_ls) - volume_30_ls)
    
    df = pd.read_sql(f"select * from finance_info_copy where date='{dt}' and itm='당기순이익' and value>=0",conn)
    code_ls = list(set(code_ls).intersection(set(df.code.values)))

    if by == None :
        return list(set(code_ls).intersection(sc_codes))
    else :
        for e in by :
            if e == 'PBR':
                df = pd.read_sql(f"select * from finance_info_copy where date='{dt}' and itm='PBR'",conn)
                code_ls = list(set(code_ls) - set(df[lambda x : x['value']<0].code.values))
            elif e == 'PCR':
                df = pd.read_sql(f"select * from finance_info_copy where date='{dt}' and itm='PCR'",conn)
                code_ls = list(set(code_ls) - set(df[lambda x : x['value']<0].code.values))
                df = pd.read_sql(f"select * from finance_info_copy where date='{dt}' and itm='영업활동현금흐름'",conn)
                code_ls = list(set(code_ls) - set(df[lambda x : x['value']<0].code.values))
                small_cfc = []
                for cd in list(set(df.code.values)):
                    try :
                        if df[(df.code==cd)]['value'].values[0] <= tdf[(tdf.code==cd)]['value'].values[0] * 0.01:
                            small_cfc.append(cd)
                    except :
                        pass
                code_ls = list(set(code_ls) - set(small_cfc))
            elif e == 'POR':
                df = pd.read_sql(f"select * from finance_info_copy where date='{dt}' and itm='POR'",conn)
                code_ls = list(set(code_ls) - set(df[lambda x : x['value']<0].code.values))
                df = pd.read_sql(f"select * from finance_info_copy where date='{dt}' and itm='영업이익'",conn)
                code_ls = list(set(code_ls) - set(df[lambda x : x['value']<0].code.values))
                small_profit = []
                for cd in list(set(df.code.values)):
                    try :
                        if df[(df.code==cd)]['value'].values[0] <= tdf[(tdf.code==cd)]['value'].values[0] * 0.01:
                            small_profit.append(Cd)
                    except :
                        pass
                code_ls = list(set(code_ls) - set(small_profit))
            else :
                raise ValueError("Can't be !!")
        return list(set(code_ls).intersection(sc_codes))

def SameSector(rst, comp):
    yes_idx = []
    for idx, row in rst.iterrows():
        try :
            a = row.A
            b = row.B
            a_sc = comp[comp.code==a]['sector'].values[0]
            b_sc = comp[comp.code==b]['sector'].values[0]
            if a_sc == b_sc :
                yes_idx.append(idx)
        except :
            pass
    return rst[rst.index.isin(yes_idx)]

def ExpectedEarning(a, b, eta, pr):
    pr = pr[[a,b]]
    log_pr = np.log(pr)
    spread = log_pr[a] - log_pr[b] * eta
    spread_df = pd.DataFrame(spread, columns=['spread'])
    invade = spread.mean() - 1*spread.std()
    invade_idx = list(spread_df[lambda x : x['spread'] <= invade].index)
    equib = spread.mean()
    equib_idx = list(spread_df[lambda x : x['spread'] >= equib].index)
    
    trade_idx = []
    flagInvade = True
    for idx, row in spread_df.iterrows():
        if flagInvade and row.spread <= invade :
            trade_idx.append(idx)
            flagInvade = False
        elif not flagInvade and row.spread >= equib :
            trade_idx.append(idx)
            flagInvade = True
    
    earnings = spread_df[spread_df.index.isin(trade_idx)].diff(1)
    earnings = earnings.iloc[[i*2+1 for i in range(0,int(len(earnings)/2))],:]
    trade_num = len(earnings)
    earnings = earnings.sum()[0]
            
    return earnings, trade_num

def AddVolMcp(df):
    mcp = pd.read_csv("./FullCache/marcap/marcap-2021-01-14.csv")
    f = lambda x : int(x.replace(',',''))
    mcp['거래량'] = mcp['거래량'].map(f)
    mean_vol = []
    mean_mcp = []
    for idx, row in df.iterrows():
        a = row.A
        b = row.B
        tmp = mcp[(mcp.종목코드.isin([a,b]))][['거래량','시가총액비중(%)']].mean()
        mean_vol.append(tmp[0])
        mean_mcp.append(tmp[1])
    df['AvgVol'] = mean_vol
    df['AvgMcpRate'] = mean_mcp
    return df

def DiffCorrelation(rst, log_pr, cutoff=0.05):
    df = rst.copy()
    df.reset_index(drop=True, inplace=True)
    diff_corr = []
    for idx, row in df.iterrows():
        a = row.A
        b = row.B
        a_v = log_pr[a].diff(1).dropna(axis=0)
        b_v = log_pr[b].diff(1).dropna(axis=0)
        r, pv = stats.pearsonr(a_v, b_v)
        if pv <= cutoff and r >=0 :
            diff_corr.append(r)
        else :
            diff_corr.append(-999.9)
    df['dcorr'] = diff_corr
    return df[lambda x : x['dcorr']!=-999.9]





def PairTrading_v4(pr, start, end, cutoff=0.05, fltr=True):
    """ 0) Log PR """
    pr = pr.astype(float)
    pr = pr[(pr.index>=start)&(pr.index<=end)]
    pr = pr.dropna(axis=1,how='any')
    log_pr = np.log(pr)
    
    drop_ls = []
    for cd in log_pr.columns:
        if len(log_pr[cd].value_counts())==1:
            drop_ls.append(cd)
        elif log_pr[cd].value_counts().iloc[:2].sum() >= len(log_pr[cd]) * 0.3 :
            drop_ls.append(cd)
    print("Almost no change in price in this period -> removed : {}".format(len(drop_ls)))
    log_pr.drop(drop_ls, axis=1, inplace=True)
    
    """ * Filtering """
    if fltr :
        dt = str(int(start[:4])-1)+'-12'
        fltrd_ls = Filtering(dt, list(log_pr.columns), by=None)
        log_pr = log_pr[fltrd_ls]
        print("Filtered by Equity & Volume -> now : {}".format(len(fltrd_ls)))
    
    """ 1) Correlation """
    cor_rst = Correlation_v3(log_pr, cutoff)
    print("Validation w.r.t Correlation : {}".format(len(cor_rst)))
    
    """ 2) Partial Correlation """
    ksp = pd.read_hdf("./FullCache/KOSPI_close.h5")
    ksp = ksp[(ksp.index>=start)&(ksp.index<=end)]
    pcor_rst = pd.DataFrame(columns = ['A','B','corr','pcorr'])
    for idx in range(len(cor_rst)):
        a = cor_rst.loc[idx, 'A']
        b = cor_rst.loc[idx, 'B']
        pcor, ppvalue = PartialCorrelation_v2(log_pr, a, b, ksp)
        if ppvalue <= cutoff :
            pcor_rst.loc[idx,'A'] = a
            pcor_rst.loc[idx,'B'] = b
            pcor_rst.loc[idx,'corr'] = cor_rst.loc[idx,'corr']
            pcor_rst.loc[idx,'pcorr'] = pcor
    pcor_rst.reset_index(drop=True,inplace=True)
    print("Validation w.r.t Partial-Correlation by KOSPI : {}".format(len(pcor_rst)))
        
    """ 3) CoIntegration & 4) Coeff Estimation """
    # Curious about negative cointeg coeff
    cointeg_rst = pd.DataFrame(columns = ['A','B','corr','pcorr','cointeg'])
    for idx in range(len(pcor_rst)):
        a = pcor_rst.loc[idx,'A']
        b = pcor_rst.loc[idx,'B']
        coint_result = coint(log_pr[a], log_pr[b])
        tmp_coeff, tmp_pvalue = coint_result[0], coint_result[1]
        if tmp_pvalue <= cutoff :
            min_pvalue = tmp_pvalue
            best_coeff = tmp_coeff
        else :
            continue
            min_pvalue = 1.0
            best_coeff = -999.9
        
        for eta in [0.1*i for i in range(1,41)]:
            spread = log_pr[a] - log_pr[b] * eta
            adfuller_rst = adfuller(spread)
            if adfuller_rst[1] <= cutoff:
                if adfuller_rst[1] < min_pvalue:
                    min_pvalue = adfuller_rst[1]
                    best_coeff = eta
                else :
                    pass
        if best_coeff != -999.9:
            cointeg_rst.loc[idx,'A'] = a
            cointeg_rst.loc[idx,'B'] = b
            cointeg_rst.loc[idx,'corr'] = pcor_rst.loc[idx,'corr']
            cointeg_rst.loc[idx,'pcorr'] = pcor_rst.loc[idx,'pcorr']
            cointeg_rst.loc[idx,'cointeg'] = best_coeff
    cointeg_rst.reset_index(drop=True, inplace=True)
    print("Validation w.r.t CoIntegration : {}".format(len(cointeg_rst)))
    #print(cointeg_rst)
    
    """ 5) Spread Estimation """
    # 6) no need to check trend since we did stationarity trend test
    valid_idx = []
    for idx in range(len(cointeg_rst)):
        a = cointeg_rst.loc[idx,'A']
        b = cointeg_rst.loc[idx,'B']
        coint_coeff = cointeg_rst.loc[idx,'cointeg']
        spread = log_pr[a] - log_pr[b] * coint_coeff
        
        """ 7) Normality """
        if stats.shapiro(spread)[1] >= cutoff :
            """ 8) Stationarity """
            if adfuller(spread)[1] <= cutoff :
                valid_idx.append(idx)
    pairs = cointeg_rst[cointeg_rst.index.isin(valid_idx)]
    pairs.reset_index(drop=True, inplace=True)
    
    """ 9) Risk & 10) Earnings """
    risk_ls = []
    earning_ls = []
    trade_ls = []
    for idx in range(len(pairs)):
        a = pairs.loc[idx,'A']
        b = pairs.loc[idx,'B']
        coint_coeff = pairs.loc[idx,'cointeg']
        spread = log_pr[a] - log_pr[b] * coint_coeff
        risk_ls.append(spread.diff().std())
        earning, trade_num = ExpectedEarning(a, b, coint_coeff, pr)
        earning_ls.append(earning*100)
        trade_ls.append(trade_num)
    pairs['risk'] = risk_ls
    pairs['earnings(%)'] = earning_ls
    pairs['trade#'] = trade_ls
    
    
    return pairs.sort_values(by=['cointeg'],ascending=False)

def PairTrading_v5(pr, start, end, cutoff=0.05, fltr=True, enter=1):
    print("Initial number of companies : {}".format(len(pr.columns)))
    """ 0) Log PR """
    pr = pr.astype(float)
    pr = pr[(pr.index>=start)&(pr.index<=end)]
    pr = pr.dropna(axis=1,how='any')
    log_pr = np.log(pr)
    
    
    drop_ls = []
    for cd in log_pr.columns:
        if len(log_pr[cd].value_counts())==1:
            drop_ls.append(cd)
        elif log_pr[cd].value_counts().iloc[:2].sum() >= len(log_pr[cd]) * 0.3 :
            drop_ls.append(cd)
    print("Almost no change in price in this period -> removed : {}".format(len(drop_ls)))
    log_pr.drop(drop_ls, axis=1, inplace=True)
    
    """ * Filtering """
    if fltr :
        dt = str(int(start[:4])-1)+'-12'
        fltrd_ls = Filtering_v2(dt, list(log_pr.columns), by=['PBR'])
        log_pr = log_pr[fltrd_ls]
        print("Filtered by NetIncome & PBR & Equity & Volume -> now : {}".format(len(fltrd_ls)))
    
    """ 1) Correlation """
    cor_rst = Correlation_v3(log_pr, cutoff)
    print("Validation w.r.t Correlation : {}".format(len(cor_rst)))
    
    """ 2) Partial Correlation """
    ksp = pd.read_hdf("./FullCache/KOSPI_close.h5")
    ksp = ksp[(ksp.index>=start)&(ksp.index<=end)]
    pcor_rst = pd.DataFrame(columns = ['A','B','corr','pcorr'])
    for idx in range(len(cor_rst)):
        a = cor_rst.loc[idx, 'A']
        b = cor_rst.loc[idx, 'B']
        pcor, ppvalue = PartialCorrelation_v2(log_pr, a, b, ksp)

        if ppvalue <= cutoff and pcor>=0:
            pcor_rst.loc[idx,'A'] = a
            pcor_rst.loc[idx,'B'] = b
            pcor_rst.loc[idx,'corr'] = cor_rst.loc[idx,'corr']
            pcor_rst.loc[idx,'pcorr'] = pcor
    pcor_rst.reset_index(drop=True,inplace=True)
    print("Validation w.r.t Partial-Correlation by KOSPI : {}".format(len(pcor_rst)))
        
    """ 3) CoIntegration & 4) Coeff Estimation """
    # Curious about negative cointeg coeff
    cointeg_rst = pd.DataFrame(columns = ['A','B','corr','pcorr','cointeg'])
    for idx in range(len(pcor_rst)):
        a = pcor_rst.loc[idx,'A']
        b = pcor_rst.loc[idx,'B']
        coint_result = coint(log_pr[a], log_pr[b])
        tmp_coeff, tmp_pvalue = coint_result[0], coint_result[1]
        if tmp_pvalue <= cutoff :
            min_pvalue = tmp_pvalue
            best_coeff = tmp_coeff
        else :
            continue
            min_pvalue = 1.0
            best_coeff = -999.9
        
        for eta in [0.1*i for i in range(1,41)]:
            spread = log_pr[a] - log_pr[b] * eta
            adfuller_rst = adfuller(spread)
            if adfuller_rst[1] <= cutoff:
                if adfuller_rst[1] < min_pvalue:
                    min_pvalue = adfuller_rst[1]
                    best_coeff = eta
                else :
                    pass
        if best_coeff != -999.9:
            cointeg_rst.loc[idx,'A'] = a
            cointeg_rst.loc[idx,'B'] = b
            cointeg_rst.loc[idx,'corr'] = pcor_rst.loc[idx,'corr']
            cointeg_rst.loc[idx,'pcorr'] = pcor_rst.loc[idx,'pcorr']
            cointeg_rst.loc[idx,'cointeg'] = best_coeff
    cointeg_rst.reset_index(drop=True, inplace=True)
    print("Validation w.r.t CoIntegration : {}".format(len(cointeg_rst)))
    #print(cointeg_rst)
    
    """ 5) Spread Estimation """
    # 6) no need to check trend since we did stationarity trend test
    valid_idx = []
    for idx in range(len(cointeg_rst)):
        a = cointeg_rst.loc[idx,'A']
        b = cointeg_rst.loc[idx,'B']
        coint_coeff = cointeg_rst.loc[idx,'cointeg']
        spread = log_pr[a] - log_pr[b] * coint_coeff
        
        """ 7) Normality """
        if stats.shapiro(spread)[1] >= cutoff :
            """ 8) Stationarity """
            if adfuller(spread)[1] <= cutoff :
                valid_idx.append(idx)
    pairs = cointeg_rst[cointeg_rst.index.isin(valid_idx)]
    pairs.reset_index(drop=True, inplace=True)
    print("Validation w.r.t Normality & Stationarity : {}".format(len(pairs)))
    
    """ 9) Risk & 10) Earnings with the number of trade"""
    risk_ls = []
    earning_ls = []
    trade_ls = []
    
    for idx in range(len(pairs)):
        a = pairs.loc[idx,'A']
        b = pairs.loc[idx,'B']
        coint_coeff = pairs.loc[idx,'cointeg']
        spread = log_pr[a] - log_pr[b] * coint_coeff
        risk_ls.append(spread.diff().std())
        earning, trade_num = ExpectedEarning(a, b, coint_coeff, pr, enter=enter)
        earning_ls.append((earning-1)*100)
        trade_ls.append(trade_num)
        
    pairs['risk'] = risk_ls
    pairs['earnings(%)'] = earning_ls
    pairs['trade#'] = trade_ls
    print("Adding Risk & Earnings & #Trade")
    """ 11) Difference Correlation """
    pairs = DiffCorrelation(pairs, log_pr, cutoff=cutoff)
    print("Validation w.r.t Diff-Correlation : {}".format(len(pairs)))
    """ 12) Add AvgVol & AvgMCPRate """
    pairs = AddVolMcp(pairs).sort_values(by=['cointeg'],ascending=False).reset_index(drop=True)
    return pairs

def PairTrading_v6(pr, start, end, up_cut=0.2, down_cut=-0.15, cutoff=0.05, fltr=True, enter=1):
    print("Initial number of companies : {}".format(len(pr.columns)))
    """ 0) Log PR """
    pr = pr.astype(float)
    pr = pr[(pr.index>=start)&(pr.index<=end)]
    pr = pr.dropna(axis=1,how='any')
    log_pr = np.log(pr)
    
    drop_ls = []
    for cd in log_pr.columns:
        if len(log_pr[cd].value_counts())==1:
            drop_ls.append(cd)
        elif log_pr[cd].value_counts().iloc[:2].sum() >= len(log_pr[cd]) * 0.3 :
            drop_ls.append(cd)
    print("Almost no change in price in this period -> removed : {}".format(len(drop_ls)))
    log_pr.drop(drop_ls, axis=1, inplace=True)
    pr.drop(drop_ls, axis=1, inplace=True)
    
    """ * Filtering """
    if fltr :
        dt = str(int(start[:4])-1)+'-12'
        fltrd_ls = Filtering_v2(dt, list(log_pr.columns), by=['PBR'])
        log_pr = log_pr[fltrd_ls]
        pr = pr[fltrd_ls]
        print("Filtered by NetIncome & PBR & Equity & Volume -> now : {}".format(len(fltrd_ls)))
    
    """ * Stop & Loss Cut"""
    pr_rt = pr.pct_change()
    out_ls = []
    for cd in pr_rt.columns:
        if len(pr_rt[lambda x : x[cd] < down_cut]) >= 2 : # 1번이라도 나오면 없애야 하나?
            out_ls.append(cd)
    survive_ls = list(set(pr.columns) - set(out_ls))
    log_pr.drop(drop_ls, axis=1, inplace=True)
    pr.drop(drop_ls, axis=1, inplace=True)
    print("Excluding stop-loss condition with {} : -> now : {}".format(down_cut, len(pr.columns)))
    
    
    """ 1) Correlation """
    cor_rst = Correlation_v3(log_pr, cutoff)
    print("Validation w.r.t Correlation : {}".format(len(cor_rst)))
    
    """ 2) Partial Correlation """
    ksp = pd.read_hdf("./FullCache/KOSPI_close.h5")
    ksp = ksp[(ksp.index>=start)&(ksp.index<=end)]
    pcor_rst = pd.DataFrame(columns = ['A','B','corr','pcorr'])
    for idx in range(len(cor_rst)):
        a = cor_rst.loc[idx, 'A']
        b = cor_rst.loc[idx, 'B']
        pcor, ppvalue = PartialCorrelation_v2(log_pr, a, b, ksp)

        if ppvalue <= cutoff and pcor>=0:
            pcor_rst.loc[idx,'A'] = a
            pcor_rst.loc[idx,'B'] = b
            pcor_rst.loc[idx,'corr'] = cor_rst.loc[idx,'corr']
            pcor_rst.loc[idx,'pcorr'] = pcor
    pcor_rst.reset_index(drop=True,inplace=True)
    print("Validation w.r.t Partial-Correlation by KOSPI : {}".format(len(pcor_rst)))
        
    """ 3) CoIntegration & 4) Coeff Estimation """
    # Curious about negative cointeg coeff
    cointeg_rst = pd.DataFrame(columns = ['A','B','corr','pcorr','cointeg'])
    for idx in range(len(pcor_rst)):
        a = pcor_rst.loc[idx,'A']
        b = pcor_rst.loc[idx,'B']
        coint_result = coint(log_pr[a], log_pr[b])
        tmp_coeff, tmp_pvalue = coint_result[0], coint_result[1]
        if tmp_pvalue <= cutoff :
            min_pvalue = tmp_pvalue
            best_coeff = tmp_coeff
        else :
            continue
            min_pvalue = 1.0
            best_coeff = -999.9
        
        for eta in [0.1*i for i in range(1,41)]:
            spread = log_pr[a] - log_pr[b] * eta
            adfuller_rst = adfuller(spread)
            if adfuller_rst[1] <= cutoff:
                if adfuller_rst[1] < min_pvalue:
                    min_pvalue = adfuller_rst[1]
                    best_coeff = eta
                else :
                    pass
        if best_coeff != -999.9:
            cointeg_rst.loc[idx,'A'] = a
            cointeg_rst.loc[idx,'B'] = b
            cointeg_rst.loc[idx,'corr'] = pcor_rst.loc[idx,'corr']
            cointeg_rst.loc[idx,'pcorr'] = pcor_rst.loc[idx,'pcorr']
            cointeg_rst.loc[idx,'cointeg'] = best_coeff
    cointeg_rst.reset_index(drop=True, inplace=True)
    print("Validation w.r.t CoIntegration : {}".format(len(cointeg_rst)))
    #print(cointeg_rst)
    
    """ 5) Spread Estimation """
    # 6) no need to check trend since we did stationarity trend test
    valid_idx = []
    for idx in range(len(cointeg_rst)):
        a = cointeg_rst.loc[idx,'A']
        b = cointeg_rst.loc[idx,'B']
        coint_coeff = cointeg_rst.loc[idx,'cointeg']
        spread = log_pr[a] - log_pr[b] * coint_coeff
        
        """ 7) Normality """
        if stats.shapiro(spread)[1] >= cutoff :
            """ 8) Stationarity """
            if adfuller(spread)[1] <= cutoff :
                valid_idx.append(idx)
    pairs = cointeg_rst[cointeg_rst.index.isin(valid_idx)]
    pairs.reset_index(drop=True, inplace=True)
    print("Validation w.r.t Normality & Stationarity : {}".format(len(pairs)))
    
    """ 9) Risk & 10) Earnings with the number of trade"""
    risk_ls = []
    earning_ls = []
    trade_ls = []
    
    for idx in range(len(pairs)):
        a = pairs.loc[idx,'A']
        b = pairs.loc[idx,'B']
        coint_coeff = pairs.loc[idx,'cointeg']
        spread = log_pr[a] - log_pr[b] * coint_coeff
        risk_ls.append(spread.diff().std())
        earning, trade_num = ExpectedEarning(a, b, coint_coeff, pr, enter=enter)
        earning_ls.append((earning-1)*100)
        trade_ls.append(trade_num)
        
    pairs['risk'] = risk_ls
    pairs['earnings(%)'] = earning_ls
    pairs['trade#'] = trade_ls
    
    A_up = []
    B_up = []
    for idx, row in pairs.iterrows():
        a = row.A
        b = row.B
        A_up.append(len(pr_rt[lambda x : x[a]>=up_cut]))
        B_up.append(len(pr_rt[lambda x : x[b]>=up_cut]))
    pairs['A_Over_{}%'.format(up_cut*100)] = A_up
    pairs['B_Over_{}%'.format(up_cut*100)] = B_up
    
    print("Adding Risk & Earnings & #Trade & #UpCut")
    """ 11) Difference Correlation """
    pairs = DiffCorrelation(pairs, log_pr, cutoff=cutoff)
    print("Validation w.r.t Diff-Correlation : {}".format(len(pairs)))
    """ 12) Add AvgVol & AvgMCPRate """
    pairs = AddVolMcp(pairs).sort_values(by=['cointeg'],ascending=False).reset_index(drop=True)
    return pairs