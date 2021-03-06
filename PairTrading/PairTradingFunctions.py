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
                                   password='******',db='INVESTAR',charset='utf8')


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

def ExpectedEarning_v2(a, b, eta, pr, enter=1, position='A'):
    with open("TradingDates.pickle", "rb") as fr:
        trading_dates = pickle.load(fr)
    pr = pr[[a,b]]
    log_pr = np.log(pr)
    spread = log_pr[a] - log_pr[b] * eta
    spread_df = pd.DataFrame(spread, columns=['spread'])
    if position=='A':
        invade = spread.mean() - enter*spread.std()
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
                
    elif position=='B':
        invade = spread.mean() + enter*spread.std()
        invade_idx = list(spread_df[lambda x : x['spread'] >= invade].index)
        equib = spread.mean()
        equib_idx = list(spread_df[lambda x : x['spread'] <= equib].index)
        
        trade_idx = []
        flagInvade = True
        for idx, row in spread_df.iterrows():
            if flagInvade and row.spread >= invade :
                trade_idx.append(idx)
                flagInvade = False
            elif not flagInvade and row.spread <= equib :
                trade_idx.append(idx)
                flagInvade = True
    else :
        raise ValueError("Can't Be.")

    
    earnings = spread_df[spread_df.index.isin(trade_idx)].diff(1)
    earnings = earnings.iloc[[i*2+1 for i in range(0,int(len(earnings)/2))],:]
    earning_dates = list(earnings.index)
    max_term = 0
    for idx in range(len(earning_dates)):
        if idx != len(earning_dates)-1 :
            term = trading_dates.index(earning_dates[idx+1]) - trading_dates.index(earning_dates[idx])
            if max_term <= term:
                max_term = term
    
    trade_num = len(earnings)
    if position=='A':
        earnings = np.exp(((np.log(earnings+1)).cumsum().iloc[-1,0]) / trade_num)
    elif position=='B':
        earnings = np.exp(((np.log((earnings*(-1))+1)).cumsum().iloc[-1,0]) / trade_num)
    
    return earnings, trade_num, max_term



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

def PairTrading_v7(pr, start, end, up_cut=0.2, down_cut=-0.15, cutoff=0.05, fltr=True, enter=1):
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
    log_pr.drop(out_ls, axis=1, inplace=True)
    pr.drop(out_ls, axis=1, inplace=True)
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
    A_earning_ls = []
    A_trade_ls = []
    A_maxterm_ls = []
    B_earning_ls = []
    B_trade_ls = []
    B_maxterm_ls = []
    
    for idx in range(len(pairs)):
        a = pairs.loc[idx,'A']
        b = pairs.loc[idx,'B']
        coint_coeff = pairs.loc[idx,'cointeg']
        spread = log_pr[a] - log_pr[b] * coint_coeff
        risk_ls.append(spread.diff().std())
        
        earning, trade_num, max_term = ExpectedEarning_v2(a, b, coint_coeff, pr, enter=enter, position='A')
        A_earning_ls.append((earning-1)*100)
        A_trade_ls.append(trade_num)
        A_maxterm_ls.append(max_term)
        
        earning, trade_num, max_term = ExpectedEarning_v2(a, b, coint_coeff, pr, enter=enter, position='B')
        B_earning_ls.append((earning-1)*100)
        B_trade_ls.append(trade_num)
        B_maxterm_ls.append(max_term)
        
        
    pairs['risk'] = risk_ls
    pairs['A_earnings(%)'] = A_earning_ls
    pairs['A_trade#'] = A_trade_ls
    pairs['A_maxterm'] = A_maxterm_ls
    pairs['B_earnings(%)'] = B_earning_ls
    pairs['B_trade#'] = B_trade_ls
    pairs['B_maxterm'] = B_maxterm_ls
    
    A_up = []
    B_up = []
    for idx, row in pairs.iterrows():
        a = row.A
        b = row.B
        A_up.append(len(pr_rt[lambda x : x[a]>=up_cut]))
        B_up.append(len(pr_rt[lambda x : x[b]>=up_cut]))
    pairs['A_Over{}%'.format(int(up_cut*100))] = A_up
    pairs['B_Over{}%'.format(int(up_cut*100))] = B_up
    
    print("Adding Risk & Earnings & #Trade & #UpCut")
    """ 11) Difference Correlation """
    pairs = DiffCorrelation(pairs, log_pr, cutoff=cutoff)
    print("Validation w.r.t Diff-Correlation : {}".format(len(pairs)))
    """ 12) Add AvgVol & AvgMCPRate """
    pairs = AddVolMcp(pairs).sort_values(by=['cointeg'],ascending=False).reset_index(drop=True)
    return pairs

def PairTrading_v8(pr, start, end, real_start, real_end, up_cut=0.2, down_cut=-0.15, cutoff=0.05, fltr=True, enter=1):
    print("Initial number of companies : {}".format(len(pr.columns)))
    """ 0) Log PR """
    pr = pr.astype(float)
    real_pr = pr[(pr.index>=real_start)&(pr.index<=real_end)]
    real_pr.dropna(axis=1,how='any')
    log_real_pr = np.log(real_pr)
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
    best_funda = pd.read_hdf("FullCache/BestFundaPattern.h5")
    sectors = list(set(best_funda.index))
    conn = pymysql.connect(host='localhost',user='root',
                                   password='tlqkfdk2',db='INVESTAR',charset='utf8')
    company = pd.read_sql("select * from company_info",conn)
    sector_codes = list(company[company.sector.isin(sectors)].code.values)
    most_sectors = list(set(log_pr.columns).intersection(set(sector_codes)))
    pr = pr[most_sectors]
    log_pr = log_pr[most_sectors]
    print("Filtered by Most Funda Sectors -> now : {}".format(len(pr.columns)))
    
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
    #survive_ls = list(set(pr.columns) - set(out_ls))
    log_pr.drop(out_ls, axis=1, inplace=True)
    pr.drop(out_ls, axis=1, inplace=True)
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
    A_earning_ls = []
    A_trade_ls = []
    A_maxterm_ls = []
    B_earning_ls = []
    B_trade_ls = []
    B_maxterm_ls = []
    
    for idx in range(len(pairs)):
        a = pairs.loc[idx,'A']
        b = pairs.loc[idx,'B']
        coint_coeff = pairs.loc[idx,'cointeg']
        spread = log_pr[a] - log_pr[b] * coint_coeff
        return spread, log_pr[a], log_pr[b], coint_coeff
        risk_ls.append(spread.diff().std())
        try :
            earning, trade_num, max_term = ExpectedEarning_v2(a, b, coint_coeff, pr, enter=enter, position='A')
            A_earning_ls.append((earning-1)*100)
            A_trade_ls.append(trade_num)
            A_maxterm_ls.append(max_term)
        except :
            A_earning_ls.append(-999)
            A_trade_ls.append(-999)
            A_maxterm_ls.append(-999)
        try:
            earning, trade_num, max_term = ExpectedEarning_v2(a, b, coint_coeff, pr, enter=enter, position='B')
            B_earning_ls.append((earning-1)*100)
            B_trade_ls.append(trade_num)
            B_maxterm_ls.append(max_term)
        except:
            B_earning_ls.append(-999)
            B_trade_ls.append(-999)
            B_maxterm_ls.append(-999)

        
    pairs['risk'] = risk_ls
    pairs['A_earnings(%)'] = A_earning_ls
    pairs['A_trade#'] = A_trade_ls
    pairs['A_maxterm'] = A_maxterm_ls
    pairs['B_earnings(%)'] = B_earning_ls
    pairs['B_trade#'] = B_trade_ls
    pairs['B_maxterm'] = B_maxterm_ls
    
#     A_up = []
#     B_up = []
#     for idx, row in pairs.iterrows():
#         a = row.A
#         b = row.B
#         A_up.append(len(pr_rt[lambda x : x[a]>=up_cut]))
#         B_up.append(len(pr_rt[lambda x : x[b]>=up_cut]))
#     pairs['A_Over{}%'.format(int(up_cut*100))] = A_up
#     pairs['B_Over{}%'.format(int(up_cut*100))] = B_up
    
    print("Adding Risk & Earnings & #Trade & #UpCut")
    """ 11) Difference Correlation """
    pairs = DiffCorrelation(pairs, log_pr, cutoff=cutoff)
    print("Validation w.r.t Diff-Correlation : {}".format(len(pairs)))
    """ 12) Add AvgVol & AvgMCPRate """
    pairs = AddVolMcp(pairs).sort_values(by=['cointeg'],ascending=False).reset_index(drop=True)
    
    
    """ 13) Checking whole results on TEST Period """
    """ CoInteg + Stationary + Normality + Corr """
    survive_ls = []
    for idx, row in pairs.iterrows():
        a = row.A
        b = row.B
        eta = row.cointeg
        a_val = log_real_pr[a]
        b_val = log_real_pr[b]
        real_spread = a_val - b_val * eta
        if coint(a_val, b_val)[1] <= cutoff:
            if adfuller(real_spread)[1] <= cutoff:
                if stats.shapiro(real_spread)[1] >= cutoff :
                    survive_ls.append(idx)
                    #tmp = stats.pearsonr(a_val, b_val)
                    #if tmp[0]>=0 and tmp[1] <= cutoff:
                    #    survive_ls.append(idx)
                        

    return pairs, pairs[pairs.index.isin(survive_ls)]

def TradeVisual(A, B, eta, start, end, pr, enter=1):
    loglv22 = np.log(pr)
    qq_log = loglv22[(loglv22.index>=start)&(loglv22.index<=end)]
    qq_full = loglv22[(loglv22.index>=start)]
    aa = qq_log[A]
    bb = qq_log[B]
    aa_full = qq_full[A]
    bb_full = qq_full[B]
    etaa = eta
    spread = aa - bb * etaa
    spread_full = aa_full - bb_full * etaa
    fig = plt.figure(figsize=(12,10))
    ax1 = fig.add_subplot(2, 1, 1)
    ax2 = fig.add_subplot(2, 1, 2)
    
    print("{} & {}".format(ldr.FindNameByCode(A),ldr.FindNameByCode(B)))
    ax1.plot(spread.reset_index().index, spread.values,color='black')
    ax1.plot(spread_full.reset_index().index[len(spread):], spread_full.values[len(spread):],color='black')

    ax1.hlines(spread.mean(),0,len(spread_full),colors='r',linewidth=3)
    ax1.hlines(spread.mean()+enter*spread.std(),0,len(spread_full),colors='y',linewidth=3)
    ax1.hlines(spread.mean()-enter*spread.std(),0,len(spread_full),colors='y',linewidth=3)
    ax1.vlines(len(spread),min(spread_full),max(spread_full),colors='g',linewidth=3)

    ax2.plot(spread_full.reset_index().index, (aa_full - aa_full.mean())/aa_full.std(),label='A')
    ax2.plot(spread_full.reset_index().index, (bb_full - bb_full.mean())/bb_full.std(), label='B')
    ax2.vlines(len(spread),-4,4,colors='g',linewidth=3)
    ax2.legend(loc='upper left')
    plt.show()
    return True

def PairTrading_v9(pr, start, end, real_start, real_end, byFunda='all', up_cut=0.2, down_cut=-0.15, cutoff=0.05, fltr=True, enter=1):
    print("Initial number of companies : {}".format(len(pr.columns)))
    """ 0) Log PR """
    pr = pr.astype(float)
    real_pr = pr[(pr.index>=real_start)&(pr.index<=real_end)]
    real_pr.dropna(axis=1,how='any')
    log_real_pr = np.log(real_pr)
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
    
    """ * Filtering """ # Funda 정보가 부족하다고 제외할 수 있는 것일까..
#     best_funda = pd.read_hdf("FullCache/BestFundaPattern.h5")
#     sectors = list(set(best_funda.index))
#     conn = pymysql.connect(host='localhost',user='root',
#                                    password='tlqkfdk2',db='INVESTAR',charset='utf8')
#     company = pd.read_sql("select * from company_info",conn)
#     sector_codes = list(company[company.sector.isin(sectors)].code.values)
#     most_sectors = list(set(log_pr.columns).intersection(set(sector_codes)))
#     pr = pr[most_sectors]
#     log_pr = log_pr[most_sectors]
#     print("Filtered by Most Funda Sectors -> now : {}".format(len(pr.columns)))
    
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
    log_pr.drop(out_ls, axis=1, inplace=True)
    pr.drop(out_ls, axis=1, inplace=True)
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
    A_earning_ls = []
    A_trade_ls = []
    A_maxterm_ls = []
    B_earning_ls = []
    B_trade_ls = []
    B_maxterm_ls = []
    
    for idx in range(len(pairs)):
        a = pairs.loc[idx,'A']
        b = pairs.loc[idx,'B']
        coint_coeff = pairs.loc[idx,'cointeg']
        spread = log_pr[a] - log_pr[b] * coint_coeff
        risk_ls.append(spread.diff().std())
        try :
            earning, trade_num, max_term = ExpectedEarning_v2(a, b, coint_coeff, pr, enter=enter, position='A')
            A_earning_ls.append((earning-1)*100)
            A_trade_ls.append(trade_num)
            A_maxterm_ls.append(max_term)
        except :
            A_earning_ls.append(-999)
            A_trade_ls.append(-999)
            A_maxterm_ls.append(-999)
        try:
            earning, trade_num, max_term = ExpectedEarning_v2(a, b, coint_coeff, pr, enter=enter, position='B')
            B_earning_ls.append((earning-1)*100)
            B_trade_ls.append(trade_num)
            B_maxterm_ls.append(max_term)
        except:
            B_earning_ls.append(-999)
            B_trade_ls.append(-999)
            B_maxterm_ls.append(-999)

        
    pairs['risk'] = risk_ls
    pairs['A_earnings(%)'] = A_earning_ls
    pairs['A_trade#'] = A_trade_ls
    pairs['A_maxterm'] = A_maxterm_ls
    pairs['B_earnings(%)'] = B_earning_ls
    pairs['B_trade#'] = B_trade_ls
    pairs['B_maxterm'] = B_maxterm_ls
    
#     A_up = []
#     B_up = []
#     for idx, row in pairs.iterrows():
#         a = row.A
#         b = row.B
#         A_up.append(len(pr_rt[lambda x : x[a]>=up_cut]))
#         B_up.append(len(pr_rt[lambda x : x[b]>=up_cut]))
#     pairs['A_Over{}%'.format(int(up_cut*100))] = A_up
#     pairs['B_Over{}%'.format(int(up_cut*100))] = B_up
    
    print("Adding Risk & Earnings & #Trade & #UpCut")
    """ 11) Difference Correlation """
    pairs = DiffCorrelation(pairs, log_pr, cutoff=cutoff)
    print("Validation w.r.t Diff-Correlation : {}".format(len(pairs)))
    """ 12) Add AvgVol & AvgMCPRate """
    pairs = AddVolMcp(pairs).sort_values(by=['cointeg'],ascending=False).reset_index(drop=True)
    
    
    """ 13) Checking whole results on TEST Period """
    """ CoInteg + Stationary + Normality + Corr """
    survive_ls = []
    for idx, row in pairs.iterrows():
        a = row.A
        b = row.B
        eta = row.cointeg
        a_val = log_real_pr[a]
        b_val = log_real_pr[b]
        real_spread = a_val - b_val * eta
        if coint(a_val, b_val)[1] <= cutoff:
            if adfuller(real_spread)[1] <= cutoff:
                if stats.shapiro(real_spread)[1] >= cutoff :
                    survive_ls.append(idx)
                    #tmp = stats.pearsonr(a_val, b_val)
                    #if tmp[0]>=0 and tmp[1] <= cutoff:
                    #    survive_ls.append(idx)
    
    """ 14) Best Funda Quantile Pattern Check """
    FundaScore_A = []
    FundaScore_B = []
    if int(real_end[4:6])>=4:
        dt = str(int(real_end[:4])-1)+'-12'
        score_dict = FundaMatch(dt, byFunda=byFunda)
        for idx, row in pairs.iterrows():
            a = row.A
            b = row.B
            FundaScore_A.append(score_dict[a])
            FundaScore_B.append(score_dict[b])
    else :
        dt = str(int(real_end[:4])-2)+'-12'
        score_dict = FundaMatch(dt, byFunda=byFunda)
        for idx, row in pairs.iterrows():
            a = row.A
            b = row.B
            FundaScore_A.append(score_dict[a])
            FundaScore_B.append(score_dict[b])
    pairs['A_Funda(+)'] = FundaScore_A
    pairs['B_Funda(+)'] = FundaScore_B
        
    return pairs, pairs[pairs.index.isin(survive_ls)]

def TradeVisual(A, B, eta, start, end, pr, enter=1):
    loglv22 = np.log(pr)
    qq_log = loglv22[(loglv22.index>=start)&(loglv22.index<=end)]
    qq_full = loglv22[(loglv22.index>=start)]
    aa = qq_log[A]
    bb = qq_log[B]
    aa_full = qq_full[A]
    bb_full = qq_full[B]
    etaa = eta
    spread = aa - bb * etaa
    spread_full = aa_full - bb_full * etaa
    fig = plt.figure(figsize=(12,10))
    ax1 = fig.add_subplot(2, 1, 1)
    ax2 = fig.add_subplot(2, 1, 2)
    
    print("{} & {}".format(ldr.FindNameByCode(A),ldr.FindNameByCode(B)))
    ax1.plot(spread.reset_index().index, spread.values,color='black')
    ax1.plot(spread_full.reset_index().index[len(spread):], spread_full.values[len(spread):],color='black')

    ax1.hlines(spread.mean(),0,len(spread_full),colors='r',linewidth=3)
    ax1.hlines(spread.mean()+enter*spread.std(),0,len(spread_full),colors='y',linewidth=3)
    ax1.hlines(spread.mean()-enter*spread.std(),0,len(spread_full),colors='y',linewidth=3)
    ax1.vlines(len(spread),min(spread_full),max(spread_full),colors='g',linewidth=3)

    ax2.plot(spread_full.reset_index().index, (aa_full - aa_full.mean())/aa_full.std(),label='A')
    ax2.plot(spread_full.reset_index().index, (bb_full - bb_full.mean())/bb_full.std(), label='B')
    ax2.vlines(len(spread),-4,4,colors='g',linewidth=3)
    ax2.legend(loc='upper left')
    plt.show()
    return True

def BestPairSelector(pair_df,comp):
    tmp = pair_df.copy()
    rank_dict = {}
    for idx in list(tmp.index):
        rank_dict[idx] = 0
    tmp['abs_cointeg'] = abs(tmp['cointeg']-1)
    tmp['best_funda'] = tmp['A_Funda(+)'] + tmp['B_Funda(+)']
    sector_ls = []
    for idx, row in tmp.iterrows():
        a = row.A
        b = row.B
        if comp[lambda x : x['code']==a].sector.values[0] == comp[lambda x : x['code']==b].sector.values[0]:
            sector_ls.append('Y')
        else :
            sector_ls.append('N')
        #sector_ls.append(comp[lambda x : x['code']==a].sector.values[0] + '-' + comp[lambda x : x['code']==b].sector.values[0])
    tmp['sector'] = sector_ls
    
    T = len(pair_df)
    """ CoInteg Coeff """
    for ith, idx in enumerate(list(tmp.sort_values(by=['abs_cointeg']).index)):
        rank_dict[idx] += T - ith
    for ith, idx in enumerate(list(tmp.sort_values(by=['corr'],ascending=False).index)):
        rank_dict[idx] += T - ith
    for ith, idx in enumerate(list(tmp.sort_values(by=['pcorr'],ascending=False).index)):
        rank_dict[idx] += T - ith
    for ith, idx in enumerate(list(tmp.sort_values(by=['dcorr'],ascending=False).index)):
        rank_dict[idx] += T - ith
    for ith, idx in enumerate(list(tmp.sort_values(by=['best_funda'],ascending=False).index)):
        rank_dict[idx] += T - ith
    for ith, idx in enumerate(list(tmp.sort_values(by=['AvgVol'],ascending=False).index)):
        rank_dict[idx] += T - ith
    for ith, idx in enumerate(list(tmp.sort_values(by=['risk'],ascending=False).index)):
        rank_dict[idx] += T - ith
    for ith, idx in enumerate(list(tmp.sort_values(by=['A_trade#'],ascending=False).index)):
        rank_dict[idx] += T - ith
    for ith, idx in enumerate(list(tmp.sort_values(by=['B_trade#'],ascending=False).index)):
        rank_dict[idx] += T - ith
    for ith, idx in enumerate(list(tmp.sort_values(by=['A_maxterm']).index)):
        rank_dict[idx] += T - ith
    for ith, idx in enumerate(list(tmp.sort_values(by=['B_maxterm']).index)):
        rank_dict[idx] += T - ith
    for ith, idx in enumerate(list(tmp.sort_values(by=['A_earnings(%)'],ascending=False).index)):
        rank_dict[idx] += T - ith
    for ith, idx in enumerate(list(tmp.sort_values(by=['B_earnings(%)'],ascending=False).index)):
        rank_dict[idx] += T - ith
    tmp['SCORE'] = tmp.index.map(rank_dict)
    tmp = tmp.drop(['best_funda','abs_cointeg'],axis=1)
    tmp = tmp.sort_values(by=['SCORE'],ascending=False).reset_index(drop=True)
    return tmp

def SpreadDiverge(a, b, eta, check_start, train_pr, test_pr, enter=2, ndays=25):
    train_pr = train_pr[[a,b]]
    log_pr = np.log(train_pr)
    spread = log_pr[a] - log_pr[b] * eta
    mu = spread.mean()
    sigma = spread.std()
    test_pr = test_pr[test_pr.index>=check_start]
    test_pr = test_pr[[a,b]]
    test_log_pr = np.log(test_pr)
    test_spread = test_log_pr[a] - test_log_pr[b] * eta
    
    spread_df = pd.DataFrame(test_spread, columns=['spread'])
    UpCut = mu + enter * sigma
    DownCut = mu - enter * sigma
    index_ls = list(spread_df.index)
    Down_ls = []
    Up_ls = []
    for idx, row in spread_df.iterrows():
        if row.spread < DownCut :
            Down_ls.append(-1)
        else :
            Down_ls.append(+1)
            
        if row.spread > UpCut :
            Up_ls.append(-1)
        else :
            Up_ls.append(+1)
    sub = spread_df.copy()
    sub['Down'] = Down_ls
    sub['Up'] = Up_ls
    max_term = 0
    new_idx = 0
    for idx in index_ls:
        if sub.loc[idx,'Down'] == -1:
            tmp = sub[sub.index>idx][lambda x : x['Down']==+1]
            if len(tmp)!= 0:
                new_idx = tmp.index[0]
                term = index_ls.index(new_idx) - index_ls.index(idx)
                if term > max_term :
                    max_term = term
            else :
                new_idx = index_ls[-1]
                term = index_ls.index(new_idx) - index_ls.index(idx) + 1
                if term > max_term :
                    max_term = term
                break
    if max_term >= ndays :
        print("Down Divergence")
        return True, max_term
    
    umax_term = 0
    new_idx = 0
    for idx in index_ls:
        if sub.loc[idx,'Up'] == -1:
            tmp = sub[sub.index>idx][lambda x : x['Up']==+1]
            if len(tmp)!= 0:
                new_idx = tmp.index[0]
                term = index_ls.index(new_idx) - index_ls.index(idx)
                if term > umax_term :
                    umax_term = term
            else :
                new_idx = index_ls[-1]
                term = index_ls.index(new_idx) - index_ls.index(idx) + 1
                if term > umax_term :
                    umax_term = term
                break
    if umax_term >= ndays :
        print("Up Divergence")
        return True, umax_term
    if max_term < umax_term :
        max_term = umax_term
    print("No Divergence")
    return False, max_term

def PairTrading_v10(pr, start, end, real_start, real_end, byFunda='all', up_cut=0.2, down_cut=-0.15, cutoff=0.05, fltr=True, enter=1, ndays=22):
    print("Initial number of companies : {}".format(len(pr.columns)))
    """ 0) Log PR """
    pr = pr.astype(float)
    real_pr = pr[(pr.index>=real_start)&(pr.index<=real_end)]
    real_pr.dropna(axis=1,how='any')
    log_real_pr = np.log(real_pr)
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
    
    """ * Filtering """ # Funda 정보가 부족하다고 제외할 수 있는 것일까..
#     best_funda = pd.read_hdf("FullCache/BestFundaPattern.h5")
#     sectors = list(set(best_funda.index))
#     conn = pymysql.connect(host='localhost',user='root',
#                                    password='tlqkfdk2',db='INVESTAR',charset='utf8')
#     company = pd.read_sql("select * from company_info",conn)
#     sector_codes = list(company[company.sector.isin(sectors)].code.values)
#     most_sectors = list(set(log_pr.columns).intersection(set(sector_codes)))
#     pr = pr[most_sectors]
#     log_pr = log_pr[most_sectors]
#     print("Filtered by Most Funda Sectors -> now : {}".format(len(pr.columns)))
    
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
    log_pr.drop(out_ls, axis=1, inplace=True)
    pr.drop(out_ls, axis=1, inplace=True)
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
    A_earning_ls = []
    A_trade_ls = []
    A_maxterm_ls = []
    B_earning_ls = []
    B_trade_ls = []
    B_maxterm_ls = []
    
    for idx in range(len(pairs)):
        a = pairs.loc[idx,'A']
        b = pairs.loc[idx,'B']
        coint_coeff = pairs.loc[idx,'cointeg']
        spread = log_pr[a] - log_pr[b] * coint_coeff
        risk_ls.append(spread.diff().std())
        try :
            earning, trade_num, max_term = ExpectedEarning_v2(a, b, coint_coeff, pr, enter=enter, position='A')
            A_earning_ls.append((earning-1)*100)
            A_trade_ls.append(trade_num)
            A_maxterm_ls.append(max_term)
        except :
            A_earning_ls.append(-999)
            A_trade_ls.append(-999)
            A_maxterm_ls.append(-999)
        try:
            earning, trade_num, max_term = ExpectedEarning_v2(a, b, coint_coeff, pr, enter=enter, position='B')
            B_earning_ls.append((earning-1)*100)
            B_trade_ls.append(trade_num)
            B_maxterm_ls.append(max_term)
        except:
            B_earning_ls.append(-999)
            B_trade_ls.append(-999)
            B_maxterm_ls.append(-999)

        
    pairs['risk'] = risk_ls
    pairs['A_earnings(%)'] = A_earning_ls
    pairs['A_trade#'] = A_trade_ls
    pairs['A_maxterm'] = A_maxterm_ls
    pairs['B_earnings(%)'] = B_earning_ls
    pairs['B_trade#'] = B_trade_ls
    pairs['B_maxterm'] = B_maxterm_ls
    
#     A_up = []
#     B_up = []
#     for idx, row in pairs.iterrows():
#         a = row.A
#         b = row.B
#         A_up.append(len(pr_rt[lambda x : x[a]>=up_cut]))
#         B_up.append(len(pr_rt[lambda x : x[b]>=up_cut]))
#     pairs['A_Over{}%'.format(int(up_cut*100))] = A_up
#     pairs['B_Over{}%'.format(int(up_cut*100))] = B_up
    
    print("Adding Risk & Earnings & #Trade & #UpCut")
    """ 11) Difference Correlation """
    pairs = DiffCorrelation(pairs, log_pr, cutoff=cutoff)
    print("Validation w.r.t Diff-Correlation : {}".format(len(pairs)))
    """ 12) Add AvgVol & AvgMCPRate """
    pairs = AddVolMcp(pairs).sort_values(by=['cointeg'],ascending=False).reset_index(drop=True)
    
    
    """ 13) Checking whole results on TEST Period """
    """ CoInteg + Stationary + Normality + Corr """
    survive_ls = []
    for idx, row in pairs.iterrows():
        a = row.A
        b = row.B
        eta = row.cointeg
        a_val = log_real_pr[a]
        b_val = log_real_pr[b]
        real_spread = a_val - b_val * eta
        if coint(a_val, b_val)[1] <= cutoff:
            if adfuller(real_spread)[1] <= cutoff:
                if stats.shapiro(real_spread)[1] >= cutoff :
                    survive_ls.append(idx)
                    #tmp = stats.pearsonr(a_val, b_val)
                    #if tmp[0]>=0 and tmp[1] <= cutoff:
                    #    survive_ls.append(idx)
    
    """ 14) Best Funda Quantile Pattern Check """
    FundaScore_A = []
    FundaScore_B = []
    if int(real_end[4:6])>=4:
        dt = str(int(real_end[:4])-1)+'-12'
        score_dict = FundaMatch(dt, byFunda=byFunda)
        for idx, row in pairs.iterrows():
            a = row.A
            b = row.B
            FundaScore_A.append(score_dict[a])
            FundaScore_B.append(score_dict[b])
    else :
        dt = str(int(real_end[:4])-2)+'-12'
        score_dict = FundaMatch(dt, byFunda=byFunda)
        for idx, row in pairs.iterrows():
            a = row.A
            b = row.B
            FundaScore_A.append(score_dict[a])
            FundaScore_B.append(score_dict[b])
    pairs['A_Funda(+)'] = FundaScore_A
    pairs['B_Funda(+)'] = FundaScore_B
    
    """ 15) Spread Divergence Check """
    divcheck_idx = []
    for idx, row in pairs.iterrows():
        a = row.A
        b = row.B
        coint_coeff = row.cointeg
        if SpreadDiverge(a, b, coint_coeff, end, pr, real_pr, enter=2, ndays=ndays)[0] == False :
            divcheck_idx.append(idx)
    pairs = pairs[pairs.index.isin(divcheck_idx)]
    print("Validation w.r.t Spread Divergence : {} survived.".format(len(divcheck_idx)))
    
        
    return pairs, pairs[pairs.index.isin(survive_ls)]

def PriceCompare(rst, pr, diff=10000, min_pr=8000, max_pr=150000):
    valid_idx = []
    for idx, row in rst.iterrows():
        a = row.A
        b = row.B
        a_pr = pr[a].mean()
        b_pr = pr[b].mean()
        if a_pr >= min_pr and b_pr >= min_pr and a_pr <= max_pr and b_pr <= max_pr:
            if abs(a_pr - b_pr) <= diff:
                valid_idx.append(idx)
    return rst[rst.index.isin(valid_idx)]

def UniverseFilter(dt):
    conn = pymysql.connect(host='localhost',user='root',
                                   password='tlqkfdk2',db='INVESTAR',charset='utf8')
    code_ls = list(pd.read_sql("select code from company_info",conn).code.values)
    start_year = str(1+int(dt[:4]))
    start_date = start_year + '-04-01'
    end_year = str(2+int(dt[:4]))
    end_date = end_year + '03-31'
    """ 1) Volume over MEDIAN """
    try :
        pr = pd.read_hdf("./FullCache/Price/price_{}.h5".format(end_year))
    except :
        pr = pd.read_hdf("./FullCache/Price/price_{}.h5".format(start_year))
    vol_last60 = pr[pr.DATE.isin(sorted(list(set(pr.DATE.values)))[-60:])].groupby('CODE').mean()
    vol_median = vol_last60[['volume']].describe().loc['50%','volume']
    vol_ls = list(vol_last60[lambda x : x['volume']>=vol_median].index)
    
    """ 2) MarketCap over MEDIAN """
    mcp = pd.read_sql(f"select * from finance_info_copy where date='{dt}' and itm='시가총액'",conn)
    mcp_median = mcp.describe().loc['50%','value']
    mcp_ls = list(mcp[lambda x : x['value']>=mcp_median].code.values)
    
    profit = pd.read_sql(f"select * from finance_info_copy where date='{dt}' and type='Y' and itm='당기순이익'",conn)
    profit_ls = list(profit[lambda x : x['value']>0].code.values)
    pbr = pd.read_sql(f"select * from finance_info_copy where date='{dt}' and type='Y' and itm='PBR' and code in {tuple(profit_ls)}",conn)
    pbr_ls = list(pbr[lambda x : x['value']>0].code.values)
    pcr = pd.read_sql(f"select * from finance_info_copy where date='{dt}' and type='Y' and itm='PCR' and code in {tuple(pbr_ls)}",conn)
    pcr_ls = list(pcr[lambda x : x['value']>0].code.values)
    por = pd.read_sql(f"select * from finance_info_copy where date='{dt}' and type='Y' and itm='POR' and code in {tuple(pcr_ls)}",conn)
    por_ls = list(por[lambda x : x['value']>0].code.values)
    
    set1 = set(set(vol_ls).union(set(mcp_ls)))
    set2 = set(por_ls)
    total_ls = list(set1.intersection(set2))
    print("Our Universe : {}".format(len(total_ls)))
    return total_ls

def PairTrading_v11(pr, start, end, real_start, real_end, byFunda='all', up_cut=0.3, down_cut=-0.20, cutoff=0.05, fltr=True, enter=1, ndays=22):
    print("Initial number of companies : {}".format(len(pr.columns)))
    """ 0) Log PR """
    pr = pr.astype(float)
    real_pr = pr[(pr.index>=real_start)&(pr.index<=real_end)]
    real_pr.dropna(axis=1,how='any')
    log_real_pr = np.log(real_pr)
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
    
#     """ * Stop & Loss Cut"""
#     pr_rt = pr.pct_change()
#     out_ls = []
#     for cd in pr_rt.columns:
#         if len(pr_rt[lambda x : x[cd] < down_cut]) >= 1 : # 1번이라도 나오면 없애야 하나?
#             out_ls.append(cd)
#     log_pr.drop(out_ls, axis=1, inplace=True)
#     pr.drop(out_ls, axis=1, inplace=True)
#     print("Excluding stop-loss condition with {} : -> now : {}".format(down_cut, len(pr.columns)))
    
    
    """ 1) Correlation """
    cor_rst = Correlation_v3(log_pr, cutoff)
    print("Validation w.r.t Correlation : {}".format(len(cor_rst)))
    
    """ 2) Partial Correlation """
    #ksp = pd.read_hdf("./FullCache/KOSPI_close.h5")
    ksp = pd.read_hdf("./FullCache/KOSPI_lv2.h5")
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
    A_earning_ls = []
    A_trade_ls = []
    A_maxterm_ls = []
    B_earning_ls = []
    B_trade_ls = []
    B_maxterm_ls = []
    
    for idx in range(len(pairs)):
        a = pairs.loc[idx,'A']
        b = pairs.loc[idx,'B']
        coint_coeff = pairs.loc[idx,'cointeg']
        spread = log_pr[a] - log_pr[b] * coint_coeff
        risk_ls.append(spread.diff().std())
        try :
            earning, trade_num, max_term = ExpectedEarning_v2(a, b, coint_coeff, pr, enter=enter, position='A')
            A_earning_ls.append((earning-1)*100)
            A_trade_ls.append(trade_num)
            A_maxterm_ls.append(max_term)
        except :
            A_earning_ls.append(-999)
            A_trade_ls.append(-999)
            A_maxterm_ls.append(-999)
        try:
            earning, trade_num, max_term = ExpectedEarning_v2(a, b, coint_coeff, pr, enter=enter, position='B')
            B_earning_ls.append((earning-1)*100)
            B_trade_ls.append(trade_num)
            B_maxterm_ls.append(max_term)
        except:
            B_earning_ls.append(-999)
            B_trade_ls.append(-999)
            B_maxterm_ls.append(-999)

        
    pairs['risk'] = risk_ls
    pairs['A_earnings(%)'] = A_earning_ls
    pairs['A_trade#'] = A_trade_ls
    pairs['A_maxterm'] = A_maxterm_ls
    pairs['B_earnings(%)'] = B_earning_ls
    pairs['B_trade#'] = B_trade_ls
    pairs['B_maxterm'] = B_maxterm_ls

    print("Adding Risk & Earnings & #Trade & #UpCut")
    """ 11) Difference Correlation """
    pairs = DiffCorrelation(pairs, log_pr, cutoff=cutoff)
    print("Validation w.r.t Diff-Correlation : {}".format(len(pairs)))
    """ 12) Add AvgVol & AvgMCPRate """
    pairs = AddVolMcp(pairs).sort_values(by=['cointeg'],ascending=False).reset_index(drop=True)
    
    
    """ 13) Checking whole results on TEST Period """
    """ CoInteg + Stationary + Normality + Corr """
    survive_ls = []
    for idx, row in pairs.iterrows():
        a = row.A
        b = row.B
        eta = row.cointeg
        a_val = log_real_pr[a]
        b_val = log_real_pr[b]
        real_spread = a_val - b_val * eta
        if coint(a_val, b_val)[1] <= cutoff:
            if adfuller(real_spread)[1] <= cutoff:
                if stats.shapiro(real_spread)[1] >= cutoff :
                    survive_ls.append(idx)
                    #tmp = stats.pearsonr(a_val, b_val)
                    #if tmp[0]>=0 and tmp[1] <= cutoff:
                    #    survive_ls.append(idx)
    
#     """ 14) Best Funda Quantile Pattern Check """
#     FundaScore_A = []
#     FundaScore_B = []
#     if int(real_end[4:6])>=4:
#         dt = str(int(real_end[:4])-1)+'-12'
#         score_dict = FundaMatch(dt, byFunda=byFunda)
#         for idx, row in pairs.iterrows():
#             a = row.A
#             b = row.B
#             try :
#                 FundaScore_A.append(score_dict[a])
#             except :
#                 FundaScore_A.append(0)
#             try :
#                 FundaScore_B.append(score_dict[b])
#             except :
#                 FundaScore_B.append(0)
                
#     else :
#         dt = str(int(real_end[:4])-2)+'-12'
#         score_dict = FundaMatch(dt, byFunda=byFunda)
#         for idx, row in pairs.iterrows():
#             a = row.A
#             b = row.B
#             try :
#                 FundaScore_A.append(score_dict[a])
#             except :
#                 FundaScore_A.append(0)
#             try :
#                 FundaScore_B.append(score_dict[b])
#             except :
#                 FundaScore_B.append(0)
#             #FundaScore_A.append(score_dict[a])
#             #FundaScore_B.append(score_dict[b])
#     pairs['A_Funda(+)'] = FundaScore_A
#     pairs['B_Funda(+)'] = FundaScore_B
    
    """ 15) Spread Divergence Check """
    divcheck_idx = []
    for idx, row in pairs.iterrows():
        a = row.A
        b = row.B
        coint_coeff = row.cointeg
        if SpreadDiverge(a, b, coint_coeff, end, pr, real_pr, enter=2, ndays=ndays)[0] == False :
            divcheck_idx.append(idx)
    pairs = pairs[pairs.index.isin(divcheck_idx)]
    print("Validation w.r.t Spread Divergence : {} survived.".format(len(divcheck_idx)))
    
        
    return pairs, pairs[pairs.index.isin(survive_ls)]

def TradeVisual_v2(A, B, eta, start, end, enter=1):
    font_path = r'C:\Users\Bae Kyungmo\Downloads\Nanumsquare_ac_TTF\Nanumsquare_ac_TTF\NanumSquare_acR.ttf'
    fontprop = fm.FontProperties(fname=font_path, size=15)
    color=iter(cm.rainbow(np.linspace(0,1,3)))
    ldr = Loader()
    pr = ldr.GetPricelv2(start,end,[A,B])
    loglv22 = np.log(pr)
    qq_log = loglv22[(loglv22.index>=start)&(loglv22.index<=end)]
    qq_full = loglv22[(loglv22.index>=start)]
    aa = qq_log[A]
    bb = qq_log[B]
    aa_full = qq_full[A]
    bb_full = qq_full[B]
    etaa = eta
    spread = aa - bb * etaa
    spread_full = aa_full - bb_full * etaa
    fig = plt.figure(figsize=(12,10))
    ax1 = fig.add_subplot(2, 1, 1)
    plt.xticks(range(0,len(spread_full),30), [spread_full.index[idx] for idx in range(0,len(spread_full),30)])
    #plt.grid(True)
    try :
        A_name = ldr.FindNameByCode(A)
        B_name = ldr.FindNameByCode(B)
    except :
        if B == '005935':
            A_name = ldr.FindNameByCode(A)
            B_name = '삼성전자(우)'
        else:
            A_name = A
            B_name = B
    plt.title("PAIR TRADING : {} & {} with COINT-COEFF {}".format(A_name,B_name,etaa),fontproperties=fontprop)    
    ax2 = fig.add_subplot(2, 1, 2)
    ax1.plot(spread.reset_index().index, spread.values,color='black')
    ax1.plot(spread_full.reset_index().index[len(spread):], spread_full.values[len(spread):],color='black')
    ax1.hlines(spread.mean(),0,len(spread_full),colors='blue',linewidth=3,linestyles='dashed')
    ax1.hlines(spread.mean()+enter*spread.std(),0,len(spread_full),colors='skyblue',linewidth=2,linestyles='dashed')
    ax1.hlines(spread.mean()-enter*spread.std(),0,len(spread_full),colors='skyblue',linewidth=2,linestyles='dashed')
    ax1.hlines(spread.mean()+1*spread.std(),0,len(spread_full),colors='gray',linewidth=1,linestyles='dashed')
    ax1.hlines(spread.mean()-1*spread.std(),0,len(spread_full),colors='gray',linewidth=1,linestyles='dashed')
    ax1.vlines(len(spread),min(spread_full),max(spread_full),colors='purple',linewidth=2,linestyles='solid')
    plt.xticks(range(0,len(spread_full),30), [spread_full.index[idx] for idx in range(0,len(spread_full),30)])
    ax2.plot(spread_full.reset_index().index, pr[A],label=A)
    ax2.plot(spread_full.reset_index().index, pr[B],label=B)
    #ax2.plot(spread_full.reset_index().index, pr[A].rolling(5,1).mean(),label='{}_MA5'.format(A),color='skyblue')
    #ax2.plot(spread_full.reset_index().index, pr[B].rolling(5,1).mean(),label='{}_MA5'.format(B),color='gray')
    ax2.vlines(len(spread),-4,4,colors='purple',linewidth=2)
    ax2.legend(loc='upper left')
    #plt.grid(True)
    plt.show()
    return True

def VolumeCheck(pairs, end):
    tdays = ldr.GetTradingDays(end=end)
    end_idx = tdays.index(end)
    start = tdays[end_idx-30]
    pr_lv1 = ldr.GetPricelv1(start,end)
    median_df = pr_lv1.groupby(['CODE'])['volume'].mean()
    median_vol = median_df.median()
    over_median = list(median_df[lambda x : x >= median_vol].index)
    codeTOmedian = median_df.to_dict()
    
    meanVol=[]
    overVol=[]
    for idx, row in pairs.iterrows():
        A=row.A
        B=row.B
        if A in over_median and B in over_median :
            overVol.append('Y')
        else :
            overVol.append('N')
        meanVol.append((codeTOmedian[A]+codeTOmedian[B])/2)
    pairs['overVol'] = overVol
    pairs['meanVol'] = meanVol
    return pairs

def PairTrading_v12(start, end, code_ls, pr_df, cutoff=0.05, enter=1, pr_diff=20000):
    ldr = Loader()
    #pr = ldr.GetPricelv2(start, end, code_ls)
    pr = pr_df.copy()
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
    
    """ 1) Correlation """
    cor_rst = Correlation_v3(log_pr, cutoff)
    print("Validation w.r.t Correlation : {}".format(len(cor_rst)))
    
    """ 2) Partial Correlation """
    ksp = pd.read_hdf("./FullCache/KOSPI_lv2.h5")
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
        
        for eta in [0.1*i for i in range(5,21)]:
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
    A_earning_ls = []
    A_trade_ls = []
    A_maxterm_ls = []
    B_earning_ls = []
    B_trade_ls = []
    B_maxterm_ls = []
    
    for idx in range(len(pairs)):
        a = pairs.loc[idx,'A']
        b = pairs.loc[idx,'B']
        coint_coeff = pairs.loc[idx,'cointeg']
        spread = log_pr[a] - log_pr[b] * coint_coeff
        risk_ls.append(spread.diff().std())
        try :
            earning, trade_num, max_term = ExpectedEarning_v2(a, b, coint_coeff, pr, enter=enter, position='A')
            A_earning_ls.append((earning-1)*100)
            A_trade_ls.append(trade_num)
            A_maxterm_ls.append(max_term)
        except :
            A_earning_ls.append(-999)
            A_trade_ls.append(-999)
            A_maxterm_ls.append(-999)
        try:
            earning, trade_num, max_term = ExpectedEarning_v2(a, b, coint_coeff, pr, enter=enter, position='B')
            B_earning_ls.append((earning-1)*100)
            B_trade_ls.append(trade_num)
            B_maxterm_ls.append(max_term)
        except:
            B_earning_ls.append(-999)
            B_trade_ls.append(-999)
            B_maxterm_ls.append(-999)

        
    pairs['risk'] = risk_ls
    pairs['A_earnings(%)'] = A_earning_ls
    pairs['A_trade#'] = A_trade_ls
    pairs['A_maxterm'] = A_maxterm_ls
    pairs['B_earnings(%)'] = B_earning_ls
    pairs['B_trade#'] = B_trade_ls
    pairs['B_maxterm'] = B_maxterm_ls

    print("Adding Risk & Earnings & #Trade & #UpCut")
    """ 11) Difference Correlation """
    pairs = DiffCorrelation(pairs, log_pr, cutoff=cutoff)
    print("Validation w.r.t Diff-Correlation : {}".format(len(pairs)))
    """ 12) Add AvgVol & AvgMCPRate """
    #pairs = AddVolMcp(pairs).sort_values(by=['cointeg'],ascending=False).reset_index(drop=True)
    A_price=[]
    B_price=[]
    pr_valid=[]
    for idx, row in pairs.iterrows():
        A = row.A
        B = row.B
        A_price.append(pr.loc[pr.index[-1],A])
        B_price.append(pr.loc[pr.index[-1],B])
        if abs(pr.loc[pr.index[-5:],A].mean() - row.cointeg * pr.loc[pr.index[-5:],B].mean()) <= pr_diff:
            pr_valid.append('Y')
        else :
            pr_valid.append('N')
        
    pairs['A_price'] = A_price
    pairs['B_price'] = B_price
    pairs['PR_VALID'] = pr_valid
    #pairs = pairs[pairs.index.isin(pr_valid)].reset_index(drop=True)
        
    return pairs