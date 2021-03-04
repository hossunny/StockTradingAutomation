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
                                   password='*',db='INVESTAR',charset='utf8')

def Correlation_v3(log_pr, cutoff=0.05):
    rst = pd.DataFrame(columns=['corr','A','B'])
    codes = list(log_pr.columns)
    for i in range(len(codes)):
        for j in range(len(codes)):
            if i<j:
                corr, pvalue = stats.pearsonr(log_pr[codes[i]], log_pr[codes[j]])
                if pvalue <= cutoff and corr >= 0:
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


def PairTrading_v12(start, end, code_ls, cutoff=0.05, enter=1, pr_diff=20000):
    ldr = Loader()
    pr = ldr.GetPricelv2(start, end, code_ls)
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