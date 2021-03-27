import logging, os, pickle
import requests, glob
from datetime import datetime, date
import pandas as pd
import time
import json, re, sys, h5py
import datetime as dt
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings(action='ignore')
import shutil
from matplotlib.pyplot import cm
import numpy as np
import scipy.stats as stats
from scipy import stats
import seaborn as sns
sys.path.append('C:\\Users\Bae Kyungmo\OneDrive\Desktop\StockTraidingAutomation\DataModeling')
from Loader_v2 import Loader

class TechnicalIndicator:
    def __init__(self, pwd):
        self.ldr = Loader(pwd=pwd)

    def MDD(self, pr):
        code = pr.columns[0]
        x = pr[code]
        """
        MDD(Maximum Draw-Down)
        :return: (peak_upper, peak_lower, mdd rate)
        """
        arr_v = np.array(x)
        peak_lower = np.argmax(np.maximum.accumulate(arr_v) - arr_v)
        peak_upper = np.argmax(arr_v[:peak_lower])
        return round((arr_v[peak_lower] - arr_v[peak_upper]) / arr_v[peak_upper],2)     
    
    def SharpeRatio(self, rt_df, bncmrk=0.02):
        SR = np.sqrt(len(rt_df))*(rt_df['return'].mean() - bncmrk) / rt_df['return'].std()
        return SR

    def SMA(self, pr, ndays=22, min_days=1):
        code = pr.columns[0]
        return pr[[code]].rolling(window=ndays, min_periods=min_days).mean()
    
    def EMA(self, pr, ndays=22, min_days=1):
        code = pr.columns[0]
        return pr[[code]].ewm(ndays,min_periods=min_days).mean()

    def MACD(self, pr_df, long_d=26, short_d=12, signal_d=9, min_days=1):
        pr = pr_df.copy()
        shortEMA = self.EMA(pr, short_d, min_days=min_days)
        longEMA = self.EMA(pr, long_d, min_days=min_days)
        pr['MACD'] = shortEMA - longEMA
        pr['Signal'] = self.EMA(pr[['MACD']], signal_d)
        return pr

    def MACD_Signal(self, dff):
        df = dff.copy()
        df.columns = ['close','MACD','Signal']
        Buy=[]
        Sell=[]
        Buy.append(np.nan)
        Sell.append(np.nan)
        flag = -1
        for i in range(1, len(df)):
            if df['MACD'][i] > df['Signal'][i] and flag != 1:
                Buy.append(df['close'][i])
                Sell.append(np.nan)
                flag = 1
            elif df['MACD'][i] < df['Signal'][i] and flag != 0:
                Sell.append(df['close'][i])
                Buy.append(np.nan)
                flag = 0
            else :
                Buy.append(np.nan)
                Sell.append(np.nan)
        df['BuySignal'] = Buy
        df['SellSignal'] = Sell
        return df
            
    def BackTest(self, obv):
        tmp = obv[(~obv['BuySignal'].isnull())|(~obv['SellSignal'].isnull())]
        tmp['BuySignal'] = tmp['BuySignal'].fillna(0)
        tmp['SellSignal'] = tmp['SellSignal'].fillna(0)
        tmp['series'] = tmp['BuySignal'] + tmp['SellSignal']
        tmp['return'] = tmp['series'].pct_change()+1
        tmp2 = tmp[tmp.SellSignal!=0.0]
        cm_rt = tmp2['return'].cumprod()[-1]
        hit_ratio = round(len(tmp2[tmp2['return']>=1]) / len(tmp2),2)
        best_rt = tmp2['return'].max()-1
        worst_rt = tmp2['return'].min()-1
        ar_mean = tmp2['return'].mean()
        gr_mean = np.exp(np.log(cm_rt)/len(tmp2))
        mdd = self.MDD(obv[['close']])
        SR = self.SharpeRatio(tmp2[['return']]-1)
        test_rst = pd.DataFrame(columns=['CumulativeReturn','HitRatio','BestReturn','WorstReturn','MDD','SimpleMean','GeoMean','#Trade'])
        test_rst.loc[0,'CumulativeReturn'] = round(cm_rt,2)
        test_rst.loc[0,'HitRatio'] = hit_ratio
        test_rst.loc[0,'BestReturn'] = round(best_rt,2)
        test_rst.loc[0,'WorstReturn'] = round(worst_rt,2)
        test_rst.loc[0,'SimpleMean'] = round(ar_mean,2)
        test_rst.loc[0,'GeoMean'] = round(gr_mean,2)
        test_rst.loc[0,'MDD'] = round(mdd*(-1),2)
        test_rst.loc[0,'SharpeRatio'] = round(SR,2)
        test_rst.loc[0,'#Trade'] = round(len(tmp2),2)
        return tmp2, test_rst

    def SMA_Label(self, pr_df, long_d=26, short_d=12, min_days=1):
        pr = pr_df.copy()
        shortSMA = self.SMA(pr, short_d, min_days=min_days)
        longSMA = self.SMA(pr, long_d, min_days=min_days)
        pr['SMA_short'] = shortSMA
        pr['SMA_long'] = longSMA
        return pr

    def SMA_Signal(self, dff):
        df = dff.copy()
        df.columns = ['close','SMA_short','SMA_long']
        Buy=[]
        Sell=[]
        Buy.append(np.nan)
        Sell.append(np.nan)
        flag = -1
        for i in range(1, len(df)):
            if df['SMA_short'][i] > df['SMA_long'][i] and flag != 1:
                Buy.append(df['close'][i])
                Sell.append(np.nan)
                flag = 1
            elif df['SMA_short'][i] < df['SMA_long'][i] and flag != 0:
                Sell.append(df['close'][i])
                Buy.append(np.nan)
                flag = 0
            else :
                Buy.append(np.nan)
                Sell.append(np.nan)
        df['BuySignal'] = Buy
        df['SellSignal'] = Sell
        return df

    def MFI(self, pr_df, ndays=14):
        pr = pr_df.copy()
        pr['TypicalPrice'] = round((pr['high']+pr['low']+pr['close'])/3,3)
        pr['MF'] = pr['TypicalPrice'] * pr['volume']
        
        P_flow = []
        N_flow = []
        P_flow.append(0)
        N_flow.append(0)
        
        for i in range(1, len(pr)):
            if pr['TypicalPrice'][i] > pr['TypicalPrice'][i-1]:
                P_flow.append(pr['MF'][i])
                N_flow.append(0)
            elif pr['TypicalPrice'][i] < pr['TypicalPrice'][i-1]:
                N_flow.append(pr['MF'][i])
                P_flow.append(0)
            else :
                P_flow.append(0)
                N_flow.append(0)
        pr['P_MF'] = P_flow
        pr['N_MF'] = N_flow
        pr['P_MF'] = pr['P_MF'].rolling(ndays,min_periods=1).sum()
        pr['N_MF'] = pr['N_MF'].rolling(ndays,min_periods=1).sum()
        pr['MFI'] = 100 * (pr['P_MF'] / (pr['P_MF'] + pr['N_MF']))
        return pr

    def MFI_Signal(self, dff, buyMFI=20, sellMFI=80):
        df = dff.copy()
        Buy = []
        Sell = []
        flag = -1
        for i in range(len(df)):
            if df['MFI'][i] <= buyMFI and flag != 1:
                Buy.append(df['adjprice'][i])
                Sell.append(np.nan)
                flag = 1
            elif df['MFI'][i] >= sellMFI and flag != 0:
                Sell.append(df['adjprice'][i])
                Buy.append(np.nan)
                flag = 0
            else :
                Buy.append(np.nan)
                Sell.append(np.nan)
        df['BuySignal'] = Buy
        df['SellSignal'] = Sell
        """
        For the first position being long, if SellSignal is prior to BuySignal
        remove that position..
        """
        firstSell = df[df['SellSignal'].notnull()].index[0]
        firstBuy = df[df['BuySignal'].notnull()].index[0]
        if firstBuy > firstSell : 
            df.loc[firstSell,'SellSignal'] = np.nan
        return df

    def MFI_Signal_All(self, dff, buyMFI=20, sellMFI=80):
        df = dff.copy()
        Buy = []
        Sell = []
        for i in range(len(df)):
            if df['MFI'][i] <= buyMFI:
                Buy.append(df['adjprice'][i])
                Sell.append(np.nan)
            elif df['MFI'][i] >= sellMFI:
                Sell.append(df['adjprice'][i])
                Buy.append(np.nan)
            else :
                Buy.append(np.nan)
                Sell.append(np.nan)
        df['BuySignal'] = Buy
        df['SellSignal'] = Sell
        return df

    def OBV(self, pr, vol, ndays=20):
        OBV = []
        OBV.append(0)
        code = pr.columns[0]
        for i in range(1, len(pr[code])):
            if pr[code][i] > pr[code][i-1]:
                OBV.append(OBV[-1] + vol[code][i])
            elif pr[code][i] < pr[code][i-1]:
                OBV.append(OBV[-1] - vol[code][i])
            else :
                OBV.append(OBV[-1])
        pr.columns = ['close']
        vol.columns = ['volume']
        total = pd.concat([pr,vol],axis=1)
        assert len(pr) == len(vol) == len(OBV)
        total['OBV'] = OBV
        total['OBV_EMA'] = total['OBV'].ewm(com=ndays).mean()
        return total

    def OBV_Signal(self, obv):
        sigPriceBuy = []
        sigPriceSell = []
        flag = -1 
        for i in range(0,len(obv)):
            if obv['OBV'][i] > obv['OBV_EMA'][i] and flag != 1:
                sigPriceBuy.append(obv['close'][i])
                sigPriceSell.append(np.nan)
                flag = 1
            elif obv['OBV'][i] < obv['OBV_EMA'][i] and flag != 0:    
                sigPriceSell.append(obv['close'][i])
                sigPriceBuy.append(np.nan)
                flag = 0
            else: 
                sigPriceBuy.append(np.nan)
                sigPriceSell.append(np.nan)
        obv['BuySignal'] = sigPriceBuy
        obv['SellSignal'] = sigPriceSell

        firstSell = obv[obv['SellSignal'].notnull()].index[0]
        firstBuy = obv[obv['BuySignal'].notnull()].index[0]
        if firstBuy > firstSell : 
            obv.loc[firstSell,'SellSignal'] = np.nan
        return obv

    def RSI(self, pr_df, ndays=14):
        pr = pr_df.copy()
        code = pr.columns[0]
        delta = pr[code].diff(1)
        delta.fillna(0,inplace=True)
        
        Up = delta.copy()
        Down = delta.copy()
        Up[Up<0] = 0
        Down[Down>0] = 0
        pr['Up'] = Up
        pr['Down'] = Down
        avgGain = self.SMA(pr[['Up']], ndays)['Up']
        avgLoss = abs(self.SMA(pr[['Down']], ndays))['Down']
        RS = avgGain / avgLoss
        RSI = 100.0 - (100.0 / (1.0 + RS))
        pr['RSI'] = RSI
        return pr

    def RSI_Signal(self, dff, buyRSI=30, sellRSI=70):
        df = dff.copy()
        df.columns = ['close','RSI']
        Buy=[]
        Sell=[]
        Buy.append(np.nan)
        Sell.append(np.nan)
        flag = -1
        for i in range(1, len(df)):
            if df['RSI'][i] <= buyRSI and flag != 1:
                Buy.append(df['close'][i])
                Sell.append(np.nan)
                flag = 1
            elif df['RSI'][i] >= sellRSI and flag != 0:
                Sell.append(df['close'][i])
                Buy.append(np.nan)
                flag = 0
            else :
                Buy.append(np.nan)
                Sell.append(np.nan)
        df['BuySignal'] = Buy
        df['SellSignal'] = Sell
        
        """
        For the first position being long, if SellSignal is prior to BuySignal
        remove that position..
        """
        firstSell = df[df['SellSignal'].notnull()].index[0]
        firstBuy = df[df['BuySignal'].notnull()].index[0]
        if firstBuy > firstSell : 
            df.loc[firstSell,'SellSignal'] = np.nan
        return df

    def RSI_Run(self, start, end, code, ndays=14, buyRSI=30, sellRSI=70, doplot=True):
        pr = self.ldr.GetPricelv2(start, end, [code])
        rsi = self.RSI(pr, ndays=ndays)
        if doplot :
            plt.style.use('fivethirtyeight')
            rsi[['RSI']].plot(figsize=(8,6))
            plt.title("RSI indicator for {}".format(code))
        rsi_sig = self.RSI_Signal(rsi[[code,'RSI']], buyRSI=buyRSI, sellRSI=sellRSI)
        if doplot :
            plt.figure(figsize=(8,6))
            plt.scatter(rsi_sig.index, rsi_sig['BuySignal'], color='green',label='BuySignal',marker='^',alpha=1)
            plt.scatter(rsi_sig.index, rsi_sig['SellSignal'], color='red',label='SellSignal',marker='v',alpha=1)
            plt.plot(rsi_sig['close'], label='Close Price', alpha=0.35)
            plt.xticks([],rotation=45)
            plt.title('RSI on {} during {} ~ {}'.format(code,start,end))
            plt.xlabel('Date',fontsize=15)
            plt.ylabel('Close Price KRW',fontsize=15)
            plt.legend(loc='upper left')
            plt.show()
        _, rst = self.BackTest(rsi_sig)
        rst.index = [code]
        return rst

    def StochasticMaker(self, pr_df, n=14, m=5, t=5):
        pr = pr_df.copy()
        pr = pr.rename(columns={'adjprice':'close'})
        ndays_high = pr['high'].rolling(window=n, min_periods=1).max()
        ndays_low = pr['low'].rolling(window=n, min_periods=1).min()
        fast_k = ((pr['close'] - ndays_low) / (ndays_high - ndays_low)) * 100
        slow_k = fast_k.ewm(span=m).mean()
        slow_d = slow_k.ewm(span=t).mean()
        pr = pr.assign(fast_k=fast_k, fast_d=slow_k, slow_k=slow_k, slow_d=slow_d)
        return pr

    def SO_Signal_v1(self, dff, buyK=20, sellK=80, buyRSI=30, sellRSI=70, velocity='fast'):
        df = dff.copy()
        if velocity =='fast':
            df = df[['close','fast_k','RSI']]
            df = df.rename(columns={'fast_k':'k'})
        elif velocity =='slow':
            df = df[['close','slow_k','RSI']]
            df = df.rename(columns={'slow_k':'k'})
        Buy=[]
        Sell=[]
        Buy.append(np.nan)
        Sell.append(np.nan)
        flag = -1
        for i in range(1, len(df)):
            if df['k'][i]<=buyK and df['RSI'][i] <= buyRSI and flag != 1:
                Buy.append(df['close'][i])
                Sell.append(np.nan)
                flag = 1
            elif df['k'][i]>=sellK  and df['RSI'][i] >= sellRSI and flag != 0:
                Sell.append(df['close'][i])
                Buy.append(np.nan)
                flag = 0
            else :
                Buy.append(np.nan)
                Sell.append(np.nan)
        df['BuySignal'] = Buy
        df['SellSignal'] = Sell
        
        """
        For the first position being long, if SellSignal is prior to BuySignal
        remove that position..
        """
        firstSell = df[df['SellSignal'].notnull()].index[0]
        firstBuy = df[df['BuySignal'].notnull()].index[0]
        if firstBuy > firstSell : 
            df.loc[firstSell,'SellSignal'] = np.nan
        return df

    def SO_Signal_v2(self, dff, buyK=20, sellK=80, buyRSI=30, sellRSI=70, velocity='fast'):
        df = dff.copy()
        if velocity =='fast':
            df = df[['close','fast_k','fast_d','RSI']]
            df = df.rename(columns={'fast_k':'k','fast_d':'d'})
        elif velocity =='slow':
            df = df[['close','slow_k','slow_d','RSI']]
            df = df.rename(columns={'slow_k':'k','slow_d':'d'})
        Buy=[]
        Sell=[]
        Buy.append(np.nan)
        Sell.append(np.nan)
        flag = -1
        for i in range(1, len(df)):
            if df['k'][i] > df['d'][i] and df['k'][i]<=buyK and flag != 1:
                Buy.append(df['close'][i])
                Sell.append(np.nan)
                flag = 1
            elif df['k'][i] < df['d'][i] and df['k'][i]>=sellK and flag != 0:
                Sell.append(df['close'][i])
                Buy.append(np.nan)
                flag = 0
            else :
                Buy.append(np.nan)
                Sell.append(np.nan)
        df['BuySignal'] = Buy
        df['SellSignal'] = Sell
        
        """
        For the first position being long, if SellSignal is prior to BuySignal
        remove that position..
        """
        firstSell = df[df['SellSignal'].notnull()].index[0]
        firstBuy = df[df['BuySignal'].notnull()].index[0]
        if firstBuy > firstSell : 
            df.loc[firstSell,'SellSignal'] = np.nan
        return df

    def TSTS(self, pr_df, long_d=130, short_d=60, signal_d=45, n=14, m=3, min_days=1):
        pr = pr_df.copy()
        shortEMA = self.EMA(pr[['close']], short_d, min_days=min_days)
        longEMA = self.EMA(pr[['close']], long_d, min_days=min_days)
        pr['MACD'] = shortEMA - longEMA
        pr['Signal'] = self.EMA(pr[['MACD']], signal_d, min_days=min_days)
        pr['LongEMA'] = longEMA
        pr['ShortEMA'] = shortEMA
        pr['MACD_HIST'] = pr['MACD'] - pr['Signal']
        
        ndays_high = pr['high'].rolling(window=n, min_periods=1).max()
        ndays_low = pr['low'].rolling(window=n, min_periods=1).min()
        fast_k = ((pr['close'] - ndays_low) / (ndays_high - ndays_low)) * 100
        slow_d = fast_k.rolling(m, min_periods=1).mean()
        pr['fast_k'] = fast_k
        pr['slow_d'] = slow_d
        
        return pr

    def TSTS_Signal(self, dff, buy=20, sell=80):
        df = dff.copy()
        Buy=[]
        Sell=[]
        Buy.append(np.nan)
        Sell.append(np.nan)
        flag = -1
        for i in range(1, len(df)):
            if df['LongEMA'][i-1] < df['LongEMA'][i] and df['slow_d'][i-1]>=buy and df['slow_d'][i]<buy and flag != 1:
                Buy.append(df['close'][i])
                Sell.append(np.nan)
                flag = 1
            elif df['LongEMA'][i-1] > df['LongEMA'][i] and df['slow_d'][i-1]<=sell and df['slow_d'][i]>sell and flag != 0:
                Sell.append(df['close'][i])
                Buy.append(np.nan)
                flag = 0
            else :
                Buy.append(np.nan)
                Sell.append(np.nan)
        df['BuySignal'] = Buy
        df['SellSignal'] = Sell
        """
        For the first position being long, if SellSignal is prior to BuySignal
        remove that position..
        """
        firstSell = df[df['SellSignal'].notnull()].index[0]
        firstBuy = df[df['BuySignal'].notnull()].index[0]
        if firstBuy > firstSell : 
            df.loc[firstSell,'SellSignal'] = np.nan
        return df

    def BB(self, pr_df, ndays=20, sigma_lvl=2):
        pr = pr_df.copy()
        pr['UpperBand'] = pr['close'].rolling(ndays, min_periods=1).mean()+pr['close'].rolling(ndays, min_periods=1).std()*sigma_lvl
        pr['LowerBand'] = pr['close'].rolling(ndays, min_periods=1).mean()-pr['close'].rolling(ndays, min_periods=1).std()*sigma_lvl
        pr['PB'] = (pr['close'] - pr['LowerBand']) / (pr['UpperBand'] - pr['LowerBand'])
        pr['BandWidth'] = (pr['UpperBand'] - pr['LowerBand']) / pr['close'].rolling(ndays, min_periods=1).mean() * 100
        return pr.fillna(method='bfill')

    def BB_Statistics(self, bb, colname):
        #code = bb.columns[0]
        col = colname
        num = len(bb)
        in_nm = 0
        up_nm = 0
        down_nm = 0
        for i in range(len(bb)):
            if bb[col][i] > bb['UpperBand'][i] :
                up_nm += 1
            elif bb[col][i] < bb['LowerBand'][i] : 
                down_nm += 1
            elif bb['LowerBand'][i] <= bb[col][i] <= bb['UpperBand'][i] : 
                in_nm += 1
        rst = pd.DataFrame(columns = ['Total','Inside','Upper','Lower'], index=[col])
        rst.loc[col,'Total'] = num
        rst.loc[col,'Inside'] = round(in_nm/num,2)
        rst.loc[col,'Upper'] = round(up_nm/num,2)
        rst.loc[col,'Lower'] = round(down_nm/num,2)
        return rst

    def BB_Signal_v1(self, dff):
        """
        Simply when close price does crossover the upper band then buy, else sell.
        """
        df = dff.copy()
        Buy=[]
        Sell=[]
        Buy.append(np.nan)
        Sell.append(np.nan)
        flag = -1
        for i in range(1, len(df)):
            if df['close'][i] < df['LowerBand'][i] and flag != 1:
                Buy.append(df['close'][i])
                Sell.append(np.nan)
                flag = 1
            elif df['close'][i] > df['UpperBand'][i] and flag != 0:
                Sell.append(df['close'][i])
                Buy.append(np.nan)
                flag = 0
            else :
                Buy.append(np.nan)
                Sell.append(np.nan)
        df['BuySignal'] = Buy
        df['SellSignal'] = Sell
        
        """
        For the first position being long, if SellSignal is prior to BuySignal
        remove that position..
        """
        firstSell = df[df['SellSignal'].notnull()].index[0]
        firstBuy = df[df['BuySignal'].notnull()].index[0]
        if firstBuy > firstSell : 
            df.loc[firstSell,'SellSignal'] = np.nan
        return df

    def BB_Run_v1(self, start, end, code,  ndays=14, sigma_lvl=2, doplot=True):
        print("==================== Description =====================")
        print("Buy when close price does crossover the upper band, else sell.")
        print("Its name is not the one usually knowns as Rarry William's Volatility Breakout Strategy.")
        
        pr = self.ldr.GetPricelv2(start, end, [code])
        pr = pr.rename(columns={code:'close'})
        bb = self.BB(pr, ndays=ndays, sigma_lvl=sigma_lvl)
        if doplot :
            plt.style.use('fivethirtyeight')
            bb.plot(figsize=(8,6))
            plt.title("Bollinger Bands for {}".format(code))
        bb_sig = self.BB_Signal_v1(bb)
        print(self.BB_Statistics(bb, colname='close'))
        if doplot :
            plt.figure(figsize=(8,6))
            plt.scatter(bb_sig.index, bb_sig['BuySignal'], color='green',label='BuySignal',marker='^',alpha=1)
            plt.scatter(bb_sig.index, bb_sig['SellSignal'], color='red',label='SellSignal',marker='v',alpha=1)
            plt.plot(bb_sig['close'], label='Close Price', alpha=0.35)
            plt.xticks([],rotation=45)
            plt.title('Bollinger Bands on {} during {} ~ {}'.format(code,start,end))
            plt.xlabel('Date',fontsize=15)
            plt.ylabel('Close Price KRW',fontsize=15)
            plt.legend(loc='upper left')
            plt.show()
        _, rst = self.BackTest(bb_sig)
        rst.index = [code]
        return rst

    def BB_Signal_v2(self, dff, buy=20, sell=80):
        """
        With MFI if MFI>80 and PB>0.8 buy, else MFI<20 and PB<0.2 sell.
        """
        df = dff.copy()
        Buy=[]
        Sell=[]
        Buy.append(np.nan)
        Sell.append(np.nan)
        flag = -1
        for i in range(1, len(df)):
            if df['PB'][i] < buy/100 and df['MFI'][i] < buy and flag != 1:
                Buy.append(df['close'][i])
                Sell.append(np.nan)
                flag = 1
            elif df['PB'][i] > sell/100 and df['MFI'][i] > sell and flag != 0:
                Sell.append(df['close'][i])
                Buy.append(np.nan)
                flag = 0
            else :
                Buy.append(np.nan)
                Sell.append(np.nan)
        df['BuySignal'] = Buy
        df['SellSignal'] = Sell
        
        """
        For the first position being long, if SellSignal is prior to BuySignal
        remove that position..
        """
        firstSell = df[df['SellSignal'].notnull()].index[0]
        firstBuy = df[df['BuySignal'].notnull()].index[0]
        if firstBuy > firstSell : 
            df.loc[firstSell,'SellSignal'] = np.nan
        return df

    def BB_Run_v2(self, start, end, code,  ndays=14, sigma_lvl=2, buy=80, sell=20, doplot=True):
        print("==================== Description =====================")
        print("When PB<20 and MFI<20 then sell, else PB>80 and MFI>80 then buy.")
        print("Using MFI for trend momentum supports Bollinger Bands not to be a false signal.")

        pr = self.ldr.GetPricelv1(start, end, [code]).sort_values(by=['DATE'])
        pr = pr.rename(columns={'adjprice':'close','OPEN':'open'})
        pr.index = pr['DATE'].to_list()
        pr = self.MFI(pr, ndays=ndays)
        bb = self.BB(pr, ndays=ndays, sigma_lvl=sigma_lvl)
        if doplot :
            plt.style.use('fivethirtyeight')
            tmp = bb[['PB','MFI']]
            tmp['PB'] = tmp['PB'] * 100
            tmp.plot(figsize=(10,6))
            plt.title("Bollinger Bands for {}".format(code))
        bb_sig = self.BB_Signal_v2(bb, buy=buy, sell=sell)
        print(self.BB_Statistics(bb, colname='close'))
        if doplot :
            plt.figure(figsize=(8,6))
            plt.scatter(bb_sig.index, bb_sig['BuySignal'], color='green',label='BuySignal',marker='^',alpha=1)
            plt.scatter(bb_sig.index, bb_sig['SellSignal'], color='red',label='SellSignal',marker='v',alpha=1)
            plt.plot(bb_sig['close'], label='Close Price', alpha=0.35)
            plt.xticks([],rotation=45)
            plt.title('Bollinger Bands on {} during {} ~ {}'.format(code,start,end))
            plt.xlabel('Date',fontsize=15)
            plt.ylabel('Close Price KRW',fontsize=15)
            plt.legend(loc='upper left')
            plt.show()
        _, rst = self.BackTest(bb_sig)
        rst.index = [code]
        return rst

    def IntradayIntensity(self, pr_df, ndays=21):
        pr = pr_df.copy()
        pr['II'] = (2 * pr['close'] - pr['high'] - pr['low']) / (pr['high'] - pr['low']) * pr['volume']
        pr['PII'] = pr['II'].rolling(ndays, min_periods=1).sum() / pr['volume'].rolling(ndays, min_periods=1).sum() * 100
        """
        the first PII value is nothing at all so just copy the next value..
        """
        pr.loc[pr.index[0],'PII'] = pr.loc[pr.index[1],'PII']
        
        return pr

    def BB_Signal_v3(self, dff, buy=0.05, sell=0.95):
        """
        With II if II>0 and PB<0.05 buy, else II<0 and PB>0.95 sell.
        """
        df = dff.copy()
        Buy=[]
        Sell=[]
        Buy.append(np.nan)
        Sell.append(np.nan)
        flag = -1
        for i in range(1, len(df)):
            if df['PB'][i] < buy and df['II'][i] > 0 and flag != 1:
                Buy.append(df['close'][i])
                Sell.append(np.nan)
                flag = 1
            elif df['PB'][i] > sell and df['II'][i] < 0 and flag != 0:
                Sell.append(df['close'][i])
                Buy.append(np.nan)
                flag = 0
            else :
                Buy.append(np.nan)
                Sell.append(np.nan)
        df['BuySignal'] = Buy
        df['SellSignal'] = Sell
        
        """
        For the first position being long, if SellSignal is prior to BuySignal
        remove that position..
        """
        firstSell = df[df['SellSignal'].notnull()].index[0]
        firstBuy = df[df['BuySignal'].notnull()].index[0]
        if firstBuy > firstSell : 
            df.loc[firstSell,'SellSignal'] = np.nan
        return df

    def BB_Run_v3(self, start, end, code,  ndays=14, sigma_lvl=2, buy=0.05, sell=0.95, doplot=True):
        print("==================== Description =====================")
        print("With II if II>0 and PB<0.05 buy, else II<0 and PB>0.95 sell.")

        pr = self.ldr.GetPricelv1(start, end, [code]).sort_values(by=['DATE'])
        pr = pr.rename(columns={'adjprice':'close','OPEN':'open'})
        pr.index = pr['DATE'].to_list()
        pr = self.IntradayIntensity(pr, ndays=ndays)
        bb = self.BB(pr, ndays=ndays, sigma_lvl=sigma_lvl)
        if doplot :
            plt.style.use('fivethirtyeight')
            plt.figure(figsize=(8,6))
            plt.bar(bb.reset_index().index, bb['II'], color='g',label='IntradyIntensity(%)Ndays')
            plt.legend(loc='upper left')
            plt.title("Bollinger Bands for {}".format(code))
        bb_sig = self.BB_Signal_v3(bb, buy=buy, sell=sell)
        print(self.BB_Statistics(bb, colname='close'))
        if doplot :
            plt.figure(figsize=(8,6))
            plt.scatter(bb_sig.index, bb_sig['BuySignal'], color='green',label='BuySignal',marker='^',alpha=1)
            plt.scatter(bb_sig.index, bb_sig['SellSignal'], color='red',label='SellSignal',marker='v',alpha=1)
            plt.plot(bb_sig['close'], label='Close Price', alpha=0.35)
            plt.xticks([],rotation=45)
            plt.title('Bollinger Bands on {} during {} ~ {}'.format(code,start,end))
            plt.xlabel('Date',fontsize=15)
            plt.ylabel('Close Price KRW',fontsize=15)
            plt.legend(loc='upper left')
            plt.show()
        _, rst = self.BackTest(bb_sig)
        rst.index = [code]
        return rst

    def SqueezeBreakout(self, bb_df, sigma_lvl=1):
        bb = bb_df.copy()
        bb_rt = bb[['BandWidth']].pct_change()
        std = bb_rt['BandWidth'].std()
        mean = bb_rt['BandWidth'].mean()
        breakout = bb_rt[(bb_rt.BandWidth >= mean + std * sigma_lvl)].index.to_list()
        if len(breakout) == 0 :
            return False
        else :
            breakout_series = []
            for idx, row in bb.iterrows():
                if idx in breakout :
                    breakout_series.append(row.BandWidth)
                else :
                    breakout_series.append(np.nan)
            bb['BreakOut'] = breakout_series
            return bb

    def BB_BW_Squeeze(self, start, end, code, ndays=14, sigma_lvl=2, buy=0.05, sell=0.95, doplot=True):
        pr = self.ldr.GetPricelv1(start, end, [code]).sort_values(by=['DATE'])
        pr = pr.rename(columns={'adjprice':'close','OPEN':'open'})
        pr.index = pr['DATE'].to_list()
        pr = self.IntradayIntensity(pr, ndays=ndays)
        bb = self.BB(pr, ndays=ndays, sigma_lvl=sigma_lvl)
        bb['MA'] = bb['close'].rolling(ndays,min_periods=1).mean()
        
        plt.style.use('fivethirtyeight')
        plt.figure(figsize=(14,10))
        plt.subplot(2,1,1)
        plt.plot(bb.reset_index().index, bb['close'], color='#0000ff', linestyle='dashed', label='Close')
        plt.plot(bb.reset_index().index, bb['UpperBand'], color='r', linestyle='dashed',label='UpperBand')
        plt.plot(bb.reset_index().index, bb['LowerBand'], color='c', linestyle='dashed',label='LowerBand')
        plt.plot(bb.reset_index().index, bb['MA'],color='k', linestyle='dashed',label='Moving Average')
        plt.fill_between(bb.reset_index().index, bb['UpperBand'], bb['LowerBand'], color='gray')
        plt.title('Bollinger Band : {}'.format(code))
        plt.legend(loc='upper left')

        plt.subplot(2,1,2)
        plt.plot(bb.reset_index().index, bb['BandWidth'],color='m',alpha=0.5,label='BandWidth')
        bb_sqz = self.SqueezeBreakout(bb, sigma_lvl = 1.5)
        plt.plot(bb_sqz.reset_index().index, bb_sqz['BreakOut'], color='g', marker='^',alpha=1)
        plt.grid(True)
        plt.legend(loc='upper left')
        plt.show()
        return bb

    def TSTS_Run(self, start, end, code, long_d=130, short_d=60, signal_d=45, n=14, m=3, min_p=1, buy=20, sell=80, doplot=True):
        pr = self.ldr.GetPricelv1(start, end, [code])
        pr = pr.rename(columns={'adjprice':'close'})
        pr.index = pr['DATE'].to_list()
        pr = pr.drop(['DATE','OPEN','CODE'],axis=1)
        tsts = self.TSTS(pr, long_d=long_d, short_d=short_d, signal_d=signal_d, n=n, m=m, min_days=min_p)
        tsts_sig = self.TSTS_Signal(tsts, buy=buy, sell=sell)
        if doplot :
            plt.style.use('fivethirtyeight')
            plt.figure(figsize=(14,14))
            p1 = plt.subplot(3,1,1)
            plt.title('Triple Screen Trading System : {}'.format(code))
            plt.grid(True)
            plt.plot(tsts_sig.reset_index().index, tsts_sig['LongEMA'], color='c', label='LongEMA')
            plt.plot(tsts_sig.reset_index().index, tsts_sig['close'], color='black', label='Close', alpha=0.35)
            plt.plot(tsts_sig.reset_index().index, tsts_sig['ShortEMA'], color='navy', label='ShortEMA')
            plt.scatter(tsts_sig.reset_index().index, tsts_sig['BuySignal'], color='green', label='BuySignal', marker='^', alpha=1,linewidths=5)
            plt.scatter(tsts_sig.reset_index().index, tsts_sig['SellSignal'], color='red', label='SellSignal', marker='v', alpha=1,linewidths=5)
            plt.xlabel('Date',fontsize=15)
            plt.ylabel('Close Price KRW',fontsize=15)
            plt.legend(loc='upper left')
            
            p2 = plt.subplot(3,1,2)
            plt.grid(True)
            plt.bar(tsts_sig.reset_index().index, tsts_sig['MACD_HIST'], color='m', label='MACD-HIST')
            plt.plot(tsts_sig.reset_index().index, tsts_sig['MACD'], color='b', label='MACD')
            plt.plot(tsts_sig.reset_index().index, tsts_sig['Signal'], color='g', label='MACD-Signal')
            plt.legend(loc='upper left')
            
            p3 = plt.subplot(3,1,3)
            plt.grid(True)
            plt.plot(tsts_sig.reset_index().index, tsts_sig['fast_k'], color='c', label='Fast%K')
            plt.plot(tsts_sig.reset_index().index, tsts_sig['slow_d'], color='k', label='Slow%D')
            plt.legend(loc='upper left')
            plt.show()
            
        _, rst = self.BackTest(tsts_sig)
        rst.index = [code]
        return rst

    def SO_Run_v1(self, start, end, code,  n=14, m=5, t=5, ndays=14, buyK=20, sellK=80, buyRSI=30, sellRSI=70, velocity='fast', doplot=True):
        print("==================== Description =====================")
        print("By (20, 80) or (30, 70) and with RSI if %K's position is under/upper of that value, buy/sell.")

        pr = self.ldr.GetPricelv1(start, end, [code])
        pr = pr.sort_values(by=['DATE'])
        pr.index = pr['DATE'].to_list()
        pr = pr.drop(['DATE','OPEN','CODE'],axis=1)
        so = self.RSI(self.StochasticMaker(pr,n=n,m=m,t=t),ndays=ndays)
        if doplot :
            plt.style.use('fivethirtyeight')
            if velocity=='fast':
                so[['RSI','fast_k','fast_d']].plot(figsize=(12,6))
            elif velocity=='slow':
                so[['RSI','slow_k','slow_d']].plot(figsize=(12,6))
            plt.title("Sthochastic Oscillator with RSI for {}".format(code))
        so_sig = self.SO_Signal_v1(so, buyK=buyK, sellK=sellK, buyRSI=buyRSI, sellRSI=sellRSI,velocity=velocity)
        if doplot :
            plt.figure(figsize=(8,6))
            plt.scatter(so_sig.index, so_sig['BuySignal'], color='green',label='BuySignal',marker='^',alpha=1)
            plt.scatter(so_sig.index, so_sig['SellSignal'], color='red',label='SellSignal',marker='v',alpha=1)
            plt.plot(so_sig['close'], label='Close Price', alpha=0.35)
            plt.xticks([],rotation=45)
            plt.title('Sthochastic Oscillator on {} during {} ~ {}'.format(code,start,end))
            plt.xlabel('Date',fontsize=15)
            plt.ylabel('Close Price KRW',fontsize=15)
            plt.legend(loc='upper left')
            plt.show()
        _, rst = self.BackTest(so_sig)
        rst.index = [code]
        return rst

    def SO_Run_v2(self, start, end, code,  n=14, m=5, t=5, ndays=14, buyK=20, sellK=80, buyRSI=30, sellRSI=70, velocity='fast', doplot=True):
        print("==================== Description =====================")
        print("If K<=20 and K start golden-cross over D then buy, else K>=80 and K start dead-cross over D then sell.")

        pr = self.ldr.GetPricelv1(start, end, [code])
        pr = pr.sort_values(by=['DATE'])
        pr.index = pr['DATE'].to_list()
        pr = pr.drop(['DATE','OPEN','CODE'],axis=1)
        so = self.RSI(self.StochasticMaker(pr,n=n,m=m,t=t),ndays=ndays)
        if doplot :
            plt.style.use('fivethirtyeight')
            if velocity=='fast':
                so[['RSI','fast_k','fast_d']].plot(figsize=(12,6))
            elif velocity=='slow':
                so[['RSI','slow_k','slow_d']].plot(figsize=(12,6))
            plt.title("Sthochastic Oscillator with RSI for {}".format(code))
        so_sig = self.SO_Signal_v2(so, buyK=buyK, sellK=sellK, buyRSI=buyRSI, sellRSI=sellRSI,velocity=velocity)
        if doplot :
            plt.figure(figsize=(8,6))
            plt.scatter(so_sig.index, so_sig['BuySignal'], color='green',label='BuySignal',marker='^',alpha=1)
            plt.scatter(so_sig.index, so_sig['SellSignal'], color='red',label='SellSignal',marker='v',alpha=1)
            plt.plot(so_sig['close'], label='Close Price', alpha=0.35)
            plt.xticks([],rotation=45)
            plt.title('Sthochastic Oscillator on {} during {} ~ {}'.format(code,start,end))
            plt.xlabel('Date',fontsize=15)
            plt.ylabel('Close Price KRW',fontsize=15)
            plt.legend(loc='upper left')
            plt.show()
        _, rst = self.BackTest(so_sig)
        rst.index = [code]
        return rst

    def OBV_Run(self, start, end, code, ndays=20, doplot=True):
        pr = self.ldr.GetPricelv2(start, end, [code])
        vol = self.ldr.GetVolumelv2(start, end, [code])
        vol = vol.astype(str)
        vol[code] = vol[code].map(lambda x : x.replace(',',''))
        vol[code] = vol[code].astype(float)
        obv = self.OBV(pr,vol,ndays=ndays)
        if doplot :
            plt.style.use('fivethirtyeight')
            obv[['OBV','OBV_EMA']].plot(figsize=(8,6))
        obv_sig = self.OBV_Signal(obv)
        if doplot :
            plt.figure(figsize=(8,6))
            plt.scatter(obv_sig.index, obv_sig['BuySignal'], color='green',label='BuySignal',marker='^',alpha=1)
            plt.scatter(obv_sig.index, obv_sig['SellSignal'], color='red',label='SellSignal',marker='v',alpha=1)
            plt.plot(obv_sig['close'], label='Close Price', alpha=0.35)
            plt.xticks([],rotation=45)
            plt.title('OBV on {} during {} ~ {}'.format(code,start,end))
            plt.xlabel('Date',fontsize=15)
            plt.ylabel('Close Price KRW',fontsize=15)
            plt.legend(loc='upper left')
            plt.show()
        _, rst = self.BackTest(obv_sig)
        rst.index = [code]
        return rst

    def MFI_Run(self, start, end, code, ndays=14, buyMFI=30, sellMFI=70, doplot=True):
        pr = self.ldr.GetPricelv1(start, end, [code])
        pr = pr.sort_values(by=['DATE'])
        pr.index = pr['DATE'].to_list()
        pr = pr.drop(['DATE','OPEN','CODE'],axis=1)
        mfi = self.MFI(pr, ndays=ndays)
        if doplot :
            plt.style.use('fivethirtyeight')
            plt.figure(figsize=(12,6))
            plt.plot(mfi.reset_index().index, mfi['MFI'],label='MFI',color='gray')
            plt.axhline(10, linestyle='--', color='orange')
            plt.axhline(buyMFI, linestyle='--', color='blue')
            plt.axhline(sellMFI, linestyle='--', color='blue')
            plt.axhline(90, linestyle='--', color='orange')
            plt.title("MFI indicator for {}".format(code))
        mfi_sig = self.MFI_Signal(mfi, buyMFI=buyMFI, sellMFI=sellMFI)
        #mfi_sig_all = MFI_Signal_All(mfi, buyMFI=buyMFI, sellMFI=sellMFI)
        mfi_sig_all = mfi_sig
        if doplot :
            plt.figure(figsize=(8,6))
            plt.scatter(mfi_sig_all.index, mfi_sig_all['BuySignal'], color='green',label='BuySignal',marker='^',alpha=1)
            plt.scatter(mfi_sig_all.index, mfi_sig_all['SellSignal'], color='red',label='SellSignal',marker='v',alpha=1)
            plt.plot(mfi_sig_all['close'], label='Close Price', alpha=0.35)
            plt.xticks([],rotation=45)
            plt.title('MFI on {} during {} ~ {}'.format(code,start,end))
            plt.xlabel('Date',fontsize=15)
            plt.ylabel('Close Price KRW',fontsize=15)
            plt.legend(loc='upper left')
            plt.show()
        #mfi_sig = mfi_sig.rename(columns={'adjprice':'close'})
        _, rst = self.BackTest(mfi_sig)
        rst.index = [code]
        return rst


    def SMA_Run(self, start, end, code, long_d=26, short_d=12, min_p=1, doplot=True):
        pr = self.ldr.GetPricelv2(start, end, [code])
        sma = self.SMA_Label(pr, long_d=long_d, short_d=short_d, min_days=min_p)
        if doplot :
            plt.style.use('fivethirtyeight')
            sma[['SMA_short','SMA_long']].plot(figsize=(8,6))
            plt.title("SMA short & long for {}".format(code))
        sma_sig = self.SMA_Signal(sma)
        if doplot :
            plt.figure(figsize=(8,6))
            plt.scatter(sma_sig.index, sma_sig['BuySignal'], color='green',label='BuySignal',marker='^',alpha=1)
            plt.scatter(sma_sig.index, sma_sig['SellSignal'], color='red',label='SellSignal',marker='v',alpha=1)
            plt.plot(sma_sig['close'], label='Close Price', alpha=0.35)
            plt.xticks([],rotation=45)
            plt.title('SMA on {} during {} ~ {}'.format(code,start,end))
            plt.xlabel('Date',fontsize=15)
            plt.ylabel('Close Price KRW',fontsize=15)
            plt.legend(loc='upper left')
            plt.show()
        _, rst = self.BackTest(sma_sig)
        rst.index = [code]
        return rst

    def MACD_Run(self, start, end, code, long_d=26, short_d=12, signal_d=9, min_p=1, doplot=True):
        pr = self.ldr.GetPricelv2(start, end, [code])
        macd = self.MACD(pr, long_d=long_d, short_d=short_d, signal_d=signal_d, min_days=min_p)
        if doplot :
            plt.style.use('fivethirtyeight')
            macd[['MACD','Signal']].plot(figsize=(8,6))
            plt.title("MACD & Signal for {}".format(code))
        macd_sig = self.MACD_Signal(macd)
        if doplot :
            plt.figure(figsize=(8,6))
            plt.scatter(macd_sig.index, macd_sig['BuySignal'], color='green',label='BuySignal',marker='^',alpha=1)
            plt.scatter(macd_sig.index, macd_sig['SellSignal'], color='red',label='SellSignal',marker='v',alpha=1)
            plt.plot(macd_sig['close'], label='Close Price', alpha=0.35)
            plt.xticks([],rotation=45)
            plt.title('MACD on {} during {} ~ {}'.format(code,start,end))
            plt.xlabel('Date',fontsize=15)
            plt.ylabel('Close Price KRW',fontsize=15)
            plt.legend(loc='upper left')
            plt.show()
        _, rst = self.BackTest(macd_sig)
        rst.index = [code]
        return rst

if __name__ == '__main__':
    print("=== Insert MariaDB password ===")
    argument = sys.argv
    del argument[0]
    TI = TechnicalIndicator(argument[0])