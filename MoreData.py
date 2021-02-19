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
import Loader
#ldr = Loader.Loader()
#conn = pymysql.connect(host='localhost',user='root', password='######',db='INVESTAR',charset='utf8')

def GetAllPrice(all_codes):
    errs = []
    year = [2010+i for i in range(11)]
    for y in year :
        start = str(y)+'0101'
        end = str(y)+'1231'
        total = pd.DataFrame()
        for cd in all_codes :
            try :
                tmp = stock.get_market_ohlcv_by_date(start, end, cd, adjusted=True)
                if len(tmp) == 0 :
                    continue
                else :
                    tmp.rename(columns={'시가':'OPEN','고가':'high','저가':'low','종가':'adjprice','거래량':'volume'},inplace=True)
                    tmp.index.names = ['DATE']
                    tmp.reset_index(inplace=True)
                    tmp['CODE'] = '{:06d}'.format(int(cd))
                    tmp['CODE'] = tmp['CODE'].astype(str)
                    total = pd.concat([total, tmp])
            except Exception as e:
                print(e)
                errs.append(str(y)+'-'+cd)
        #total.to_csv("./FullCache/Price/price_{}.csv".format(str(y)),index=False)
        total.to_hdf("./FullCache/Price/price_{}.h5".format(str(y)),key='price')
        print("{} is finished - size : {}".format(y, total.shape))
    return errs

def lv2(df_ls):
    real_total = pd.DataFrame()
    for df in df_ls :
        dates = sorted(list(set(df.DATE.values)))
        codes = list(set(df.CODE.values))
        total = pd.DataFrame()
        for cd in codes :
            tmp = df[lambda x : x['CODE']==cd].sort_values(by=['DATE'])[['DATE','adjprice']]
            tmp.index = [dt.strftime("%Y-%m-%d") for dt in tmp.DATE]
            tmp.drop(['DATE'],axis=1,inplace=True)
            tmp.columns = [cd]
            total = pd.concat([total,tmp],axis=1)
        real_total = pd.concat([real_total,total],axis=0)
    return real_total


def lv2_cache():
    #real_total = pd.DataFrame()
    years = [i for i in range(2010,2022)]
    for y in years :
        print(y)
        df = pd.read_hdf("FullCache/Price/price_{}.h5".format(str(y)))
        dates = sorted(list(set(df.DATE.values)))
        codes = list(set(df.CODE.values))
        total = pd.DataFrame()
        for cd in codes :
            tmp = df[lambda x : x['CODE']==cd].sort_values(by=['DATE'])[['DATE','adjprice']]
            tmp.index = [dt.strftime("%Y-%m-%d") for dt in tmp.DATE]
            tmp.drop(['DATE'],axis=1,inplace=True)
            tmp.columns = [cd]
            total = pd.concat([total,tmp],axis=1)
        total.to_hdf("FullCache/Price/lv2_price_{}.h5".format(str(y)),key='price')
        #real_total = pd.concat([real_total,total],axis=0)
    return True

""" 
# Checking whether any item I want is not there or there being different name.
%%time
non_match={}
for fp in files :
    non = []
    tmp = pd.read_excel(fp,sheet_name=2)
    code = tmp.loc[0,'symbol'].split(':')[-1]
    for dt in sorted(list(set(tmp['date'].values))):
        items = list(tmp[tmp['date']==dt]['name_ko'].values)
        if '자산총계' not in items :
            non.append(dt+'-'+'자산총계')
        elif '자본총계' not in items :
            non.append(dt+'-'+'자본총계')
        elif '당기순이익(손실)' not in items :
            if '당기순이익' not in items :
                flag = False
                for itm in items :
                    if '당기순이익' in itm :
                        flag = True
                if flag == False :
                    non.append(dt+'-'+'당기순이익(손실)|당기순이익')
        elif '영업이익(손실)' not in items :
            flag = False
            for itm in items :
                if '영업이익' in itm :
                    flag = True
            if flag == False :
                non.append(dt+'-'+'영업이익(손실)')
        elif '매출액' not in items or '총수익' not in items :
            if '수익' not in items :
                non.append(dt+'-'+'매출액|총수익|수익')
        elif '영업활동현금흐름' not in items or '영업활동으로인한현금흐름' not in items :
            non.append(dt+'-'+'영업활동현금흐름')
    if len(non)!= 0:
        non_match[code] = non

"""

"""
# Actual Extracting funda values from raw total funda files.
%%time
total = pd.DataFrame(columns = ['code','date','itm','type','value','check'])
errs = []
for fp in files :
    tmp = pd.read_excel(fp,sheet_name=2)
    code = tmp.loc[0,'symbol'].split(':')[-1]
    for dt in sorted(list(set(tmp['date'].values))):
        if dt>='2009-12-01' and dt <= '2015-12-31':
            sub = tmp[tmp['date']==dt]
            items = list(sub['name_ko'].values)
            for itm in items :
                try :
                    row_df = pd.DataFrame(columns=['code','date','itm','type','value','check'])
                    row_df.loc[0,'code'] = code
                    row_df.loc[0,'date'] = dt
                    row_df.loc[0,'type'] = 'Y'
                    #print(itm)
                    if '자산총계' in itm :
                        if itm == '자산총계':
                            row_df.loc[0,'itm'] = '자산총계'
                            row_df.loc[0,'value'] = sub[sub['name_ko']=='자산총계']['value'].values[0]
                            row_df.loc[0,'check'] = itm
                            total = pd.concat([total, row_df])
                            continue

                    if '자본총계' in itm :
                        if itm == '자본총계':
                            row_df.loc[0,'itm'] = '자본총계'
                            row_df.loc[0,'value'] = sub[sub['name_ko']=='자본총계']['value'].values[0]
                            row_df.loc[0,'check'] = itm
                            total = pd.concat([total, row_df])
                            continue
                    
                    if '당기순이익(손실)' in itm :
                        if itm == '당기순이익(손실)':
                            row_df.loc[0,'itm'] = '당기순이익'
                            row_df.loc[0,'value'] = sub[sub['name_ko']=='당기순이익(손실)']['value'].values[0]
                            row_df.loc[0,'check'] = '당기순이익(손실)'
                            total = pd.concat([total, row_df])
                            continue
                    else :
                        if '당기순이익(손실)' not in items:
                            if itm == '당기순이익':
                                row_df.loc[0,'itm'] = '당기순이익'
                                row_df.loc[0,'value'] = sub[sub['name_ko']=='당기순이익']['value'].values[0]
                                row_df.loc[0,'check'] = '당기순이익'
                                total = pd.concat([total, row_df])
                                continue
                            elif itm == '[당기순이익]':
                                row_df.loc[0,'itm'] = '당기순이익'
                                row_df.loc[0,'value'] = sub[sub['name_ko']=='[당기순이익]']['value'].values[0]
                                row_df.loc[0,'check'] = '[당기순이익]'
                                total = pd.concat([total, row_df])
                                continue

                    if '영업이익(손실)' in itm :
                        if itm == '영업이익(손실)':
                            row_df.loc[0,'itm'] = '영업이익'
                            row_df.loc[0,'value'] = sub[sub['name_ko']=='영업이익(손실)']['value'].values[0]
                            row_df.loc[0,'check'] = '영업이익(손실)'
                            total = pd.concat([total, row_df])
                            continue
                    else :
                        if '영업이익(손실)' not in items :
                            if itm == '영업이익':
                                row_df.loc[0,'itm'] = '영업이익'
                                row_df.loc[0,'value'] = sub[sub['name_ko']=='영업이익']['value'].values[0]
                                row_df.loc[0,'check'] = '영업이익'
                                total = pd.concat([total, row_df])
                                continue
                            elif itm == '계속영업이익(손실)':
                                row_df.loc[0,'itm'] = '영업이익'
                                row_df.loc[0,'value'] = sub[sub['name_ko']=='계속영업이익(손실)']['value'].values[0]
                                row_df.loc[0,'check'] = '계속영업이익(손실)'
                                total = pd.concat([total, row_df])
                                continue
                            elif itm == '계속영업이익':
                                row_df.loc[0,'itm'] = '영업이익'
                                row_df.loc[0,'value'] = sub[sub['name_ko']=='계속영업이익']['value'].values[0]
                                row_df.loc[0,'check'] = '계속영업이익'
                                total = pd.concat([total, row_df])
                                continue
                    if '매출액' in itm :
                        if itm == '매출액':
                            row_df.loc[0,'itm'] = '매출액'
                            row_df.loc[0,'value'] = sub[sub['name_ko']=='매출액']['value'].values[0]
                            row_df.loc[0,'check'] = itm
                            total = pd.concat([total, row_df])
                            continue
                    else :
                        if '수익' in itm :
                            if '매출액' not in items:
                                if itm == '총수익':
                                    row_df.loc[0,'itm'] = '매출액'
                                    row_df.loc[0,'value'] = sub[sub['name_ko']=='총수익']['value'].values[0]
                                    row_df.loc[0,'check'] = '총수익'
                                    total = pd.concat([total, row_df])
                                    continue
                                else :
                                    if '총수익' not in items and itm == '수익':
                                        row_df.loc[0,'itm'] = '매출액'
                                        row_df.loc[0,'value'] = sub[sub['name_ko']=='수익']['value'].values[0]
                                        row_df.loc[0,'check'] = '수익'
                                        total = pd.concat([total, row_df])
                                        continue
                            
                    if '영업활동현금흐름' in itm or '영업활동으로인한현금흐름' in itm :
                        if itm == '영업활동현금흐름':
                            row_df.loc[0,'itm'] = '영업활동현금흐름'
                            row_df.loc[0,'value'] = sub[sub['name_ko']=='영업활동현금흐름']['value'].values[0]
                            row_df.loc[0,'check'] = '영업활동현금흐름'
                            total = pd.concat([total, row_df])
                            continue
                        elif itm == '영업활동으로인한현금흐름':
                            row_df.loc[0,'itm'] = '영업활동현금흐름'
                            row_df.loc[0,'value'] = sub[sub['name_ko']=='영업활동으로인한현금흐름']['value'].values[0]
                            row_df.loc[0,'check'] = '영업활동으로인한현금흐름'
                            total = pd.concat([total, row_df])
                            continue
                        
                    #total = pd.concat([total, row_df])               
                except Exception as e:
                    print(e)
                    errs.append(fp+'|'+dt+'|'+str(itm))
"""

def MoreBStoDB(df, conn):
    errs = []
    cr = conn.cursor()
    for idx, row in df.iterrows():
        try :
            cd = row.code
            dt = row.date
            it = row.itm
            tp = row.type
            val = row.value
            if len(pd.read_sql(f"select * from finance_info_copy where code='{cd}' and date='{dt}' and itm='{it}' and type='Y'",conn))==0:
                qry = f"insert into finance_info_copy values('{cd}','{dt}','{it}','{tp}','{val}')"
                cr.execute(qry)
                conn.commit()
            print('-->',end=' ')
        except Exception as e:
            print(e)
            errs.append(cd+'|'+str(idx))
    return errs

def Funda(dt):
    start = str(int(dt[:4])+1)+'-04-01'
    end = str(int(dt[:4])+2)+'-03-31'
    pr1 = pd.read_hdf("FullCache/Price/lv2_price_{}.h5".format(str(int(dt[:4])+1)))
    pr2 = pd.read_hdf("FullCache/Price/lv2_price_{}.h5".format(str(int(dt[:4])+2)))
    pr = pd.concat([pr1,pr2])
    pr = pr.astype(float)
    pr = pr[(pr.index>=start)&(pr.index<=end)]
    funda_codes = pd.read_pickle("FundaPatternCodes_v2.pickle")
    dct = {}
    for i, e in enumerate(['2016-12','2017-12','2018-12','2019-12']):
        dct[e] = i
    funda_pattern = funda_codes[dct[dt]]
    all_sector = list(set(funda_pattern.index))
    all_funda = ['PBR','PCR','PER','POR','PSR','EPS','BPS','ROE','ROA','시가총액']
    total = pd.DataFrame(columns=['Funda','Quantile','MeanEarning'])
    for sc in all_sector:
        for fd in all_funda :
            max_earning = -99999
            best_quantile = '1'
            for q in ['1','2','3']:
                tmp_codes = funda_pattern[(funda_pattern.index==sc)&(funda_pattern['FD-Q']==fd+'-'+q)]['Codes'].values[0]
                tmp_earning = np.log(1+pr[tmp_codes].pct_change()).cumsum().iloc[-1,:].mean()
                if tmp_earning > max_earning:
                    max_earning = tmp_earning
                    best_quantile = q
            tmp_df = pd.DataFrame(index=[sc], columns=['Funda','Quantile','MeanEarning'])
            tmp_df.loc[sc,'Funda'] = fd
            tmp_df.loc[sc,'Quantile'] = best_quantile
            tmp_df.loc[sc,'MeanEarning'] = max_earning
            total = pd.concat([total, tmp_df])
    return total

def FundaPattern():
    dts = ['2016-12','2017-12','2018-12','2019-12']
    all_funda = pd.DataFrame()
    for dt in dts :
        all_funda = pd.concat([all_funda,Funda(dt)])
    all_sector = list(set(all_funda.index))
    fundas = list(set(all_funda.Funda.values))
    rst = pd.DataFrame(columns=['Funda','BestQuantile'])
    for sc in all_sector :
        for fd in fundas :
            b_q = all_funda[(all_funda.index==sc)&(all_funda.Funda==fd)]['Quantile'].value_counts().index[0]
            tmp = pd.DataFrame(index=[sc], columns=['Funda','BestQuantile'])
            tmp.loc[sc,'Funda'] = fd
            tmp.loc[sc,'BestQuantile'] = b_q
            rst = pd.concat([rst,tmp])
    return rst