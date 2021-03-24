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

def GetDailyPrice(all_codes):
    all_codes = all_codes + ['005935','005385','066575']
    print(len(all_codes))
    errs = []
    tmp_pr = pd.read_hdf("FullCache/Price/price_{}.h5".format(str(year)))
    with open("TradingDates.pickle","rb") as fr:
        trading_dates = pickle.load(fr)
    last_update = tmp_pr.index[-1]
    idx = trading_dates.index(last_update)
    start_date = trading_dates[idx+1]
    today = datetime.now().strftime("%Y-%m-%d")
    year=[today[:4]]
    for y in year :
        start = start_date.replace('-','')
        end = today.replace('-','')
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
        total = pd.concat([tmp_pr,total])
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

def FundaPattern_v2(end_dt = '2019-12'):
    dates = ['2016-12','2017-12','2018-12','2019-12']
    dts = []
    for dt in dates:
        if dt <= end_dt:
            dts.append(dt)
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

def FundaMatch(dt, pwd,byFunda='all'):
    conn = pymysql.connect(host='localhost',user='root', password=pwd,db='INVESTAR',charset='utf8')
    company = pd.read_sql("select * from company_info", conn)
    if dt == '2019-12':
        best_funda = pd.read_hdf("FullCache/BestFundaPattern.h5")
    else :
        best_funda = FundaPattern_v2(dt)
    dt_idx = {'2016-12':0,'2017-12':1,'2018-12':2,'2019-12':3}
    sectors = list(set(best_funda.index))
    company = company[company.sector.isin(sectors)]
    funda_codes = pd.read_pickle("FundaPatternCodes_v2.pickle")[dt_idx[dt]]
    codes = list(company.code.values)
    bestfunda_dict = {}
    for sc in sectors :
        sub_ls = []
        for fd in ['PBR','PCR','PER','POR','PSR','EPS','BPS','ROE','ROA','시가총액'] : 
            sub = best_funda[(best_funda.index==sc)&(best_funda.Funda==fd)]
            sub_ls.append(sub.loc[sc,'Funda'] + '-' + sub.loc[sc,'BestQuantile'])
        bestfunda_dict[sc] = sub_ls
    agg_codes = []
    for key in bestfunda_dict.keys():
        for bfd in bestfunda_dict[key]:
            if byFunda == 'all':
                agg_codes += funda_codes[(funda_codes.index==key)&(funda_codes['FD-Q']==bfd)].loc[key,'Codes']
            elif type(byFunda)==list and len(byFunda)!=0:
                if bfd[:3] in byFunda :
                    agg_codes += funda_codes[(funda_codes.index==key)&(funda_codes['FD-Q']==bfd)].loc[key,'Codes']
            else :
                raise ValueError("byFunda Insertion Error")
    score_dict = {}
    for cd in codes :
        if cd in agg_codes:
            score_dict[cd] = agg_codes.count(cd)
        else :
            score_dict[cd] = 0
    return score_dict

def MedianFundaPatternFull(pr):
    comp = pd.read_sql("select * from company_info",conn)
    sectors = list(comp['sector'].value_counts().index)
    fundas = ['PBR','PCR','POR','PSR','PER','EPS','BPS','ROE','ROA','시가총액']
    rst = pd.DataFrame()
    for dt in ['2016-12','2017-12','2018-12','2019-12']:
        start = str(int(dt[:4])+1)+'-04-01'
        end = str(int(dt[:4])+2)+'-03-31'
        sub_pr = pr[(pr.index>=start)&(pr.index<=end)]
        total = pd.DataFrame(index=sectors,columns=fundas)
        for sc in sectors :
            sc_ls = list(comp[lambda x : x['sector']==sc].code.values)
            for fd in fundas :
                try :
                    tmp_fd = pd.read_sql(f"select * from finance_info_copy where date='{dt}' and itm='{fd}' and code in {tuple(sc_ls)}",conn)
                    median = tmp_fd.describe().loc['50%','value']
                    low_ls = list(tmp_fd[lambda x : x['value']<=median].code.values)
                    high_ls = list(tmp_fd[lambda x : x['value']>median].code.values)
                    low_er = np.log(sub_pr[low_ls].pct_change()+1).cumsum().iloc[-1].mean()
                    high_er = np.log(sub_pr[high_ls].pct_change()+1).cumsum().iloc[-1].mean()
                    if low_er >= high_er:
                        total.loc[sc,fd] = 'L'
                    else :
                        total.loc[sc,fd] = 'H'
                except :
                    pass
        rst = pd.concat([rst,total])
    final = pd.DataFrame(index=sectors, columns=fundas)
    for sc in sectors :
        for fd in fundas :
            if len(rst.loc[sc][fd].value_counts())==1:
                final.loc[sc,fd] = rst.loc[sc][fd].values[0]
    return final

def MedianFundaTestFull(pr, param=True, cutoff=0.05):
    comp = pd.read_sql("select * from company_info",conn)
    sectors = list(comp['sector'].value_counts().index)
    fundas = ['PBR','PCR','POR','PSR','PER','EPS','BPS','ROE','ROA','시가총액']
    rst = pd.DataFrame()
    for dt in ['2016-12','2017-12','2018-12','2019-12']:
        start = str(int(dt[:4])+1)+'-04-01'
        end = str(int(dt[:4])+2)+'-03-31'
        sub_pr = pr[(pr.index>=start)&(pr.index<=end)]
        total = pd.DataFrame(index=sectors,columns=fundas)
        for sc in sectors :
            sc_ls = list(comp[lambda x : x['sector']==sc].code.values)
            for fd in fundas :
                try :
                    tmp_fd = pd.read_sql(f"select * from finance_info_copy where date='{dt}' and itm='{fd}' and code in {tuple(sc_ls)}",conn)
                    median = tmp_fd.describe().loc['50%','value']
                    low_ls = list(tmp_fd[lambda x : x['value']<=median].code.values)
                    high_ls = list(tmp_fd[lambda x : x['value']>median].code.values)
                    low_array = np.log(sub_pr[low_ls].pct_change()+1).cumsum().iloc[-1]
                    high_array = np.log(sub_pr[high_ls].pct_change()+1).cumsum().iloc[-1]
                    if param :
                        if stats.ttest_ind(low_array, high_array,  equal_var=False)[1] <= cutoff:
                            pass
                        else :
                            continue
                    else :
                        if stats.mannwhitneyu(low_array, high_array)[1] <= cutoff:
                            pass
                        else :
                            continue
                    low_er = low_array.mean()
                    high_er = high_array.mean()
                    if low_er >= high_er:
                        total.loc[sc,fd] = 'L'
                    else :
                        total.loc[sc,fd] = 'H'
                except :
                    pass
        rst = pd.concat([rst,total])
    
    return rst

def MedianFundaTestFull_v2(pr, param=True, cutoff=0.05):
    comp = pd.read_sql("select * from company_info",conn)
    sectors = list(comp['sector'].value_counts().index)
    fundas = ['PBR','PCR','POR','PSR','PER','EPS','BPS','ROE','ROA','시가총액']
    median_df = pd.DataFrame(index=sectors,columns=fundas)
    rst = pd.DataFrame()
    for dt in ['2016-12','2017-12','2018-12','2019-12']:
        start = str(int(dt[:4])+1)+'-04-01'
        end = str(int(dt[:4])+2)+'-03-31'
        sub_pr = pr[(pr.index>=start)&(pr.index<=end)]
        total = pd.DataFrame(index=sectors,columns=fundas)
        for sc in sectors :
            sc_ls = list(comp[lambda x : x['sector']==sc].code.values)
            for fd in fundas :
                try :
                    tmp_fd = pd.read_sql(f"select * from finance_info_copy where date='{dt}' and itm='{fd}' and code in {tuple(sc_ls)}",conn)
                    median = tmp_fd.describe().loc['50%','value']
                    low_ls = list(tmp_fd[lambda x : x['value']<=median].code.values)
                    high_ls = list(tmp_fd[lambda x : x['value']>median].code.values)
                    low_array = np.log(sub_pr[low_ls].pct_change()+1).cumsum().iloc[-1]
                    high_array = np.log(sub_pr[high_ls].pct_change()+1).cumsum().iloc[-1]
                    if param :
                        if stats.ttest_ind(low_array, high_array,  equal_var=False)[1] <= cutoff:
                            pass
                        else :
                            continue
                    else :
                        if stats.mannwhitneyu(low_array, high_array)[1] <= cutoff:
                            pass
                        else :
                            continue
                    low_er = low_array.mean()
                    high_er = high_array.mean()
                    if low_er >= high_er:
                        total.loc[sc,fd] = 'L'
                    else :
                        total.loc[sc,fd] = 'H'
                        
                    if dt=='2019-12':
                        median_df.loc[sc,fd] = median
                except :
                    pass
        rst = pd.concat([rst,total])
    
    return rst, median_df

def BisectFundaTest(dt,pr,param=True,cutoff=0.05):
    comp = pd.read_sql("select * from company_info",conn)
    sectors = list(comp['sector'].value_counts().index)
    fundas = ['PBR','PCR','POR','PSR','PER','EPS','BPS','ROE','ROA','시가총액']
    total = pd.DataFrame(index=sectors,columns=fundas)
    for sc in sectors :
        sc_ls = list(comp[lambda x : x['sector']==sc].code.values)
        for fd in fundas :
            try :
                tmp_fd = pd.read_sql(f"select * from finance_info_copy where date='{dt}' and itm='{fd}' and code in {tuple(sc_ls)}",conn)
                median = tmp_fd.describe().loc['50%','value']
                low_ls = list(tmp_fd[lambda x : x['value']<=median].code.values)
                high_ls = list(tmp_fd[lambda x : x['value']>median].code.values)
                low_array = np.log(pr[low_ls].pct_change()+1).cumsum().iloc[-1]
                high_array = np.log(pr[high_ls].pct_change()+1).cumsum().iloc[-1]
                if param :
                    if stats.ttest_ind(low_array, high_array,  equal_var=False)[1] <= cutoff:
                        pass
                    else :
                        continue
                else :
                    if stats.mannwhitneyu(low_array, high_array)[1] <= cutoff:
                        pass
                    else :
                        continue
                low_er = low_array.mean()
                high_er = high_array.mean()
                if low_er >= high_er:
                    total.loc[sc,fd] = 'L'
                else :
                    total.loc[sc,fd] = 'H'
            except :
                pass
    return total

def BisectFundaSimple(dt,pr):
    comp = pd.read_sql("select * from company_info",conn)
    sectors = list(comp['sector'].value_counts().index)
    fundas = ['PBR','PCR','POR','PSR','PER','EPS','BPS','ROE','ROA','시가총액']
    total = pd.DataFrame(index=sectors,columns=fundas)
    for sc in sectors :
        sc_ls = list(comp[lambda x : x['sector']==sc].code.values)
        for fd in fundas :
            try :
                tmp_fd = pd.read_sql(f"select * from finance_info_copy where date='{dt}' and itm='{fd}' and code in {tuple(sc_ls)}",conn)
                median = tmp_fd.describe().loc['50%','value']
                low_ls = list(tmp_fd[lambda x : x['value']<=median].code.values)
                high_ls = list(tmp_fd[lambda x : x['value']>median].code.values)
                low_er = np.log(pr[low_ls].pct_change()+1).cumsum().iloc[-1].mean()
                high_er = np.log(pr[high_ls].pct_change()+1).cumsum().iloc[-1].mean()
                if low_er >= high_er:
                    total.loc[sc,fd] = 'L'
                else :
                    total.loc[sc,fd] = 'H'
            except :
                pass
    return total

def SectorNThemaCodeGen(url):
    code_raw = pd.read_html(url,encoding='cp949')[2]['종목명'].dropna(axis=0).values
    code_raw = [e.replace(' *','') for e in code_raw]
    code = []
    for e in code_raw:
        try :
            code.append(ldr.FindCodeByName(e))
        except :
            pass
    return code

# total_dict = {}
# total_dict['제약'] = SectorNThemaCodeGen('https://finance.naver.com/sise/sise_group_detail.nhn?type=upjong&no=35')
# total_dict['반도체'] = SectorNThemaCodeGen('https://finance.naver.com/sise/sise_group_detail.nhn?type=upjong&no=202')
# total_dict['디스플레이'] = SectorNThemaCodeGen('https://finance.naver.com/sise/sise_group_detail.nhn?type=upjong&no=199')
# total_dict['IT'] = SectorNThemaCodeGen('https://finance.naver.com/sise/sise_group_detail.nhn?type=upjong&no=154')
# total_dict['코로나'] = SectorNThemaCodeGen('https://finance.naver.com/sise/sise_group_detail.nhn?type=theme&no=436')
# total_dict['유전자'] = SectorNThemaCodeGen('https://finance.naver.com/sise/sise_group_detail.nhn?type=theme&no=376')
# total_dict['2차전지'] = SectorNThemaCodeGen('https://finance.naver.com/sise/sise_group_detail.nhn?type=theme&no=64')
# total_dict['클라우드'] = SectorNThemaCodeGen('https://finance.naver.com/sise/sise_group_detail.nhn?type=theme&no=276')
# total_dict['쿠팡'] = SectorNThemaCodeGen('https://finance.naver.com/sise/sise_group_detail.nhn?type=theme&no=474')
# total_dict['증강현실'] = SectorNThemaCodeGen('https://finance.naver.com/sise/sise_group_detail.nhn?type=theme&no=289')
# total_dict['자율주행차'] = SectorNThemaCodeGen('https://finance.naver.com/sise/sise_group_detail.nhn?type=theme&no=362')
# total_dict['4차산업'] = SectorNThemaCodeGen('https://finance.naver.com/sise/sise_group_detail.nhn?type=theme&no=375')
# total_dict['음성인식'] = SectorNThemaCodeGen('https://finance.naver.com/sise/sise_group_detail.nhn?type=theme&no=302')
# total_dict['인터넷은행'] = SectorNThemaCodeGen('https://finance.naver.com/sise/sise_group_detail.nhn?type=theme&no=343')
# total_dict['게임'] = list(set(set(SectorNThemaCodeGen('https://finance.naver.com/sise/sise_group_detail.nhn?type=theme&no=265')).union(set(SectorNThemaCodeGen('https://finance.naver.com/sise/sise_group_detail.nhn?type=theme&no=42')))))
# total_dict['전자결제'] = SectorNThemaCodeGen('https://finance.naver.com/sise/sise_group_detail.nhn?type=theme&no=272')
# total_dict['5G'] = SectorNThemaCodeGen('https://finance.naver.com/sise/sise_group_detail.nhn?type=theme&no=373')

###################################################################################################

def GetDailyKOSPI_lv1(today='2021-02-25'):
    
    end = today.replace('-','')
    
    url = 'https://m.stock.naver.com/api/json/sise/dailySiseIndexListJson.nhn?code=KOSPI&pageSize=100'
    res = requests.get(url,headers={"User-Agent":"Chrome 88 on Windows 10"})
    rst = res.text
    tmp = res.json()
    kospi = pd.DataFrame(tmp['result']['siseList'])
    kospi['dt'] = kospi['dt'].astype(str)
    
    original = pd.read_hdf("./FullCache/KOSPI.h5")
    print("original size : {}".format(original.shape))
    
    last_date = max(original.dt.values)
    new_kospi = kospi[lambda x : x['dt'] > last_date]
    print("update size : {}".format(new_kospi.shape))
    
    original = pd.concat([new_kospi,original])
    print("merged size : {}".format(original.shape))
    
    original.to_hdf("./FullCache/KOSPI.h5",key='kospi')
    new_kospi.to_hdf("./FullCache/KOSPI_update_{}.h5".format(end),key='kospi')
    
    return True

def GetDailyKOSPI_lv2(today='2021-02-25'):
    
    if os.path.isfile(glob.glob("./FullCache/KOSPI_update_*.h5")[0]):
        pass
    else :
        raise ValueError("lv1 update file does not exist!!!")
        
    tmp = pd.read_hdf(glob.glob("./FullCache/KOSPI_update_*.h5")[0])
    total = pd.read_hdf("./FullCache/KOSPI_lv2.h5")
    print('total size : {}'.format(total.shape))
    
    dates = [dt[:4]+'-'+dt[4:6]+'-'+dt[6:8] for dt in list(tmp.dt.values)]
    tmp.index = dates
    tmp = tmp[['ncv']].sort_index()
    tmp.columns = ['close']
    print("update size : {}".format(tmp.shape))
    assert len(tmp)!=0
    total = pd.concat([total,tmp]).sort_index()
    print("merged size : {}".format(total.shape))
    total.to_hdf("./FullCache/KOSPI_lv2.h5",key='kospi')
    print("Daily lv2 KOSPI update is finished -> {} ~ {}".format(min(tmp.index), max(tmp.index)))
    
    os.remove(glob.glob("./FullCache/KOSPI_update_*.h5")[0])
    return True

def GetDailyPrice_lv1(today='2021-02-25',pwd='****'):
    conn = pymysql.connect(host='localhost',user='root',
                                   password=pwd,db='INVESTAR',charset='utf8')
    all_codes = list(pd.read_sql("select * from company_info",conn).code.values)
    all_codes = all_codes + ['005935','005385','066575']
    print("The number of codes : {}".format(len(all_codes)))
    errs = []
    year = today[:4]
    tmp_pr = pd.read_hdf("FullCache/Price/price_{}.h5".format(str(year)))
    print('original size : {}'.format(tmp_pr.shape))
    with open("TradingDates.pickle","rb") as fr:
        trading_dates = pickle.load(fr)
    start_date = pd.to_datetime(str(max(tmp_pr.DATE))).strftime("%Y-%m-%d")
    start = start_date.replace('-','')
    end = today.replace('-','')
    total = pd.DataFrame()
    assert start != end
    print("Updating prices during : {} ~ {}".format(start_date, today))
    for cd in all_codes :
        try :
            tmp = stock.get_market_ohlcv_by_date(start, end, cd, adjusted=True)
            if len(tmp) == 0 :
                errs.append(start+'|'+end+'|'+cd)
            else :
                tmp.rename(columns={'시가':'OPEN','고가':'high','저가':'low','종가':'adjprice','거래량':'volume'},inplace=True)
                tmp.index.names = ['DATE']
                tmp.reset_index(inplace=True)
                tmp['CODE'] = '{:06d}'.format(int(cd))
                tmp['CODE'] = tmp['CODE'].astype(str)
                total = pd.concat([total, tmp])
        except Exception as e:
            print(e)
            errs.append(start+'|'+end+'|'+cd)
    total.reset_index(drop=True, inplace=True)
    dates = [pd.to_datetime(str(e)).strftime("%Y-%m-%d") for e in list(total.DATE.values)]
    total['DATE'] = dates
    duplicates = list(total[total.DATE==start_date].index)
    if len(duplicates) != 0 :
        total.drop(index=duplicates, axis=0, inplace=True)
    
    new_df = total.copy()
    new_df.to_hdf("./FullCache/Price/price_update_{}.h5".format(end),key='price')
    total = pd.concat([tmp_pr, total])
    dates = [pd.to_datetime(str(e)).strftime("%Y-%m-%d") for e in list(total.DATE.values)]
    total['DATE'] = dates
    total.to_hdf("./FullCache/Price/price_{}.h5".format(str(year)),key='price')
    print('merged size : {}'.format(total.shape))
    print("Daily lv1 Price update is finished -> {} ~ {}".format(min(dates), max(dates)))
    return new_df, total, errs

def GetDailyPrice_lv2(today='2021-02-25'):
    year = today[:4]
    original_pr = pd.read_hdf("./FullCache/Price/lv2_price_{}.h5".format(str(year)))
    total_pr = pd.read_hdf("./FullCache/Price/lv2_price_total.h5")
    if os.path.isfile(glob.glob("./FullCache/Price/price_update_*.h5")[0]):
        pass
    else :
        raise ValueError("lv1 update file does not exist!!!")
    tmp_pr = pd.read_hdf(glob.glob("./FullCache/Price/price_update_*.h5")[0])
    print('original size : {}'.format(original_pr.shape))
    print('total size : {}'.format(total_pr.shape))
    
    tmp_lv2 = lv2(tmp_pr).astype(float)
    assert len(tmp_lv2)!=0
    print('update size : {}'.format(tmp_lv2.shape))
    total_pr = pd.concat([total_pr,tmp_lv2]).sort_index()
    original_pr = pd.concat([original_pr,tmp_lv2]).sort_index()
    
    total_pr.to_hdf("./FullCache/Price/lv2_price_total.h5",key='price')
    original_pr.to_hdf("./FullCache/Price/lv2_price_{}.h5".format(str(year)),key='price')
    print('merged size : {}'.format(original_pr.shape))
    print("Daily lv2 Price update is finished -> {} ~ {}".format(min(tmp_lv2.index), max(tmp_lv2.index)))
    
    os.remove(glob.glob("./FullCache/Price/price_update_*.h5")[0])
    
    return True

def lv2(df):
    codes = list(set(df.CODE.values))
    total = pd.DataFrame()
    for cd in codes :
        tmp = df[lambda x : x['CODE']==cd]
        tmp.index = list(tmp.DATE.values)
        tmp = tmp[['adjprice']]
        tmp.sort_index(inplace=True)
        tmp.columns = [cd]
        total = pd.concat([total,tmp],axis=1)
    return total

def GetMarcapCrawler(today='2021-02-25'):
    driver_path = "C:/Users/Bae Kyungmo/OneDrive/Desktop/WC_basic/chromedriver.exe"
    url = "http://data.krx.co.kr/contents/MDC/MDI/mdiLoader/index.cmd?menuId=MDC0201"
    browser = Chrome(driver_path)
    browser.maximize_window()
    browser.get(url)
    pyautogui.moveTo(262, 484, duration=1.0)
    pyautogui.click()
    pyautogui.moveTo(301, 521, duration=1.0)
    pyautogui.click()
    pyautogui.moveTo(323, 549, duration=1.0)
    pyautogui.click()
    pyautogui.moveTo(732, 419, duration=1.0)
    time.sleep(12)
    pyautogui.doubleClick()
    time.sleep(1)
    browser.find_elements_by_xpath('//*[@id="trdDd"]')[0].send_keys(today.replace('-',''))
    pyautogui.moveTo(1580, 377, duration=1.0)
    time.sleep(1)
    pyautogui.click()
    time.sleep(1)
    pyautogui.moveTo(1647, 473, duration=1.0)
    pyautogui.click()
    pyautogui.moveTo(1550, 608, duration=1.0)
    pyautogui.click()
    time.sleep(3)
    try :
        fp = glob.glob("C:/Users/Bae Kyungmo/Downloads/data*")[0]
    except :
        raise ValueError("Update file is not downloaded!!!")
    if os.path.isfile(fp):
        to_fp = "./FullCache/marcap/marcap-{}.csv".format(today)
        os.rename(fp, to_fp)
    return to_fp

def GetDailyMarcap(today='2021-02-25'):
    with open("TradingDates.pickle","rb") as fr:
        trading_dates = pickle.load(fr)
    if today not in trading_dates:
        print("The date is not a trading date.")
        return False
    year = today[:4]
    total = pd.read_csv("./FullCache/marcap/marcap-{}.csv".format(year))
    if len(total[lambda x : x['Date']==today])!=0:
        print("Already data exist at that date.")
        return False
    fp = GetMarcapCrawler(today=today)
    tmp = pd.read_csv(fp, encoding='cp949')
    assert len(tmp)!=0
    num_col = total.shape[1]
    print("original size : {}".format(total.shape))
    tmp = tmp.rename(columns={'종목코드':'Code','종목명':'Name','종가':'Close','대비':'Changes','등락률':'ChagesRatio','거래량':'Volume','거래대금':'Amount',
                   '시가':'Open','고가':'High','저가':'Low','시가총액':'Marcap','상장주식수':'Stocks','시장구분':'Market','소속부':'Dept'})
    tmp['Date'] = today
    print("update size : {}".format(tmp.shape))
    total = pd.concat([total,tmp]).sort_values(by=['Date']).reset_index(drop=True)
    print("merged size : {}".format(total.shape))
    total.to_csv("./FullCache/marcap/marcap-{}.csv".format(year),index=False)
    print("updating marcap is finished.")
    assert num_col == total.shape[1]
    os.remove(fp)
    return True

def GetDailyPrice_lv1(today='2021-02-25'):
    all_codes = ldr.codes + ['005935','005385','066575']
    print("The number of codes : {}".format(len(all_codes)))
    errs = []
    year = today[:4]
    tmp_pr = pd.read_hdf("../FullCache/Price/price_{}.h5".format(str(year)))
    print('original size : {}'.format(tmp_pr.shape))
    trading_dates = ldr.GetTradingDays()
    assert today in trading_dates
    start_date = pd.to_datetime(str(max(tmp_pr.DATE))).strftime("%Y-%m-%d")
    start = start_date.replace('-','')
    end = today.replace('-','')
    total = pd.DataFrame()
    assert start != end
    print("Updating prices during : {} ~ {}".format(start_date, today))
    for cd in all_codes :
        try :
            tmp = stock.get_market_ohlcv_by_date(start, end, cd, adjusted=True)
            if len(tmp) == 0 :
                errs.append(start+'|'+end+'|'+cd)
            else :
                tmp.rename(columns={'시가':'OPEN','고가':'high','저가':'low','종가':'adjprice','거래량':'volume'},inplace=True)
                if len(tmp[lambda x : x['volume']==0]) > 0 :
                    continue
                tmp.index.names = ['DATE']
                tmp.reset_index(inplace=True)
                tmp['CODE'] = '{:06d}'.format(int(cd))
                tmp['CODE'] = tmp['CODE'].astype(str)
                tmp['OPEN'] = tmp['OPEN'].astype(int)
                tmp['high'] = tmp['high'].astype(int)
                tmp['low'] = tmp['low'].astype(int)
                tmp['adjprice'] = tmp['adjprice'].astype(int)
                tmp['volume'] = tmp['volume'].astype(int)
                total = pd.concat([total, tmp])
        except Exception as e:
            print(e)
            errs.append(start+'|'+end+'|'+cd)
    assert len(total)!=0
    total.reset_index(drop=True, inplace=True)
    dates = [pd.to_datetime(str(e)).strftime("%Y-%m-%d") for e in list(total.DATE.values)]
    total['DATE'] = dates
    duplicates = list(total[total.DATE==start_date].index)
    if len(duplicates) != 0 :
        total.drop(index=duplicates, axis=0, inplace=True)
    
    new_df = total.copy()
    new_df.to_hdf("../FullCache/Price/price_update_{}.h5".format(end),key='price')
    total = pd.concat([tmp_pr, total])
    dates = [pd.to_datetime(str(e)).strftime("%Y-%m-%d") for e in list(total.DATE.values)]
    total['DATE'] = dates
    total.to_hdf("../FullCache/Price/price_{}.h5".format(str(year)),key='price')
    print('merged size : {}'.format(total.shape))
    print("Daily lv1 Price update is finished -> {} ~ {}".format(min(dates), max(dates)))
    return True

def DelistComplement():
    pr1 = ldr.GetPricelv1('2010-01-01','2021-03-10')
    pr1 = pr1[lambda x : x['volume']==0]
    dl_codes = list(set(pr1.CODE.values))
    delist_dict = {}
    for cd in dl_codes:
        tmp = pr1[pr1.CODE==cd].sort_values(by=['DATE'])
        delist_dict[cd] = tmp.DATE.to_list()
    return delist_dict

def lv2_volume(tdf):
    df = tdf[lambda x : x['Volume']!=0]
    codes = sorted(list(set(df.Code.values)))
    total = pd.DataFrame()
    for cd in codes :
        tmp = df[lambda x : x['Code']==cd]
        tmp.index = sorted(list(tmp.Date.values))
        tmp = tmp[['Volume']]
        tmp.sort_index(inplace=True)
        tmp.columns = [cd]
        tmp = tmp.astype(float)
        total = pd.concat([total,tmp],axis=1)
    return total

def lv2_marcap(tdf):
    df = tdf[lambda x : x['Volume']!=0]
    codes = sorted(list(set(df.Code.values)))
    total = pd.DataFrame()
    for cd in codes :
        tmp = df[lambda x : x['Code']==cd]
        tmp.index = sorted(list(tmp.Date.values))
        tmp = tmp[['Marcap']]
        tmp.sort_index(inplace=True)
        tmp.columns = [cd]
        tmp = tmp.astype(float)
        total = pd.concat([total,tmp],axis=1)
    return total

def GetDailyMarcap_lv1_lv2(today='2021-02-25'):
    with open("../FullCache/TradingDates.pickle","rb") as fr:
        trading_dates = pickle.load(fr)
    if today not in trading_dates:
        print("The date is not a trading date.")
        return False
    year = today[:4]
    total = pd.read_csv("../FullCache/marcap/marcap-{}.csv".format(year))
    if len(total[lambda x : x['Date']==today])!=0:
        print("Already data exist at that date.")
        return False
    fp = GetMarcapCrawler(today=today)
    tmp = pd.read_csv(fp, encoding='cp949')
    assert len(tmp)!=0
    num_col = total.shape[1]
    print("original size : {}".format(total.shape))
    tmp = tmp.rename(columns={'종목코드':'Code','종목명':'Name','종가':'Close','대비':'Changes','등락률':'ChagesRatio','거래량':'Volume','거래대금':'Amount',
                   '시가':'Open','고가':'High','저가':'Low','시가총액':'Marcap','상장주식수':'Stocks','시장구분':'Market','소속부':'Dept'})
    tmp['Date'] = today
    print("update size : {}".format(tmp.shape))
    total = pd.concat([total,tmp]).sort_values(by=['Date']).reset_index(drop=True)
    print("merged size : {}".format(total.shape))
    total.to_csv("../FullCache/marcap/marcap-{}.csv".format(year),index=False)
    print("updating marcap is finished.")
    assert num_col == total.shape[1]
    
    print("making daily volume update using new marcap")
    vol_lv2 = pd.read_hdf("../FullCache/VOLUME_lv2.h5")
    update_vol = lv2_volume(tmp)
    vol_lv2 = pd.concat([vol_lv2, update_vol]).sort_index()
    vol_lv2.to_hdf("../FullCache/VOLUME_lv2.h5",key='volume')
    
    print("making daily marcap update using new marcap")
    total_lv2 = pd.read_hdf("../FullCache/marcap/lv2_marcap_total.h5")
    update_lv2 = lv2_marcap(tmp)
    total_lv2 = pd.concat([total_lv2, update_lv2]).sort_index()
    total_lv2.to_hdf("../FullCache/marcap/lv2_marcap_total.h5",key='marcap')
    os.remove(fp)
    return True

def CommaRemove(mm):
    df = mm.copy()
    for cd in df.columns :
        tmp = df[[cd]].astype(str)
        tmp = tmp[cd].map(lambda x : x.replace(',',''))
        df[cd] = tmp.astype(float)
    return df

def FloatConvert(mm):
    df = mm.copy()
    for cd in df.columns :
        df[cd] = df[[cd]].astype(float)
    return df