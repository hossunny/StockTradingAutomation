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

def FundaMatch(dt, byFunda='all'):
    conn = pymysql.connect(host='localhost',user='root', password='tlqkfdk2',db='INVESTAR',charset='utf8')
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