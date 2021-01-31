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
    df = GetExpectedReturn_v2(df,True)
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
                    tmp_df = sub_df[lambda x : x[cl]>0]+1
                    gmeans.append(np.exp(np.log(tmp_df.T.prod(axis=1))/tmp_df.T.notna().sum(1)).values[0])
                    means.append(tmp_df[cl].mean())
                    if tmp_df[cl].mean == np.nan:
                        print(cl)

                    #tmp_df_m = (sub_df[lambda x : x[cl]<0]+1)
                    tmp_df_m = sub_df[lambda x : x[cl]<0]*(-1)+1
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

def GetExpectedReturn_v2(df, initial=True):
    if initial:
        start_date = df.index[0]
        with open("./TradingDates.pickle","rb") as fr :
            td_days = pickle.load(fr)
        idx = td_days.index(df.index[0])
        dates = td_days[idx-19:idx+1]
        sub_df = ldr.GetPrice(dates[0],dates[-1],list(df.columns),item='adjprice',colname='code')
        tdf = df.copy()
        tdf.iloc[0,:] = sub_df.mean().values
        return (tdf - tdf.iloc[0,:]) / tdf.iloc[0,:]
    else :
        return ((df - df.shift(1)) / df.shift(1)).fillna(0)



def AnnualSectorDist(sc_df_ls):
    """sc_df_ls = [sc2016,sc2017,sc2018,sc2019]"""
    rst = pd.DataFrame(columns=['Date','Funda','Pattern','C21-Best','C22-Best'])
    year = 2016
    for df in sc_df_ls:
        for sc in list(set(df.index)):
            for fd in ['PBR','PCR','POR','PSR','PER','EPS','BPS','ROE','ROA','시가총액']:
                tmp = pd.DataFrame(index=[sc],columns=['Date','Funda','Pattern','C21-Best','C22-Best'])
                sub_df = df[df['FD-Q'].isin([fd+'-'+str(i) for i in [1,2,3]])].loc[sc]
                tmp.loc[sc,'Date'] = str(year)+'-12'
                tmp.loc[sc,'Funda'] = fd
                tmp.loc[sc,'C21-Best'] = sub_df.sort_values(by=['C21-GMean'],ascending=False).loc[sc,'FD-Q'][0][-1]
                tmp.loc[sc,'C22-Best'] = sub_df.sort_values(by=['C22-GMean']).loc[sc,'FD-Q'][0][-1]
                if tmp.loc[sc,'C21-Best'] == tmp.loc[sc,'C22-Best']:
                    tmp.loc[sc,'Pattern'] = 'Y'
                else :
                    tmp.loc[sc,'Pattern'] = 'N'
                rst = pd.concat([rst,tmp])
        year += 1
    return rst

def QuarterSectorDist(sc_df_ls):
    """sc_df_ls = [sc2016,sc2017,sc2018,sc2019]"""
    rst = pd.DataFrame(columns=['Date','Funda','Pattern','C21-Best','C22-Best'])
    ith = 0
    dts = ['2019-09','2020-03','2020-06']
    for df in sc_df_ls:
        for sc in list(set(df.index)):
            for fd in ['PBR','PCR','POR','PSR','PER','EPS','BPS','ROE','ROA','시가총액']:
                tmp = pd.DataFrame(index=[sc],columns=['Date','Funda','Pattern','C21-Best','C22-Best'])
                sub_df = df[df['FD-Q'].isin([fd+'-'+str(i) for i in [1,2,3]])].loc[sc]
                tmp.loc[sc,'Date'] = dts[ith]
                tmp.loc[sc,'Funda'] = fd
                tmp.loc[sc,'C21-Best'] = sub_df.sort_values(by=['C21-GMean'],ascending=False).loc[sc,'FD-Q'][0][-1]
                tmp.loc[sc,'C22-Best'] = sub_df.sort_values(by=['C22-GMean']).loc[sc,'FD-Q'][0][-1]
                if tmp.loc[sc,'C21-Best'] == tmp.loc[sc,'C22-Best']:
                    tmp.loc[sc,'Pattern'] = 'Y'
                else :
                    tmp.loc[sc,'Pattern'] = 'N'
                rst = pd.concat([rst,tmp])
        ith += 1
    return rst

def Patternize(sec_df, fltr=None):
    """3 : 0.5/0.3/0.2 | 4 :"""
    total = pd.DataFrame(columns=['PBR','PCR','POR','PSR','PER','EPS','BPS','ROE','ROA','시가총액'])
    for sc in list(set(sec_df.index)):
        tmp = pd.DataFrame(index=[sc],columns=['PBR','PCR','POR','PSR','PER','EPS','BPS','ROE','ROA','시가총액'])
        for fd in list(set(sec_df.Funda.values)):
            try :
                sub_df = sec_df[(sec_df.Pattern=='Y')&(sec_df.index.isin([sc]))&(sec_df.Funda==fd)]
                #tmp = pd.DataFrame(index=[sc],columns=['PBR','PCR','POR','PSR','PER','EPS','BPS','ROE','ROA','시가총액'])
                if len(sub_df)==4 :
                    if len(set(sub_df['C21-Best'].values))==1:
                        tmp.loc[sc,fd] = str(sub_df['C21-Best'].values[0]) + '-Type1'
                    elif len(set(sub_df['C21-Best'].values))==2:
                        tmp.loc[sc,fd] = str(sub_df.sort_values(by=['Date'])['C21-Best'].values[-1]) + '-Type2'
                    elif len(set(sub_df['C21-Best'].values))==3:
                        tmp.loc[sc,fd] = str(sub_df['C21-Best'].value_counts().sort_values(ascending=False).index[0]) + '-Type3'
                    else : # 이 경우는 없겠네;; 분할이 1~3 뿐이니까
                        #tmp.loc[sc,fd] = '-1-Type4'
                        pass
                elif len(sub_df)!=0:
                    sub_df = sec_df[(sec_df.index.isin([sc]))&(sec_df.Funda==fd)]
                    tmp.loc[sc,fd] = str(sub_df['C22-Best'].value_counts().sort_values(ascending=False).index[0]) + '-Type4'
                else :
                    sub_df = sec_df[(sec_df.index.isin([sc]))&(sec_df.Funda==fd)]
                    tmp.loc[sc,fd] = str(sub_df['C22-Best'].value_counts().sort_values(ascending=False).index[0]) + '-Type5'
            except :
                pass
        total = pd.concat([total,tmp])
        
    if fltr == None :
        return total
    else :
        for sc in list(total.index) :
            for fd in list(total.columns):
                if total.loc[sc,fd].split('-')[-1] in fltr :
                    total.loc[sc,fd] = np.nan
        return total

def AnnualSectorDist_v3(sc_df_ls, initial=True):
    """sc_df_ls = [sc2016,sc2017,sc2018,sc2019]"""
    rst = pd.DataFrame(columns=['Date','Funda','Pattern','C31-Best','C32-Best'])
    year = 2016
    for df in sc_df_ls:
        if year == 2019 :
            start = str(year+1)+'-03-31'
            end = str(year+1)+'-12-31'
        else :
            start = str(year+1)+'-03-31'
            end = str(year+2)+'-02-28'
        for sc in list(set(df.index)):
            for fd in ['PBR','PCR','POR','PSR','PER','EPS','BPS','ROE','ROA','시가총액']:
                tmp = pd.DataFrame(index=[sc],columns=['Date','Funda','Pattern','C31-Best','C32-Best'])
                ger_total = pd.DataFrame()
                mer_total = pd.DataFrame()
                for ith in ['1','2','3']:
                    sc_idx = df[df['FD-Q'].isin([fd+'-'+ith])]['Codes'].values[0]
                    pr = ldr.GetPrice(start, end, sc_idx, item='adjprice',colname='code')
                    pr = pr.dropna(axis=1,how='any')
                    er = GetExpectedReturn_v2(pr,initial) +1
                    gm_pr = pd.DataFrame(np.exp(np.log(er.prod(axis=1))/er.notna().sum(1)),columns=[fd+'-'+ith])
                    am_pr = pd.DataFrame(er.T.mean(), columns=[fd+'-'+ith])
                    ger_total = pd.concat([ger_total, gm_pr],axis=1).dropna(axis=0,how='any')
                    mer_total = pd.concat([mer_total, am_pr],axis=1).dropna(axis=0,how='any')
                a = b = c = 0
                for d in list(ger_total.index):
                    idx = ger_total.loc[d].idxmax()
                    if idx == fd+'-1': a+=1
                    elif idx == fd+'-2' : b+=1
                    elif idx == fd+'-3' : c+=1
                    else : raise ValueError("Can't be !!!")
                var = {a:fd+'-1', b:fd+'-2', c:fd+'-3'}
                tmp.loc[sc,'C31-Best'] = var.get(max(var))
                x = y = z = 0
                for d in list(mer_total.index):
                    idx = mer_total.loc[d].idxmax()
                    if idx == fd+'-1': x+=1
                    elif idx == fd+'-2' : y+=1
                    elif idx == fd+'-3' : z+=1
                    else : raise ValueError("Can't be !!!")
                var2 = {x:fd+'-1', y:fd+'-2', z:fd+'-3'}
                tmp.loc[sc,'C32-Best'] = var2.get(max(var2))
                tmp.loc[sc,'Date'] = str(year)+'-12'
                tmp.loc[sc,'Funda'] = fd
                if tmp.loc[sc,'C31-Best'] == tmp.loc[sc,'C32-Best']:
                    tmp.loc[sc,'Pattern'] = 'Y'
                else :
                    tmp.loc[sc,'Pattern'] = 'N'
                rst = pd.concat([rst,tmp])
        year += 1
    return rst

def AnnualSectorDist_v2(sc_df_ls):
    """sc_df_ls = [sc2016,sc2017,sc2018,sc2019]"""
    rst = pd.DataFrame(columns=['Date','Funda','Pattern','C2-GBest','C2-MBest'])
    year = 2016
    for df in sc_df_ls:
        for sc in list(set(df.index)):
            for fd in ['PBR','PCR','POR','PSR','PER','EPS','BPS','ROE','ROA','시가총액']:
                tmp = pd.DataFrame(index=[sc],columns=['Date','Funda','Pattern','C2-GBest','C2-MBest'])
                sub_df = df[df['FD-Q'].isin([fd+'-'+str(i) for i in [1,2,3]])].loc[sc]
                tmp.loc[sc,'Date'] = str(year)+'-12'
                tmp.loc[sc,'Funda'] = fd
                tmp.loc[sc,'C2-GBest'] = sub_df.sort_values(by=['C2-GMean'],ascending=False).loc[sc,'FD-Q'][0][-1]
                tmp.loc[sc,'C2-MBest'] = sub_df.sort_values(by=['C2-Mean']).loc[sc,'FD-Q'][0][-1]
                if tmp.loc[sc,'C2-GBest'] == tmp.loc[sc,'C2-MBest']:
                    tmp.loc[sc,'Pattern'] = 'Y'
                else :
                    tmp.loc[sc,'Pattern'] = 'N'
                rst = pd.concat([rst,tmp])
        year += 1
    return rst

def Patternize_v3(sec_df, fltr=None):
    """3 : 0.5/0.3/0.2 | 4 :"""
    total = pd.DataFrame(columns=['PBR','PCR','POR','PSR','PER','EPS','BPS','ROE','ROA','시가총액'])
    for sc in list(set(sec_df.index)):
        tmp = pd.DataFrame(index=[sc],columns=['PBR','PCR','POR','PSR','PER','EPS','BPS','ROE','ROA','시가총액'])
        for fd in list(set(sec_df.Funda.values)):
            try :
                sub_df = sec_df[(sec_df.Pattern=='Y')&(sec_df.index.isin([sc]))&(sec_df.Funda==fd)]
                #tmp = pd.DataFrame(index=[sc],columns=['PBR','PCR','POR','PSR','PER','EPS','BPS','ROE','ROA','시가총액'])
                if len(sub_df)==4 :
                    if len(set(sub_df['C32-Best'].values))==1:
                        tmp.loc[sc,fd] = str(sub_df['C32-Best'].values[0]) + '-Type1'
                    elif len(set(sub_df['C32-Best'].values))==2:
                        tmp.loc[sc,fd] = str(sub_df.sort_values(by=['Date'])['C32-Best'].values[-1]) + '-Type2'
                    elif len(set(sub_df['C32-Best'].values))==3:
                        tmp.loc[sc,fd] = str(sub_df['C32-Best'].value_counts().sort_values(ascending=False).index[0]) + '-Type3'
                    else : # 이 경우는 없겠네;; 분할이 1~3 뿐이니까
                        #tmp.loc[sc,fd] = '-1-Type4'
                        pass
                elif len(sub_df)!=0:
                    sub_df = sec_df[(sec_df.index.isin([sc]))&(sec_df.Funda==fd)]
                    tmp.loc[sc,fd] = str(sub_df['C32-Best'].value_counts().sort_values(ascending=False).index[0]) + '-Type4'
                else :
                    sub_df = sec_df[(sec_df.index.isin([sc]))&(sec_df.Funda==fd)]
                    tmp.loc[sc,fd] = str(sub_df['C32-Best'].value_counts().sort_values(ascending=False).index[0]) + '-Type5'
            except :
                pass
        total = pd.concat([total,tmp])
        
    if fltr == None :
        return total
    else :
        for sc in list(total.index) :
            for fd in list(total.columns):
                if total.loc[sc,fd].split('-')[-1] in fltr :
                    total.loc[sc,fd] = np.nan
        return total

def FPVisual(FP, sc_df, dt):
    start = str(int(dt[:4])+1)+'-03-31'
    if dt != '2019-12':
        end = str(int(dt[:4])+2)+'-02-28'
    else :
        end = '2020-12-31'
    for sc in list(FP.index):
        sc_dict = FP.loc[sc,:].dropna(axis=0).to_dict()
        if len(sc_dict)==0:
            continue
        else :
            for sc_key in sc_dict.keys():
                BestQuantile = sc_dict[sc_key].split('-')[0]
                color=iter(cm.rainbow(np.linspace(0,1,3)))
                plt.figure(figsize=(16,12))
                print(sc, sc_key, sc_dict[sc_key])
                plt.title("Total Expected Return in sector {} with {}".format(sc,sc_key))
                plt.xlabel(f"{sc_key} with 3 quantile")
                plt.ylabel('Expected Return Rate')
                for ith in [1,2,3]:
                    if str(ith) == BestQuantile :
                        sc_idx = sc_df[sc_df.index.isin([sc])&(sc_df['FD-Q']==sc_key+'-'+str(ith))]['Codes'].values[0]
                        pr = ldr.GetPrice(start, end, sc_idx, item='adjprice',colname='code')
                        #print('shape of origin pr : ',pr.shape)
                        pr = pr.dropna(axis=1, how='any')
                        #print('shape of non-nan pr : ',pr.shape)
                        mean_er = GetExpectedReturn_v2(pr,False).T.mean()
                        #c = next(color)
                        plt.plot(list(mean_er.index), mean_er.values, color='black', linestyle='--', label='Best : {}'.format(sc_key+'-'+str(ith)))
                    else :
                        sc_idx = sc_df[sc_df.index.isin([sc])&(sc_df['FD-Q']==sc_key+'-'+str(ith))]['Codes'].values[0]
                        pr = ldr.GetPrice(start, end, sc_idx, item='adjprice',colname='code')
                        #print('shape of origin pr : ',pr.shape)
                        pr = pr.dropna(axis=1, how='any')
                        #print('shape of non-nan pr : ',pr.shape)
                        mean_er = GetExpectedReturn_v2(pr,True).T.mean()
                        c = ['b','g','r'][ith-1]
                        plt.plot(list(mean_er.index), mean_er.values, color=c, linestyle='--', label=sc_key+'-'+str(ith))
                
                plt.legend(loc='upper left')
                plt.grid(True)
                plt.show()
                        
    return True

def AnnualSectorDist_v4(sc_df_ls, initial=True):
    """sc_df_ls = [sc2016,sc2017,sc2018,sc2019]"""
    rst = pd.DataFrame(columns=['Date','Funda','Pattern','C31-Best','C31-Count','C32-Best','C32-Count','C33-Best','C33-Count'])
    year = 2016
    for df in sc_df_ls:
        if year == 2019 :
            start = str(year+1)+'-03-31'
            end = str(year+1)+'-12-31'
        else :
            start = str(year+1)+'-03-31'
            end = str(year+2)+'-02-28'
        for sc in ['반도체 제조업']:#list(set(df.index)):
            for fd in ['PBR','PCR','POR','PSR','PER','EPS','BPS','ROE','ROA','시가총액']:
                tmp = pd.DataFrame(index=[sc],columns=['Date','Funda','Pattern','C31-Best','C31-Count','C32-Best','C32-Count','C33-Best','C33-Count'])
                ger_total = pd.DataFrame()
                mer_total = pd.DataFrame()
                positives = []
                for ith in ['1','2','3']:
                    sc_idx = df[df['FD-Q'].isin([fd+'-'+ith])]['Codes'].values[0]
                    pr = ldr.GetPrice(start, end, sc_idx, item='adjprice',colname='code')
                    pr = pr.dropna(axis=1,how='any')
                    er = GetExpectedReturn_v2(pr,initial) +1
                    gm_pr = pd.DataFrame(np.exp(np.log(er.prod(axis=1))/er.notna().sum(1)),columns=[fd+'-'+ith])
                    am_pr = pd.DataFrame(er.T.mean(), columns=[fd+'-'+ith])
                    ger_total = pd.concat([ger_total, gm_pr],axis=1).dropna(axis=0,how='any')
                    mer_total = pd.concat([mer_total, am_pr],axis=1).dropna(axis=0,how='any')
                    positives.append(len(gm_pr[lambda x : x[fd+'-'+ith]>=1]))
                a = b = c = 0
                for d in list(ger_total.index):
                    idx = ger_total.loc[d].idxmax()
                    if idx == fd+'-1': a+=1
                    elif idx == fd+'-2' : b+=1
                    elif idx == fd+'-3' : c+=1
                    else : raise ValueError("Can't be !!!")
                var = {a:'1', b:'2', c:'3'}
                tmp.loc[sc,'C31-Best'] = var.get(max(var))
                tmp.loc[sc,'C31-Count'] = [a,b,c]
                x = y = z = 0
                for d in list(mer_total.index):
                    idx = mer_total.loc[d].idxmax()
                    if idx == fd+'-1': x+=1
                    elif idx == fd+'-2' : y+=1
                    elif idx == fd+'-3' : z+=1
                    else : raise ValueError("Can't be !!!")
                var2 = {x:'1', y:'2', z:'3'}
                tmp.loc[sc,'C32-Best'] = var2.get(max(var2))
                var3 = {positives[0]:'1', positives[1]:'2', positives[2]:'3'}
                tmp.loc[sc,'C33-Best'] = var3.get(max(var3))
                tmp.loc[sc,'C33-Count'] = positives
                tmp.loc[sc,'C32-Count'] = [x,y,z]
                tmp.loc[sc,'Date'] = str(year)+'-12'
                tmp.loc[sc,'Funda'] = fd
                if tmp.loc[sc,'C31-Best'] == tmp.loc[sc,'C32-Best']:
                    tmp.loc[sc,'Pattern'] = 'Y'
                else :
                    tmp.loc[sc,'Pattern'] = 'N'
                rst = pd.concat([rst,tmp])
        year += 1
    return rst