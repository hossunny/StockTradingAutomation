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

class UniverseScreener:

    def __init__(self):
        self.conn = pymysql.connect(host='localhost',user='root',
                                   password='******',db='INVESTAR',charset='utf8')
        with self.conn.cursor() as curs:
            sql_load = """
            SELECT CODE, COMPANY FROM COMPANY_INFO
            """
            curs.execute(sql_load)
            comps_ls = curs.fetchall()
            self.codes = [str(e[0]) for e in comps_ls]
            self.comps = [str(e[1]) for e in comps_ls]
            
        self.conn.commit()
        with open("./TradingDates.pickle", "rb") as fr:
            self.td_days = pickle.load(fr)
        self.items = ['매출액', '영업이익', '영업이익(발표기준)', '세전계속사업이익', '당기순이익', '당기순이익(지배)', '당기순이익(비지배)', '자산총계',
                     '부채총계', '자본총계', '자본총계(지배)', '자본총계(비지배)', '자본금', '영업활동현금흐름', '투자활동현금흐름', '재무활동현금흐름',
                     'CAPEX', 'FCF', '이자발생부채', '영업이익률', '순이익률', 'ROE(%)', 'ROA(%)', '부채비율', '자본유보율', 'EPS(원)', 'PER(배)', 'BPS(원)',
                     'PBR(배)', '현금DPS(원)', '현금배당수익률', '현금배당성향(%)', '발행주식수(보통주)','시가총액','상장주식수','시가총액비중(%)',
                     'PBR','PER','PCR','POR','PSR','ROE','ROA','EPS','BPS']
    
    def GetExpectedReturn(df, initial=True):
        if initial:
            return (df - df.iloc[0,:]) / df.iloc[0,:]
        else :
            return ((df - df.shift(1)) / df.shift(1)).fillna(0)
    
    def Filtering(dt, by=None):
        conn = self.conn
        code_ls = list(pd.read_sql("select code from company_info",conn).code.values)
        """Basic Filtering"""
        # 자본 총계가 하위 50% or 연평균(?합?) 거래량이 하위 30%
        tdf = pd.read_sql(f"select * from finance_info_copy where date='{dt}' and itm='자본총계'",conn)
        equity_half = tdf['value'].quantile(q=0.3, interpolation='nearest')
        equity_half_ls = set(tdf[lambda x : x['value']<=equity_half].code.values)
        code_ls = list(set(code_ls) - equity_half_ls)
        
        df = pd.read_sql(f"select code, date, volume from daily_price where date between '{dt[:4]+'-01-01'}' and '{dt[:4]+'-12-31'}'",conn)
        volume_30 = df.groupby(by='code').mean()['volume'].quantile(q=0.2,interpolation='nearest')
        volume_30_ls = set(df[lambda x : x['volume']<=volume_30].code.values)
        code_ls = list(set(code_ls) - volume_30_ls)
        
        if by == None :
            return code_ls
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
            return code_ls

    def VisualTest(dt, conn, term=22, itm='PBR', unit='Y', fiterby=['PBR','PCR','POR']):
        #ldr = Loader()
        code_ls = list(pd.read_sql("select code from company_info",conn).code.values)
        with open("./TradingDates.pickle", "rb") as fr:
            td_days = pickle.load(fr)
        """Finding start date"""
        cn = conn.cursor()
        cn.execute("select max(date) from daily_price where code='005930'")
        last_update = cn.fetchone()[0].strftime("%Y-%m-%d")
        start=''
        date_ls=[]
        if unit == 'Y':
            next_year = str(int(dt[:4])+1)
            for td in td_days:
                if td >= next_year+'-01-01':
                    if td <= next_year+'04-31':
                        start = td
                    else :
                        break
            pointer = td_days.index(start)
            while (pointer <= td_days.index(last_update)):
                date_ls.append(td_days[pointer])
                pointer += term
        elif unit == 'Q':
            pass
        else :
            raise ValueError("Can't be..")
                
        
        sub_ls = Filtering(dt, conn, by=fiterby)
        #start_dt = str(int(dt[:4])+1)+'-01-01'
        #end_dt = str(int(dt[:4])+1)+'-12-31'
        
        fn_df = ldr.GetFinance(dt[:4]+'-01-01', dt+'-31', item=itm, code_ls=sub_ls, unit='Y', colname='code')
        fn_df.dropna(axis=1, inplace=True)
        #pr_df = ldr.GetPrice(Q1, Q5, sub_ls, item='adjprice',colname='code')
        pr_df = ldr.GetPricePerTerm(date_ls, sub_ls, item='adjprice',colname='code')
                
        #pr_df = pr_df[pr_df.index.isin([Q1,Q2,Q3,Q4,Q5])]
        pr_df.dropna(axis=1, inplace=True)
        sub_ls = list(set(fn_df.columns).intersection(set(pr_df.columns)))
        print("Universe Size : ",len(sub_ls))
        fn_df = fn_df[sub_ls].T
        fn_df.rename({dt:"value"},axis=1,inplace=True)
        pr_df = pr_df[sub_ls]
        pr_df = GetExpectedReturn(pr_df).T
        pr_df.drop(columns=pr_df.columns[0],axis=1,inplace=True)
        total = pd.concat([fn_df,pr_df],axis=1)
        total = total.groupby(pd.qcut(total['value'],10)).agg(['mean'])
        total.columns = ['value']+date_ls[1:]
        
        date_ls = date_ls[1:]
        row_space = round(len(date_ls)/3)
        fig = plt.figure(figsize=(20,60))
        for i in range(len(date_ls)):
            #plt.subplot(row_space,2,i+1)
            ax = fig.add_subplot(row_space,3,1+i)
            plt.title("Expected Return at {} with {}".format(date_ls[i], itm))
            ax.scatter(total['value'],total[date_ls[i]],color='g')
            #plt.scatter(total['value'],total[date_ls[i]],color='g')

        plt.show()
        return total

    def VisualTest_v4(dt, conn, cut=10, term=22, itm='PBR', unit='Y', filterby=['PBR','PCR','POR']):
        #ldr = Loader()
        code_ls = list(pd.read_sql("select code from company_info",conn).code.values)
        with open("./TradingDates.pickle", "rb") as fr:
            td_days = pickle.load(fr)
        """Finding start date"""
        cn = conn.cursor()
        cn.execute("select max(date) from daily_price where code='005930'")
        last_update = cn.fetchone()[0].strftime("%Y-%m-%d")
        start=''
        end=''
        date_ls=[]
        if unit == 'Y':
            next_year = str(int(dt[:4])+1)
            for td in td_days:
                if td >= next_year+'-01-01':
                    if td <= next_year+'-03-03':
                        start = td
                    elif td <= str(int(next_year)+1)+'-03-03' and td <= last_update:
                        end = td
                    else :
                        break
            pointer = td_days.index(start)
            while (pointer <= td_days.index(end)):
                date_ls.append(td_days[pointer])
                pointer += term
        elif unit == 'Q':
            print("Not implemented yet.")
            pass
        else :
            raise ValueError("Can't be..")
                
        print("Start Date : ",start)
        #print("Dates : ",date_ls)
        sub_ls = Filtering(dt, conn, by=filterby)
        #start_dt = str(int(dt[:4])+1)+'-01-01'
        #end_dt = str(int(dt[:4])+1)+'-12-31'
        
        fn_df = ldr.GetFinance(dt[:4]+'-01-01', dt+'-31', item=itm, code_ls=sub_ls, unit='Y', colname='code')
        fn_df.dropna(axis=1, inplace=True)
        #pr_df = ldr.GetPrice(Q1, Q5, sub_ls, item='adjprice',colname='code')
        pr_df = ldr.GetPricePerTerm(date_ls, sub_ls, item='adjprice',colname='code')
                
        #pr_df = pr_df[pr_df.index.isin([Q1,Q2,Q3,Q4,Q5])]
        pr_df.dropna(axis=1, inplace=True)
        sub_ls = list(set(fn_df.columns).intersection(set(pr_df.columns)))
        print("Universe Size : ",len(sub_ls))
        fn_df = fn_df[sub_ls].T
        fn_df.rename({dt:"value"},axis=1,inplace=True)
        pr_df = pr_df[sub_ls]
        pr_df = GetExpectedReturn(pr_df).T
        pr_df.drop(columns=pr_df.columns[0],axis=1,inplace=True)
        total = pd.concat([fn_df,pr_df],axis=1)
        raw_df = total.copy()
        total = total.groupby(pd.qcut(total['value'],cut)).agg(['mean'])
        #print(total.columns)
        #print(date_ls)
        total.columns = ['value']+date_ls[1:]
        
        date_ls = date_ls[1:]
        """
        row_space = round(len(date_ls)/3)
        fig = plt.figure(figsize=(20,60))
        for i in range(len(date_ls)):
            #plt.subplot(row_space,2,i+1)
            ax = fig.add_subplot(row_space,3,1+i)
            plt.title("Expected Return at {} with {}".format(date_ls[i], itm))
            ax.scatter(total['value'],total[date_ls[i]],color='g')
            #plt.scatter(total['value'],total[date_ls[i]],color='g')

        plt.show()"""
        
        color=iter(cm.rainbow(np.linspace(0,1,cut)))
        plt.figure(figsize=(15,10))
        plt.title("Total Expected Return at some points with {}".format(itm))
        plt.xlabel(f"{itm} with {cut} quantile")
        plt.ylabel('Expected Return Rate')
        for i in range(cut):
            c = next(color)
            plt.plot(total.columns[1:], total.iloc[i,1:], color=c, marker='o', linestyle='--', label='{}-with-{}quantile'.format(itm,i+1))
        #plt.legend(loc='best')
        plt.legend(loc='upper left')
        plt.grid(True)           
        plt.show()
        return total, sub_ls, raw_df

    def VisualTest_Q(dt, conn, cut=10, term=5, itm='PBR', unit='1Q', filterby=['PBR','PCR','POR']):
        #ldr = Loader()
        code_ls = list(pd.read_sql("select code from company_info",conn).code.values)
        with open("./TradingDates.pickle", "rb") as fr:
            td_days = pickle.load(fr)
        """Finding start date"""
        cn = conn.cursor()
        cn.execute("select max(date) from daily_price where code='005930'")
        last_update = cn.fetchone()[0].strftime("%Y-%m-%d")
        start=''
        end=''
        date_ls=[]
        year = dt[:4]
        if unit=='1Q':
            for td in td_days:
                if td >= year+'-05-15':
                    if td <= year+'-05-31':
                        start = td
                    elif td <= year+'-08-10':
                        end = td
                    else :
                        break
            pointer = td_days.index(start)
            while (pointer <= td_days.index(end)):
                date_ls.append(td_days[pointer])
                pointer += term
        elif unit=='2Q':
            for td in td_days:
                if td >= year+'-08-15':
                    if td <= year+'-08-31':
                        start = td
                    elif td <= year+'-11-10':
                        end = td
                    else :
                        break
            pointer = td_days.index(start)
            while (pointer <= td_days.index(end)):
                date_ls.append(td_days[pointer])
                pointer += term
        elif unit=='3Q':
            for td in td_days:
                if td >= year+'-11-15':
                    if td <= year+'11-31':
                        start = td
                    elif td <= str(int(year)+1)+'-03-01':
                        end = td
                    else :
                        break
            pointer = td_days.index(start)
            while (pointer <= td_days.index(end)):
                date_ls.append(td_days[pointer])
                pointer += term
        else :
            raise ValueError("Not proper quarter!")
        
        print("Start Date : ",start)
        sub_ls = Filtering(dt, conn, by=filterby)
        if unit=='1Q':
            fn_df = ldr.GetFinance(dt[:4]+'-02-01', dt+'-31',item=itm, code_ls=sub_ls, unit='Q', colname='code')
        elif unit=='2Q':
            fn_df = ldr.GetFinance(dt[:4]+'-05-01', dt+'-31',item=itm, code_ls=sub_ls, unit='Q', colname='code')
        elif unit=='3Q':
            fn_df = ldr.GetFinance(dt[:4]+'-08-01', dt+'-31',item=itm, code_ls=sub_ls, unit='Q', colname='code')
        #fn_df = ldr.GetFinance(dt[:4]+'-01-01', dt+'-31', item=itm, code_ls=sub_ls, unit='Q', colname='code')
        fn_df.dropna(axis=1, inplace=True)
        #pr_df = ldr.GetPrice(Q1, Q5, sub_ls, item='adjprice',colname='code')
        pr_df = ldr.GetPricePerTerm(date_ls, sub_ls, item='adjprice',colname='code')
                
        #pr_df = pr_df[pr_df.index.isin([Q1,Q2,Q3,Q4,Q5])]
        pr_df.dropna(axis=1, inplace=True)
        sub_ls = list(set(fn_df.columns).intersection(set(pr_df.columns)))
        print("Universe Size : ",len(sub_ls))
        fn_df = fn_df[sub_ls].T
        fn_df.rename({dt:"value"},axis=1,inplace=True)
        pr_df = pr_df[sub_ls]
        pr_df = GetExpectedReturn(pr_df).T
        pr_df.drop(columns=pr_df.columns[0],axis=1,inplace=True)
        total = pd.concat([fn_df,pr_df],axis=1)
        #return fn_df, pr_df, total
        raw_df = total.copy()
        total = total.groupby(pd.qcut(total['value'],cut)).agg(['mean'])
        #print(total.columns)
        #print(date_ls)
        total.columns = ['value']+date_ls[1:]
        
        date_ls = date_ls[1:]
        color=iter(cm.rainbow(np.linspace(0,1,cut)))
        plt.figure(figsize=(15,10))
        plt.title("Total Expected Return at some points with {}".format(itm))
        plt.xlabel(f"{itm} with {cut} quantile")
        plt.ylabel('Expected Return Rate')
        for i in range(cut):
            c = next(color)
            plt.plot(total.columns[1:], total.iloc[i,1:], color=c, marker='o', linestyle='--', label='{}-with-{}quantile'.format(itm,i+1))
        #plt.legend(loc='best')
        plt.legend(loc='upper left')
        plt.grid(True)           
        plt.show()
        return total   #, sub_ls, raw_df

    def MultipleSubset(dt, univ, conn, cut=10, by=None):
        if by==None:
            by = ['PBR','PER','PCR','POR','PSR','ROE','ROA','EPS','BPS','시가총액']
        total = pd.DataFrame(index=univ)
        #total['code'] = univ
        for itm in by :
            df = pd.read_sql(f"select code, value from finance_info_copy where code in {tuple(univ)} and date='{dt}' and itm='{itm}'",conn)
            df.index = df.code
            df.drop(['code'],axis=1,inplace=True)
            sub = pd.DataFrame((pd.qcut(df['value'],cut)))
            tmp = df.groupby(pd.qcut(df['value'],cut)).agg(['mean'])
            dct = {}
            for ith, itvl in enumerate(tmp.index):
                dct[itvl] = ith + 1
            df[itm] = sub['value'].map(dct)
            
            total = pd.concat([total, df[[itm]]],axis=1)
        return total

    def concatall(df_ls,dt_ls):
        rst = pd.DataFrame()
        for idx, df in enumerate(df_ls):
            df[dt_ls[idx]+'-interval'] = df.index
            df.reset_index(drop=True, inplace=True)
            df.rename(columns={'mean':dt_ls[idx]+'-mean','gmean':dt_ls[idx]+'-gmean','label':dt_ls[idx]+'-label'},inplace=True)
            rst = pd.concat([rst,df],axis=1)
        return rst

    def VisualTest_Y_v3(dt, conn, cut=10, term=22, itm='PBR', unit='Y', filterby=['PBR','PCR','POR']):
        #ldr = Loader()
        code_ls = list(pd.read_sql("select code from company_info",conn).code.values)
        with open("./TradingDates.pickle", "rb") as fr:
            td_days = pickle.load(fr)
        """Finding start date"""
        cn = conn.cursor()
        cn.execute("select max(date) from daily_price where code='005930'")
        last_update = cn.fetchone()[0].strftime("%Y-%m-%d")
        start=''
        end=''
        date_ls=[]
        if unit == 'Y':
            next_year = str(int(dt[:4])+1)
            for td in td_days:
                if td >= next_year+'-01-01':
                    if td <= next_year+'-03-03':
                        start = td
                    elif td <= str(int(next_year)+1)+'-03-03' and td <= last_update:
                        end = td
                    else :
                        break
            pointer = td_days.index(start)
            while (pointer <= td_days.index(end)):
                date_ls.append(td_days[pointer])
                pointer += term
        elif unit == 'Q':
            print("Not implemented yet.")
            pass
        else :
            raise ValueError("Can't be..")
                
        print("Start Date : ",start)
        sub_ls = Filtering(dt, conn, by=filterby)
        fn_df = ldr.GetFinance(dt[:4]+'-01-01', dt+'-31', item=itm, code_ls=sub_ls, unit='Y', colname='code')
        fn_df.dropna(axis=1, inplace=True)
        pr_df = ldr.GetPricePerTerm(date_ls, sub_ls, item='adjprice',colname='code')
        pr_df.dropna(axis=1, inplace=True)
        sub_ls = list(set(fn_df.columns).intersection(set(pr_df.columns)))
        print("Universe Size : ",len(sub_ls))
        fn_df = fn_df[sub_ls].T
        fn_df.rename({dt:"value"},axis=1,inplace=True)
        pr_df = pr_df[sub_ls]
        pr_df = GetExpectedReturn(pr_df).T
        pr_df.drop(columns=pr_df.columns[0],axis=1,inplace=True)
        total = pd.concat([fn_df,pr_df],axis=1)
        raw_df = total.copy()
        total = total.groupby(pd.qcut(total['value'],cut)).agg(['mean'])
        total.columns = ['value']+date_ls[1:]
        date_ls = date_ls[1:]
        tmp = pd.DataFrame(total.drop(['value'],axis=1).T.mean())
        tmp.columns = list(tmp.columns[:-1])+['mean']
        labels = [i+1 for i in range(len(total))]
        total['label']=labels
        total = pd.concat([total,tmp],axis=1)
        #total.sort_values(by=['mean'],ascending=False,inplace=True)
        tmp = total.drop(['value','label','mean'],axis=1) + 1
        total['gmean'] = np.exp(np.log(tmp.prod(axis=1))/tmp.notna().sum(1)).values
        total.sort_values(by=['gmean'],ascending=False,inplace=True)
        
        return total[['mean','gmean','label']]

    def VisualTest_Q_v2(dt, conn, cut=10, term=5, itm='PBR', unit='1Q', filterby=['PBR','PCR','POR']):
        #ldr = Loader()
        code_ls = list(pd.read_sql("select code from company_info",conn).code.values)
        with open("./TradingDates.pickle", "rb") as fr:
            td_days = pickle.load(fr)
        """Finding start date"""
        cn = conn.cursor()
        cn.execute("select max(date) from daily_price where code='005930'")
        last_update = cn.fetchone()[0].strftime("%Y-%m-%d")
        start=''
        end=''
        date_ls=[]
        year = dt[:4]
        if unit=='1Q':
            for td in td_days:
                if td >= year+'-05-15':
                    if td <= year+'-05-31':
                        start = td
                    elif td <= year+'-08-10' and td<=last_update:
                        end = td
                    else :
                        break
            pointer = td_days.index(start)
            while (pointer <= td_days.index(end)):
                date_ls.append(td_days[pointer])
                pointer += term
        elif unit=='2Q':
            for td in td_days:
                if td >= year+'-08-15':
                    if td <= year+'-08-31':
                        start = td
                    elif td <= year+'-11-10' and td <= last_update:
                        end = td
                    else :
                        break
            pointer = td_days.index(start)
            while (pointer <= td_days.index(end)):
                date_ls.append(td_days[pointer])
                pointer += term
        elif unit=='3Q':
            for td in td_days:
                if td >= year+'-11-15':
                    if td <= year+'11-31':
                        start = td
                    elif td <= str(int(year)+1)+'-03-01' and td <= last_update:
                        end = td
                    else :
                        break
            pointer = td_days.index(start)
            while (pointer <= td_days.index(end)):
                date_ls.append(td_days[pointer])
                pointer += term
        else :
            raise ValueError("Not proper quarter!")
        print("Start Date : ",start)
        sub_ls = Filtering(dt, conn, by=filterby)
        if unit=='1Q':
            fn_df = ldr.GetFinance(dt[:4]+'-02-01', dt+'-31',item=itm, code_ls=sub_ls, unit='Q', colname='code')
        elif unit=='2Q':
            fn_df = ldr.GetFinance(dt[:4]+'-05-01', dt+'-31',item=itm, code_ls=sub_ls, unit='Q', colname='code')
        elif unit=='3Q':
            fn_df = ldr.GetFinance(dt[:4]+'-08-01', dt+'-31',item=itm, code_ls=sub_ls, unit='Q', colname='code')
        #fn_df = ldr.GetFinance(dt[:4]+'-01-01', dt+'-31', item=itm, code_ls=sub_ls, unit='Q', colname='code')
        fn_df.dropna(axis=1, inplace=True)
        #pr_df = ldr.GetPrice(Q1, Q5, sub_ls, item='adjprice',colname='code')
        pr_df = ldr.GetPricePerTerm(date_ls, sub_ls, item='adjprice',colname='code')
                
        #pr_df = pr_df[pr_df.index.isin([Q1,Q2,Q3,Q4,Q5])]
        pr_df.dropna(axis=1, inplace=True)
        sub_ls = list(set(fn_df.columns).intersection(set(pr_df.columns)))
        print("Universe Size : ",len(sub_ls))
        fn_df = fn_df[sub_ls].T
        fn_df.rename({dt:"value"},axis=1,inplace=True)
        pr_df = pr_df[sub_ls]
        pr_df = GetExpectedReturn(pr_df).T
        pr_df.drop(columns=pr_df.columns[0],axis=1,inplace=True)
        total = pd.concat([fn_df,pr_df],axis=1)
        raw_df = total.copy()
        total = total.groupby(pd.qcut(total['value'],cut)).agg(['mean'])
        #print(total.columns)
        #print(date_ls)
        total.columns = ['value']+date_ls[1:]
        
        date_ls = date_ls[1:]
        color=iter(cm.rainbow(np.linspace(0,1,cut)))
        plt.figure(figsize=(15,10))
        plt.title("Total Expected Return at some points with {}".format(itm))
        plt.xlabel(f"{itm} with {cut} quantile")
        plt.ylabel('Expected Return Rate')
        for i in range(cut):
            c = next(color)
            plt.plot(total.columns[1:], total.iloc[i,1:], color=c, marker='o', linestyle='--', label='{}-with-{}quantile'.format(itm,i+1))
        #plt.legend(loc='best')
        plt.legend(loc='upper left')
        plt.grid(True)           
        plt.show()
        tmp = pd.DataFrame(total.drop(['value'],axis=1).T.mean())
        tmp.columns = list(tmp.columns[:-1])+['mean']
        labels = [i+1 for i in range(len(total))]
        total['label']=labels
        total = pd.concat([total,tmp],axis=1)
        total.sort_values(by=['mean'],ascending=False,inplace=True)
        return total.iloc[:3,:]   #, sub_ls, raw_df

    def VisualTest_Q_v3(dt, conn, cut=10, term=5, itm='PBR', unit='1Q', filterby=['PBR','PCR','POR']):
        #ldr = Loader()
        code_ls = list(pd.read_sql("select code from company_info",conn).code.values)
        with open("./TradingDates.pickle", "rb") as fr:
            td_days = pickle.load(fr)
        """Finding start date"""
        cn = conn.cursor()
        cn.execute("select max(date) from daily_price where code='005930'")
        last_update = cn.fetchone()[0].strftime("%Y-%m-%d")
        start=''
        end=''
        date_ls=[]
        year = dt[:4]
        if unit=='1Q':
            for td in td_days:
                if td >= year+'-05-15':
                    if td <= year+'-05-31':
                        start = td
                    elif td <= year+'-08-10' and td<=last_update:
                        end = td
                    else :
                        break
            pointer = td_days.index(start)
            while (pointer <= td_days.index(end)):
                date_ls.append(td_days[pointer])
                pointer += term
        elif unit=='2Q':
            for td in td_days:
                if td >= year+'-08-15':
                    if td <= year+'-08-31':
                        start = td
                    elif td <= year+'-11-10' and td <= last_update:
                        end = td
                    else :
                        break
            pointer = td_days.index(start)
            while (pointer <= td_days.index(end)):
                date_ls.append(td_days[pointer])
                pointer += term
        elif unit=='3Q':
            for td in td_days:
                if td >= year+'-11-15':
                    if td <= year+'11-31':
                        start = td
                    elif td <= str(int(year)+1)+'-03-01' and td <= last_update:
                        end = td
                    else :
                        break
            pointer = td_days.index(start)
            while (pointer <= td_days.index(end)):
                date_ls.append(td_days[pointer])
                pointer += term
        else :
            raise ValueError("Not proper quarter!")
        print("Start Date : ",start)
        sub_ls = Filtering(dt, conn, by=filterby)
        if unit=='1Q':
            fn_df = ldr.GetFinance(dt[:4]+'-02-01', dt+'-31',item=itm, code_ls=sub_ls, unit='Q', colname='code')
        elif unit=='2Q':
            fn_df = ldr.GetFinance(dt[:4]+'-05-01', dt+'-31',item=itm, code_ls=sub_ls, unit='Q', colname='code')
        elif unit=='3Q':
            fn_df = ldr.GetFinance(dt[:4]+'-08-01', dt+'-31',item=itm, code_ls=sub_ls, unit='Q', colname='code')
        #fn_df = ldr.GetFinance(dt[:4]+'-01-01', dt+'-31', item=itm, code_ls=sub_ls, unit='Q', colname='code')
        fn_df.dropna(axis=1, inplace=True)
        #pr_df = ldr.GetPrice(Q1, Q5, sub_ls, item='adjprice',colname='code')
        pr_df = ldr.GetPricePerTerm(date_ls, sub_ls, item='adjprice',colname='code')
                
        #pr_df = pr_df[pr_df.index.isin([Q1,Q2,Q3,Q4,Q5])]
        pr_df.dropna(axis=1, inplace=True)
        sub_ls = list(set(fn_df.columns).intersection(set(pr_df.columns)))
        print("Universe Size : ",len(sub_ls))
        fn_df = fn_df[sub_ls].T
        fn_df.rename({dt:"value"},axis=1,inplace=True)
        pr_df = pr_df[sub_ls]
        pr_df = GetExpectedReturn(pr_df).T
        pr_df.drop(columns=pr_df.columns[0],axis=1,inplace=True)
        total = pd.concat([fn_df,pr_df],axis=1)
        raw_df = total.copy()
        total = total.groupby(pd.qcut(total['value'],cut)).agg(['mean'])
        #print(total.columns)
        #print(date_ls)
        total.columns = ['value']+date_ls[1:]
        
        date_ls = date_ls[1:]
        tmp = pd.DataFrame(total.drop(['value'],axis=1).T.mean())
        tmp.columns = list(tmp.columns[:-1])+['mean']
        labels = [i+1 for i in range(len(total))]
        total['label']=labels
        total = pd.concat([total,tmp],axis=1)
        #total.sort_values(by=['mean'],ascending=False,inplace=True)
        tmp = total.drop(['value','label','mean'],axis=1) + 1
        total['gmean'] = np.exp(np.log(tmp.prod(axis=1))/tmp.notna().sum(1)).values
        total.sort_values(by=['gmean'],ascending=False,inplace=True)
        return total[['mean','gmean','label']]

    def GeometricMean(df):
        """Note that df must be ii x di"""
        gmean = pd.DataFrame(np.exp(np.log((df+1).prod(axis=1))/(df+1).notna().sum(1))).T
        gmean = np.exp(np.log(gmean.prod(axis=1))/gmean.notna().sum(1)).values[0]
        return gmean

    def FundaComparison(dt, end, term=10, funda_ls=['PBR','PCR']):
        """Paradox of Simpson"""
        with open("./TradingDates.pickle","rb") as fr :
            td_days = pickle.load(fr)
        conn = pymysql.connect(host='localhost',user='root', password='******',db='INVESTAR',charset='utf8')
        cn = conn.cursor()
        cn.execute("select max(date) from daily_price where code='005930'")
        last_update = cn.fetchone()[0].strftime("%Y-%m-%d")
        date_ls=[]
        end_date=''
        next_year = str(int(dt[:4])+1)
        if end > last_update : end = last_update 
        for td in td_days :
            if td >= next_year+'-01-01':
                if td <= next_year+'-03-03':
                    start = td
                elif td <= end :
                    end_date = td
                else :
                    break
        pointer = td_days.index(start)
        while (pointer <= td_days.index(end_date)):
            date_ls.append(td_days[pointer])
            pointer += term
        
        filtered_ls = Filtering(dt, conn, by=['PBR','PCR','POR'])
        print("Initial Filtered Univ : {}".format(len(filtered_ls)))
        print("EX : {}".format(filtered_ls[:5]))
        if funda_ls == None :
            raise ValueError("Funda must be inserted.")
        elif len(funda_ls) == 1 :
            funda_univ = FundaUniv(dt, funda_ls=funda_ls)
            print("Funda Filtered Univ : {}".format(len(funda_univ)))
            print("EX : {}".format(funda_univ[:5]))
            pr_df = ldr.GetPricePerTerm(date_ls, funda_univ, item='adjprice',colname='code')        
            pr_df.dropna(axis=1, inplace=True)
            pr_df = GetExpectedReturn(pr_df).T
            pr_df.drop(columns=pr_df.columns[0],axis=1,inplace=True)
            gmean_funda = GeometricMean(pr_df)
            
            rest_ls = list(set(filtered_ls) - set(funda_univ))
            print("Rest Filtered Univ : {}".format(len(rest_ls)))
            print("EX : {}".format(rest_ls[:5]))
            pr_df = ldr.GetPricePerTerm(date_ls, rest_ls, item='adjprice',colname='code')      
            pr_df.dropna(axis=1, inplace=True)
            print("After Nan erased : {}".format(pr_df.shape[1]))
            pr_df = GetExpectedReturn(pr_df).T
            pr_df.drop(columns=pr_df.columns[0],axis=1,inplace=True)
            gmean_rest = GeometricMean(pr_df)
            return pd.DataFrame(data=[gmean_funda, gmean_rest], index=['FundaGmean','RestGmean'], columns=[start+'~'+end_date]).T
        else :
            funda_univ = FundaUniv(dt, funda_ls=funda_ls)
            if len(funda_univ)==0:
                total = pd.DataFrame(columns=[start+'~'+end_date]).T
            else :    
                print("Funda-All Filtered Univ : {}".format(len(funda_univ)))
                print("EX : {}".format(funda_univ[:5]))
                pr_df = ldr.GetPricePerTerm(date_ls, funda_univ, item='adjprice',colname='code')        
                pr_df.dropna(axis=1, inplace=True)
                pr_df = GetExpectedReturn(pr_df).T
                pr_df.drop(columns=pr_df.columns[0],axis=1,inplace=True)
                gmean_funda = GeometricMean(pr_df)

                rest_ls = list(set(filtered_ls) - set(funda_univ))
                print("Rest-All Filtered Univ : {}".format(len(rest_ls)))
                print("EX : {}".format(rest_ls[:5]))
                pr_df = ldr.GetPricePerTerm(date_ls, rest_ls, item='adjprice',colname='code')      
                pr_df.dropna(axis=1, inplace=True)
                print("After Nan erased : {}".format(pr_df.shape[1]))
                pr_df = GetExpectedReturn(pr_df).T
                pr_df.drop(columns=pr_df.columns[0],axis=1,inplace=True)
                gmean_rest = GeometricMean(pr_df)

                total = pd.DataFrame(data=[gmean_funda, gmean_rest], index=['FundaAllGmean','RestAllGmean'], columns=[start+'~'+end_date]).T
            print('-----------------------------------------------------------------------------------------------------------------')
            
            for fd in funda_ls :
                funda_univ = FundaUniv(dt, funda_ls=[fd])
                print("Funda Filtered Univ : {}".format(len(funda_univ)))
                print("EX : {}".format(funda_univ[:5]))
                pr_df = ldr.GetPricePerTerm(date_ls, funda_univ, item='adjprice',colname='code')        
                pr_df.dropna(axis=1, inplace=True)
                pr_df = GetExpectedReturn(pr_df).T
                pr_df.drop(columns=pr_df.columns[0],axis=1,inplace=True)
                gmean_funda = GeometricMean(pr_df)
                
                rest_ls = list(set(filtered_ls) - set(funda_univ))
                print("Rest Filtered Univ : {}".format(len(rest_ls)))
                print("EX : {}".format(rest_ls[:5]))
                pr_df = ldr.GetPricePerTerm(date_ls, rest_ls, item='adjprice',colname='code')      
                pr_df.dropna(axis=1, inplace=True)
                print("After Nan erased : {}".format(pr_df.shape[1]))
                pr_df = GetExpectedReturn(pr_df).T
                pr_df.drop(columns=pr_df.columns[0],axis=1,inplace=True)
                gmean_rest = GeometricMean(pr_df)
            
                tmp = pd.DataFrame(data=[gmean_funda, gmean_rest], index=['{}Gmean'.format(fd),'Rest{}Gmean'.format(fd)], columns=[start+'~'+end_date]).T
                
                total = pd.concat([total,tmp],axis=1)
            return total


    def FundaUniv(dt, funda_ls=['PBR','PCR']):
        conn = pymysql.connect(host='localhost',user='root', password='******',db='INVESTAR',charset='utf8')
        filtered_ls = Filtering(dt, conn, by=['PBR','PCR','POR'])
        total = MultipleSubset(dt, filtered_ls, conn, cut=5)
        total.dropna(axis=0, how='any', inplace=True)
        
        pbr = pcr = por = psr = per = eps = bps = roe = roa = mcp = [1,2,3,4,5]
        if funda_ls != None:
            for fd in funda_ls :
                if fd == 'PBR' : pbr = [3,5]
                elif fd == 'PCR' : pcr = [3,2]
                elif fd == 'POR' : por = [3,4,5]
                elif fd == 'PSR' : psr = [3,2]
                elif fd == 'PER' : per = [3,4]
                elif fd == 'EPS' : eps = [2,4]
                elif fd == 'BPS' : bps = [4,1]
                elif fd == 'ROE' : roe = [5,4,3]
                elif fd == 'ROA' : roa = [5,4]
                elif fd == 'MCP' : mcp = [4,1,2]
                else : pass
        return list(total[(total.PBR.isin(pbr))&(total.PCR.isin(pcr))&(total.POR.isin(por))&(total.PSR.isin(psr))
                    &(total.PER.isin(per))&(total.EPS.isin(eps))&(total.BPS.isin(bps))&(total.ROE.isin(roe))
                    &(total.ROA.isin(roa))&(total.시가총액.isin(mcp))].index)    

    def SummaryDataFrame(dt, end, term=10, funda_ls=['PBR','PCR'], initial=True):
        """Paradox of Simpson"""
        with open("./TradingDates.pickle","rb") as fr :
            td_days = pickle.load(fr)
        conn = pymysql.connect(host='localhost',user='root', password='******',db='INVESTAR',charset='utf8')
        cn = conn.cursor()
        cn.execute("select max(date) from daily_price where code='005930'")
        last_update = cn.fetchone()[0].strftime("%Y-%m-%d")
        date_ls=[]
        end_date=''
        next_year = str(int(dt[:4])+1)
        if end > last_update : end = last_update 
        for td in td_days :
            if td >= next_year+'-01-01':
                if td <= next_year+'-03-03':
                    start = td
                elif td <= end :
                    end_date = td
                else :
                    break
        pointer = td_days.index(start)
        while (pointer <= td_days.index(end_date)):
            date_ls.append(td_days[pointer])
            pointer += term
        
        filtered_ls = Filtering(dt, conn, by=['PBR','PCR','POR'])
        print("Initial Filtered Univ : {}".format(len(filtered_ls)))
        print("EX : {}".format(filtered_ls[:5]))
        total = pd.DataFrame(index = filtered_ls, columns = funda_ls+['gmean'])
        fn_df = pd.read_sql(f"select code, itm, value from finance_info_copy where code in {tuple(filtered_ls)} and date='{dt}' and itm in {tuple(funda_ls)}",conn)
        fn_df = fn_df[lambda x : x['value']!=-999.9]
        for idx, row in fn_df.iterrows():
            total.loc[row.code, row.itm] = row.value
        
        pr_df = ldr.GetPricePerTerm(date_ls, filtered_ls, item='adjprice',colname='code')        
        pr_df.dropna(axis=1, inplace=True)
        pr_df = GetExpectedReturn(pr_df, initial=initial).T
        pr_df.drop(columns=pr_df.columns[0],axis=1,inplace=True)
        gmean_df = pd.DataFrame(np.exp(np.log((pr_df+1).prod(axis=1))/(pr_df+1).notna().sum(1)))
        gmean_df.columns = ['gmean']
        for cdx, row in gmean_df.iterrows():
            total.loc[cdx,'gmean'] = row.gmean
        return total.astype(float)

    def CorrTable(df):
        ls = list(df.columns)
        for i in range(len(ls)) :
            for j in range(len(ls)) :
                if i<j :
                    if stats.pearsonr(df[ls[i]],df[ls[j]])[1] < 0.05 :
                        print("{} & {} are correlated -> corr : {} | p-value : {}".format(ls[i], ls[j], stats.pearsonr(df[ls[i]],df[ls[j]])[0], stats.pearsonr(df[ls[i]],df[ls[j]])[1]))
        return df.corr()

    def SectorFundaPrice(dt, fltr_by = None):
        conn = pymysql.connect(host='localhost',user='root',password='******',db='INVESTAR',charset='utf8')
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
        total = pd.DataFrame(index = sctr_ls, columns=['C1','C1(%)'])
        for sc in sctr_ls :
            sc_idx = list(tmp[lambda x : x['sector']==sc].index)
            sub_df = df[sc_idx]
            cnt = 0
            for c in sub_df.columns :
                if len(sub_df[lambda x : x[c]>=0.3])>0 or (sub_df[c].values[-1]>=0.2) :
                    if len(sub_df[lambda x : x[c]<=-0.23])==0 :
                        cnt += 1
            total.loc[sc,'C1'] = cnt
            total.loc[sc,'C1(%)'] = cnt / len(sc_idx)
        
        return total, df


if __name__ == '__main__':
    print("Starting UniverseScreener...")
    #uvscr = UniverseScreener()