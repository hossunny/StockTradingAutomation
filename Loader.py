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

class Loader:
    def __init__(self):
        self.conn = pymysql.connect(host='localhost',user='root',
                                   password='tlqkfdk2',db='INVESTAR',charset='utf8')
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
                
    def __del__(self):
        """Disconnecting MariaDB"""
        self.conn.close()
        
    def FindCodeByName(self, comp_nm):
        idx = self.comps.index(comp_nm)
        return self.codes[idx]
    
    def FindNameByCode(self, comp_code):
        idx = self.codes.index(comp_code)
        return self.comps[idx]

    def ShowItems(self):
        return self.items
    
    def GetPrice(self, start, end, code_ls, item='close'):
        total = pd.DataFrame()
        for cd in code_ls :
            sql = f"SELECT DATE, {item} FROM DAILY_PRICE WHERE CODE = '{cd}' AND DATE BETWEEN '{start}' AND '{end}';"
            tmp_df = pd.read_sql(sql, self.conn)
            dt_ls = list(tmp_df['DATE'])
            dt_ls = [dt.strftime("%Y-%m-%d") for dt in dt_ls]
            tmp_df.index = dt_ls
            tmp_df.drop(['DATE'], axis=1, inplace=True)
            tmp_df.columns = [self.FindNameByCode(str(cd))]
            total = pd.concat([total, tmp_df], axis=1)
        total.sort_index(inplace=True)
        return total
    
    def BSLoader_v1(self, start, end, code_ls, item='EPS', unit='Annual'):
        """ Cache를 바로 parsing해서 만드는 lv2 matrix """
        """ 생각해보니까 DB에 넣어도 되잖아? (CODE,DATE)를 Primary Key로 하면.. """
        """ 중요한 점은 이런 BS DATA는 알게 되는 날짜가 한참 뒤라는 거네."""
        if item not in self.items:
            for itm in self.items :
                if item in itm :
                    item = itm
        total = pd.DataFrame()
        for cd in code_ls:
            tmp = pd.read_hdf(glob.glob(f'./FullCache/{unit}/fs_{unit.lower()}_{cd}_*')[0], key=cd, mode='r')
            tmp_cols = []
            for dt in tmp.columns :
                if dt >= start[:7] and dt <= end[:7] :
                    tmp_cols.append(dt)
            tmp_df = pd.DataFrame(data=tmp[tmp_cols].loc[item,:])
            comp_nm = self.FindNameByCode(cd)
            tmp_df.columns = [comp_nm] # list('abc') -> ['a','b','c'] || ['abc'] -> ['abc'] !!!
            tmp_df.sort_index(ascending=True, inplace=True)
            total = pd.concat([total, tmp_df],axis=1)
        return total

    def BSLoader_v2(self, start, end, code_ls, item='EPS', unit='Y', colname='name'):
        print("Note that the date you requested is for the non-PIT-ness data.")
        if item not in self.items:
            for itm in self.items:
                if item in itm:
                    item = itm
        #cursor = self.conn.cursor()
        total = pd.DataFrame()
        for cd in code_ls :
            tmp = pd.read_sql(f"select date, value from finance_info where code='{cd}' and item='{item}' and type='{unit}' and date between '{start}' and '{end}'",self.conn)
            tmp.index = tmp.date.values
            tmp.drop(['date'],axis=1,inplace=True)
            if colname == 'name':
                tmp.columns = [self.FindNameByCode(cd)]
            elif colname == 'code':
                tmp.columns = [str(cd)]
            else :
                raise ValueError("colname should be either 'name' or 'code'.")
            total = pd.concat([total, tmp],axis=1)
            
        total.sort_index(inplace=True)
        return total

    def GetFinance(self, start, end, code_ls=None, item='EPS', unit='Y', colname='name'):
        print("Note that the date you requested is for the non-PIT-ness data.")
        total = pd.DataFrame()
        if item not in self.items:
            for itm in self.items:
                if item in itm:
                    item = itm
                else :
                    raise ValueError("Item you requested does not exist..")
        #cursor = self.conn.cursor()
        if code_ls == None:
            code_ls = self.codes
        for cd in code_ls :
            tmp = pd.read_sql(f"select date, value from finance_info_copy where code='{cd}' and itm='{item}' and type='{unit}' and date between '{start}' and '{end}'",self.conn)
            tmp.index = tmp.date.values
            tmp.drop(['date'],axis=1,inplace=True)
            if item in ['EPS','BPS']:
                tmp['value'] = tmp['value'].map(lambda x : x * 100000000)
            elif item in ['PBR','PER','PCR','POR','PSR']:
                tmp['value'] = tmp['value'].map(lambda x : x / 100000000)
            else : #ROE, ROA don't need unit scaler
                pass
            if colname == 'name':
                tmp.columns = [self.FindNameByCode(cd)]
            elif colname == 'code':
                tmp.columns = [str(cd)]
            else :
                raise ValueError("colname should be either 'name' or 'code'.")
            total = pd.concat([total, tmp],axis=1)

        total.sort_index(inplace=True)
        
        return total
        
    def BSLoader_v4(self, start, end, code_ls=None, item='EPS', unit='Y', colname='name'):
        print("Note that the date you requested is for the non-PIT-ness data.")
        total = pd.DataFrame()
        if item not in self.items:
            for itm in self.items:
                if item in itm:
                    item = itm
                else :
                    raise ValueError("Item you requested does not exist..")
        #cursor = self.conn.cursor()
        if code_ls == None:
            code_ls = self.codes
        
        tmp = pd.read_sql(f"select code, date, value from finance_info_copy where code in {tuple(code_ls)} and itm='{item}' and type='{unit}' and date between '{start}' and '{end}'",self.conn)
        tmp.index = tmp.date.values
        for cd in code_ls :
            sub = tmp[lambda x : x['code']==cd]
            sub.drop(['code','date'],axis=1,inplace=True)
            if item in ['EPS','BPS']:
                sub['value'] = sub['value'].map(lambda x : x * 100000000)
            elif item in ['PBR','PER','PCR','POR','PSR']:
                sub['value'] = sub['value'].map(lambda x : x / 100000000)
            else : #ROE, ROA don't need unit scaler
                pass
            if colname == 'name':
                sub.columns = [self.FindNameByCode(cd)]
            elif colname == 'code':
                sub.columns = [str(cd)]
            else :
                raise ValueError("colname should be either 'name' or 'code'.")
            total = pd.concat([total, sub],axis=1)
        total.sort_index(inplace=True)
        
        
        return total

if __name__ == '__main__':
    print("Starting Loader...")
    loader = Loader()