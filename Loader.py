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
        """Naver Finance : Financial Summary Crawler"""
        self.conn = pymysql.connect(host='localhost',user='root',
                                   password='secret><',db='INVESTAR',charset='utf8')
        with self.conn.cursor() as curs:
            sql_load = """
            SELECT CODE, COMPANY FROM COMPANY_INFO
            """
            curs.execute(sql_load)
            comps_ls = curs.fetchall()
            self.codes = [str(e[0]) for e in comps_ls]
            self.comps = [str(e[1]) for e in comps_ls]
            
        self.conn.commit()
                
    def __del__(self):
        """Disconnecting MariaDB"""
        self.conn.close()
        
    def FindCodeByName(self, comp_nm):
        idx = self.comps.index(comp_nm)
        return self.codes[idx]
    
    def FindNameByCode(self, comp_code):
        idx = self.codes.index(comp_code)
        return self.comps[idx]
    
    def PriceLoader(self, start, end, code_ls, item='close'):
        total = pd.DataFrame()
        for cd in code_ls :
            sql = f"SELECT DATE, {item} FROM DAILY_PRICE WHERE CODE = '{cd}' AND DATE BETWEEN '{start}' AND '{end}';"
            tmp_df = pd.read_sql(sql, self.conn)
            tmp_df.index = list(tmp_df['DATE'])
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

    def BSLoader_v3(self, start, end, code_ls, item='EPS', unit='Y', colname='name'):
        print("Note that the date you requested is for the non-PIT-ness data.")
        Q1 = ''#year+'-'+'03-31'
        Q2 = ''#year+'-'+'06-31'
        Q3 = ''#year+'-'+'09-31'
        Y = ''#year+'-'+'12-31'
        for td in self.td_days :
            if td >= year+'-'+'01-01':
                if td <= year+'-03-31':
                    Q1 = td
                elif td <= year+'-06-31':
                    Q2 = td
                elif td <= year+'-09-31':
                    Q3 = td
                elif td <= year+'-12-31':
                    Y = td
                else :
                    break
        total = pd.DataFrame()
        if item in ['시가총액','상장주식수','시가총액비중(%)']:
            if item == '시가총액':
                item = 'Marcap'
            elif item == '시가총액비중(%)':
                item == 'MarcapRatio'
            elif item == '상장주식수':
                item == 'Stocks'
            else :
                raise ValueError("Wrong Item!!!")
            file_path = "./FullCache/marcap/marcap-{}.csv"
            start_y = start[:4]
            end_y = start[:4]
            years = [y for y in range(int(start_y), int(end_y+1))]
            for idx, year in enumerate(years) :
                sub = pd.read_csv(file_path.format())
                if idx == 0 :
                    sub = sub[(sub.Code.isin(code_ls))&(sub.Date >= start)][['Code','Date',item]]
                elif idx == len(years)-1 :
                    sub = sub[(sub.Code.isin(code_ls))&(sub.Date <= end)][['Code','Date',item]]
                else :
                    sub = sub[(sub.Code.isin(code_ls))][['Code','Date',item]]
                total = pd.concat([total,sub],axis=1)
                for cd in code_ls :
                    print("not done yet...")
            
            
            
        else :
            if item not in self.items:
                for itm in self.items:
                    if item in itm:
                        item = itm
                    else :
                        raise ValueError("Item you requested does not exist..")
            #cursor = self.conn.cursor()

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


if __name__ == '__main__':
    print("Starting Loader...")
    loader = Loader()