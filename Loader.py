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

class Loader:
    def __init__(self):
        """Naver Finance : Financial Summary Crawler"""
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




if __name__ == '__main__':
    print("Starting Loader...")
    loader = Loader()