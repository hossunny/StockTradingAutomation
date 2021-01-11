import requests, glob
from datetime import datetime
from bs4 import BeautifulSoup
from datetime import datetime
import pandas as pd
import time
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
            sql = f"SELECT DATE, {item} FROM DAILY_PRICE WHERE CODE = '{cd}' AND DATE BETWEEN {start} AND {end};"
            tmp_df = pd.read_sql(sql, self.conn)
            print(tmp_df)
            tmp_df.index = list(tmp_df['date'])
            tmp_df.drop(['date'], axis=1, inplace=True)
            tmp_df.columns = [self.FindNameByCode(str(cd))]
            total = pd.concat([total, tmp_df], axis=1)
        total.sort_index(inplace=True)
        return total




if __name__ == '__main__':
    print("Starting Loader...")
    loader = Loader()