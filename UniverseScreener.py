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

class UniverseScreener:

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


if __name__ == '__main__':
    print("Starting UniverseScreener...")
    #uvscr = UniverseScreener()