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
import pyautogui
import shutil

class BStoDB:
    def __init__(self):
        self.update_date = datetime.today().strftime('%Y-%m-%d')
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

        self.conn.commit() # May not be needed..        
    
    def __del__(self):
        """Disconnecting MariaDB"""
        self.conn.close()

    def BStoDBSave(self):
        errs = []
        cursor = self.conn.cursor()
        with open('./NanBSList.pickle','rb') as fr:
            nanbslist = pickle.load(fr)
        sql = """
            CREATE TABLE IF NOT EXISTS finance_info (
                code VARCHAR(20),
                date VARCHAR(40),
                item VARCHAR(60),
                type VARCHAR(20),
                value DOUBLE,
                PRIMARY KEY (code, date, item, type))
            """
        cursor.execute(sql)
        self.conn.commit()
        
        for cd in self.codes :
            try :
                file_paths = glob.glob(f"./FullCache/*/*{cd}*.h5")
                if len(file_paths) == 0 :
                    if cd in nanbslist :
                        continue
                    else :
                        errs.append(cd+'|nan')
                        continue
                tp = 'N'
                for f in file_paths :
                    if 'annual' in f :
                        tp = 'Y'
                    elif 'quarter' in f :
                        tp = 'Q'
                    else :
                        raise ValueError("nontype..?")
                    tmp_df = pd.read_hdf(f).T
                    items = list(tmp_df.columns)
                    for idx, row in tmp_df.iterrows():
                        if str(idx)[:4] == 'None':
                            continue
                        else :
                            for itm in items :
                                if len(pd.read_sql(f"select * from finance_info where code='{cd}' and date='{idx}' and item='{itm}' and type='{tp}'",self.conn)) == 0 :
                                    cursor.execute(f"INSERT INTO finance_info values('{cd}','{idx}','{itm}','{tp}',{row[itm]})")
                                    self.conn.commit()
            except :
                errs.append(cd+'|why')
        print("BStoDB is done.")
        return errs 

if __name__ == '__main__':
    print("Starting BStoDB uploader...")
    bts = BStoDB()
    ers = BstoDBSave()