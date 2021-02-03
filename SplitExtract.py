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

class SplitExtract:
    def __init__(self):
        """Naver Finance : Financial Summary Crawler"""
        self.driver_path = "C:/Users/Bae Kyungmo/OneDrive/Desktop/WC_basic/chromedriver.exe"
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
        self.url = 'https://finance.naver.com/item/fchart.nhn?code={}'
    
    def __del__(self):
        """Disconnecting MariaDB"""
        self.conn.close()
        
    def Extractor(self):
        #total = pd.DataFrame(columns=['code','company','split_info'])
        total_ls = []
        errors = []
        browser = Chrome(self.driver_path)
        browser.maximize_window()
        for ii, cd in enumerate(self.codes[:100]) :
            try :
                sub_ls = []
                split_info = ''
                browser.get(self.url.format(cd))
                for _ in range(5):
                    browser.find_elements_by_xpath('//*[@id="content"]/div[2]/cq-context/div[1]/div[2]/div/div[2]/div[1]')[0].click()
                for _ in range(10):
                    pyautogui.moveTo(369, 899)
                    pyautogui.dragTo(1064,899, 1, button='left')
                html = BeautifulSoup(browser.page_source, 'html.parser')
                html_div = html.find_all('div',attrs={"class":"scheduleMarker dividend"})
                for idx in range(len(html_div)):
                    if html_div[idx].text != '':
                        if idx != 0 :
                            split_info += '-'
                        split_info += html_div[idx].text
                sub_ls.append(cd)
                sub_ls.append(self.comps[ii])
                sub_ls.append(split_info)
                total_ls.append(sub_ls)
            except :
                errors.append(cd)
        total = pd.DataFrame(data=total_ls, columns=['code','company','split_info'])
        with open('./splitinfo_table.pickle','wb') as fw:
            pickle.dump(total, fw)
        if len(errors)!=0:
            with open('./splitinfo_failedlist.picke','wb') as fw:
                pickle.dump(errors, fw)
        return total, errors

    def DBSave(self, df):
        errs = []
        cursor = self.conn.cursor()
        sql = """
            CREATE TABLE IF NOT EXISTS split_info (
                code VARCHAR(20),
                company VARCHAR(40),
                split VARCHAR(100),
                PRIMARY KEY (code))
            """
        cursor.execute(sql)
        self.conn.commit()
        
        for idx, row in df.iterrows():
            if len(pd.read_sql(f"select * from split_info where code='{row.code}' and company='{row.company}'", self.conn)) == 0 :
                try :
                    #print(f"INSERT INTO split_info (code,company,split) values({row.code},{row.company},{row.split_info})")
                    cursor.execute(f"INSERT INTO split_info values('{row.code}','{row.company}','{row.split_info}')")
                    self.conn.commit()
                except :
                    if len(df[lambda x : x['code']==row.code].split_info.values[0]) >= 15 : #rough하게 VARCHAR(40) 안 넘도록
                        short_info = '폭발|' + df[lambda x : x['code']==row.code].split_info.values[0][-15:]
                        cursor.execute(f"INSERT INTO split_info values('{row.code}','{row.company}','{short_info}')")
                        self.conn.commit()
                    else :
                        errs.append(row)
        print("Uploading to DB is successfully finished.")
        return errs #"Uploading to DB is successfully finished."

if __name__ == '__main__':
    print("Starting SplitExtract Function...")
    spltexrt = SplitExtract()
    df, ers = spltexrt.Extractor()
    dbers = spltexrt.DBSave(df)