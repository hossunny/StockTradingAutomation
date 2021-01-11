import logging, os
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
import warnings
warnings.filterwarnings(action='ignore')

class BSCrawler:
    def __init__(self, argv):
        """Naver Finance : Financial Summary Crawler"""
        self.driver_path = "C:/Users/Bae Kyungmo/OneDrive/Desktop/WC_basic/chromedriver.exe"
        self.update_date = datetime.today().strftime('%Y-%m-%d')
        self.conn = pymysql.connect(host='localhost',user='root',
                                   password='tlqkfdk2',db='INVESTAR',charset='utf8')
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        self.logger.addHandler(stream_handler)
        file_handler = logging.FileHandler('BSCrawling.log')
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        self.logger.info(f'Starting BSCrawler -> date : {self.update_date}')
        
        with self.conn.cursor() as curs:
            sql_load = """
            SELECT CODE, COMPANY FROM COMPANY_INFO
            """
            curs.execute(sql_load)
            comps_ls = curs.fetchall()
            self.codes = [str(e[0]) for e in comps_ls]
            self.comps = [str(e[1]) for e in comps_ls]
            
        self.conn.commit() # May not be needed..        
        self.url = 'https://finance.naver.com/item/coinfo.nhn?code={}&target=finsum_more'
        
    def __del__(self):
        """Disconnecting MariaDB"""
        self.conn.close()
        
    def DataExistChecker(self, df, n=3):
        """At least three of items should be checked."""
        pointer=0
        for i in range(n):
            for idx, col in enumerate(df.columns):
                if df.iloc[i,idx] == 0.0 or df.iloc[i,idx] == -1.0:
                    pointer = idx
                    break
        return df.columns[pointer]

    def EmptyChecker(self, df, cds):
        if df.shape[0] == 0 or df.shape[1] == 0:
            raise ValueError("There is no info in this dataframe!! : {}".format(cds))
    
    def crawler(self):
        browser = Chrome(self.driver_path)
        for idx, cds in enumerate(self.codes):
            try: 
                if os.path.isfile(glob.glob('./FullCache/Annual/*{}*'.format(cds))[0]):
                    continue
            except :
                pass
            print("Crawling *Annual* Financial Summary of *{}*...".format(self.comps[idx]+'-'+cds))
            url = self.url.format(cds)
            requests.get(url).raise_for_status()
            browser.get(url)
            try :
                browser.switch_to.frame(browser.find_element_by_id('coinfo_cp'))
                WebDriverWait(browser, 2).until(EC.element_to_be_clickable((By.XPATH,'//*[@id="cns_Tab21"]')))
            except :
                self.logger.error(f'There is no *coinfo_cp* about this company -> {self.comps[idx]+"-"+cds}')
                continue
            browser.find_elements_by_xpath('//*[@id="cns_Tab21"]')[0].click()
            html = BeautifulSoup(browser.page_source, 'html.parser')
            html_Y = html.find('table',{'class':'gHead01 all-width','summary':'주요재무정보를 제공합니다.'}) 
            html_tmp = html_Y.find('thead').find_all('tr')[1].find_all('th',attrs={"class":re.compile("^r03c")})
            dates = [''.join(re.findall('[0-9/]',html_tmp[i].text)).replace('/','-') for i in range(len(html_tmp))]
            html_tmp = html_Y.find('tbody').find_all('tr')
            cols = []
            for i in range(len(html_tmp)):
                if '\xa0' in html_tmp[i].find('th').text:
                    item = re.sub('\xa0','',html_tmp[i].find('th').text)
                else:
                    item = html_tmp[i].find('th').text
                cols.append(item)
            values = []
            for i in range(len(html_tmp)):
                tmp = html_tmp[i].find_all('td')
                value_tmp = []
                for j in range(len(tmp)):
                    try :
                        if tmp[j].text == '':
                            value_tmp.append(0.0)
                        else:
                            value_tmp.append(float(tmp[j].text.replace(',','')))
                    except :
                        value_tmp.append(-1.0)

                values.append(value_tmp)
            df = pd.DataFrame(data=values, columns=dates, index=cols)
            self.EmptyChecker(df, cds)
            #last_info_date = self.DataExistChecker(df)
            df.to_hdf(f'./FullCache/Annual/fs_annual_{cds}_{self.update_date}.h5',key=cds,mode='w')
            
            print("Crawling *Quarter* Financial Summary of *{}*...".format(self.comps[idx]+'-'+cds))
            WebDriverWait(browser, 1).until(EC.element_to_be_clickable((By.XPATH,'//*[@id="cns_Tab22"]')))
            browser.find_elements_by_xpath('//*[@id="cns_Tab22"]')[0].click()
            html = BeautifulSoup(browser.page_source, 'html.parser')
            html_Y = html.find('table',{'class':'gHead01 all-width','summary':'주요재무정보를 제공합니다.'}) 
            html_tmp = html_Y.find('thead').find_all('tr')[1].find_all('th',attrs={"class":re.compile("^r03c")})
            dates = [''.join(re.findall('[0-9/]',html_tmp[i].text)).replace('/','-') for i in range(len(html_tmp))]
            html_tmp = html_Y.find('tbody').find_all('tr')
            cols = []
            for i in range(len(html_tmp)):
                if '\xa0' in html_tmp[i].find('th').text:
                    item = re.sub('\xa0','',html_tmp[i].find('th').text)
                else:
                    item = html_tmp[i].find('th').text
                cols.append(item)
            values = []
            for i in range(len(html_tmp)):
                tmp = html_tmp[i].find_all('td')
                value_tmp = []
                for j in range(len(tmp)):
                    try :
                        if tmp[j].text == '':
                            value_tmp.append(-999.9)
                        else:
                            value_tmp.append(float(tmp[j].text.replace(',','')))
                    except :
                        value_tmp.append(-999.9)

                values.append(value_tmp)
            df = pd.DataFrame(data=values, columns=dates, index=cols)
            self.EmptyChecker(df, cds)
            #last_info_date = self.DataExistChecker(df)
            df.to_hdf(f'./FullCache/Quarter/fs_quarter_{cds}_{self.update_date}.h5',key=cds,mode='w')
            
        return "Cache generation is well done." 

if __name__ == '__main__':
    print("Starting BS Crawler...")
    argument = sys.argv
    del argument[0]
    print("Mode you requested : {}".format(argument[0]))
    bs_crawler = BSCrawler(argument[0])
    bs_crawler.crawler()