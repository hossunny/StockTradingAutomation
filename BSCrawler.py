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

class BSCrawler:
    def __init__(self, argv):
        """Naver Finance : Financial Summary Crawler"""
        self.driver_path = "C:/Users/Bae Kyungmo/OneDrive/Desktop/WC_basic/chromedriver.exe"
        self.update_date = datetime.today().strftime('%Y-%m-%d')
        self.conn = pymysql.connect(host='localhost',user='root',
                                   password='secret><',db='INVESTAR',charset='utf8')
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        self.logger.addHandler(stream_handler)
        file_handler = logging.FileHandler('BSCrawling.log')
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        self.logger.info('=======================================================================')
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
    
    def DataValidation(self, values):
        flag = False
        cnt = 0
        for i in range(len(values)):
            for j in range(len(values[i])):
                if values[i][j] != 0.0:
                    flag = True
                    cnt += 1
        return flag, cnt
    
    def UpdateDataInfo(self, unit='Annual'):
        # unit : Annual / Quarter
        try:
            with open('./updateinfo_{}.pickle'.format(unit),'rb') as fr:
                self.updateinfo = pickle.load(fr)
        except:
                self.updateinfo = {} 
        self.logger.info(f'Updating data existence and its corresponding update info. -> date : {self.update_date}')
        nocoinfo=[]
        yescoinfo=[]
        sub_dict={}
        for cd in self.codes :
            try :
                if os.path.isfile(glob.glob('./FullCache/{}/*{}*'.format(unit,cd))[0]):
                    yescoinfo.append(cd)
                else :
                    nocoinfo.append(cd)
            except :
                nocoinfo.append(cd)
        """ N : No CoInfo / Y : Yes CoInfo / U : Update Notice"""
        sub_dict['N'] = nocoinfo
        sub_dict['Y'] = yescoinfo
        new_info = []
        if len(self.updateinfo.keys()) == 0 :
            sub_dict['U'] = new_info
        else :
            prev_date = sorted(list(self.updateinfo.keys()))[-1]
            prev_Y = self.updateinfo[prev_date]['Y']
            prev_N = self.updateinfo[prev_date]['N']
            diff_Y = list(set(prev_Y) - set(yescoinfo))
            self.logger.info('diff_Y list : {}'.format(diff_Y))
            if len(diff_Y) != 0 :
                diff_Y = ['-'+e for e in diff_Y]
            diff_N = list(set(prev_N) - set(nocoinfo))
            self.logger.info('diff_N list : {}'.format(diff_Y))
            if len(diff_N) != 0 :
                for e in diff_N:
                    if e in yescoinfo :
                        new_info.append('+'+e)
            new_info = diff_Y + new_info
        sub_dict['U'] = new_info
        self.updateinfo[self.update_date] = sub_dict
        with open('./updateinfo_{}.pickle'.format(unit),'wb') as fw:
            pickle.dump(self.updateinfo, fw) 
        self.logger.info('Updating UpdateInfo Dictionary is done.')

    def DelistingChecker(self, ndays=30):
        dels_comps = []
        for cd in self.codes:
            sql = f"SELECT DATE FROM DAILY_PRICE WHERE CODE = '{cd}';"
            temp_dates = list(pd.read_sql(sql, self.conn)['DATE'])
            temp_dates = [str(dt) for dt in temp_dates]
            temp_dates = [date(int(dt[:4]),int(dt[5:7]),int(dt[8:])) for dt in temp_dates]
            diff_day = 0
            for idx in range(len(temp_dates)):
                if idx != len(temp_dates)-1:
                    diff_day = (temp_dates[idx+1] - temp_dates[idx]).days
                    if diff_day >= ndays :
                        dels_comps.append((cd,temp_dates[idx],temp_dates[idx+1]))
        return dels_comps
    
    def StockSplitChecker(self, pwrs=2):
        stsplt_comps = [] # 액면 분할
        revsplt_comps = [] # 액면 병합
        for cd in self.codes:
            if cd in ['013890','037030']:#self.DelistingChecker(30):
                continue
            sql = f"SELECT DATE, CLOSE FROM DAILY_PRICE WHERE CODE = '{cd}';"
            tmp_df = pd.read_sql(sql,self.conn)
            tmp_df.index = list(tmp_df['DATE'])
            tmp_df.drop(['DATE'], axis=1, inplace=True)
            tmp_df.sort_index(inplace=True)
            tmp_prices = list(tmp_df['CLOSE'])
            for idx in range(len(tmp_prices)):
                if idx != len(tmp_prices)-1:
                    if tmp_prices[idx+1] >= tmp_prices[idx]*pwrs :
                        revsplt_comps.append((cd,tmp_df.index[idx],tmp_prices[idx],tmp_df.index[idx+1],tmp_prices[idx+1]))
                    elif tmp_prices[idx+1] <= float(tmp_prices[idx]) / pwrs :
                        stsplt_comps.append((cd,tmp_df.index[idx],tmp_prices[idx],tmp_df.index[idx+1],tmp_prices[idx+1]))
        return stsplt_comps, revsplt_comps    
    
    def TimeInverter(self, dt, hyphen=False):
        Y = str(dt.year)
        M = '0'+str(dt.month) if int(dt.month)<10 else str(dt.month)
        D = '0'+str(dt.day) if int(dt.day)<10 else str(dt.day)
        if hyphen :
            return Y+'-'+M+'-'+D
        else :
            return Y+M+D
    
    def AdjPriceMerge(self):
        self.logger.info('Merging AdjPrice from krx with my DB.')
        errors=[]
        cursor = self.conn.cursor()
        cursor.execute("SELECT DISTINCT CODE FROM daily_price WHERE adjprice IS NULL")
        nulls = cursor.fetchall()
        nulls = [x[0] for x in nulls]
        for cd in nulls : #self.codes :
            try :
                """
                try_sql = f"select count(*) from daily_price where code='{cd}' and adjprice>0"
                if cursor.execute(try_sql) == 0 :
                    pass
                else :
                    if cursor.execute(f"select count(*) from daily_price where code='{cd}' and adjprice>0") < cursor.execute(f"select count(*) from daily_price where code='{cd}'"):
                        pass
                    else :
                        continue
                """
                sql = f"select min(date), max(date) from daily_price where code='{cd}'"
                cursor.execute(sql)
                rst = cursor.fetchone()
                min_date, max_date = rst[0], rst[1]
                try :
                    min_date = self.TimeInverter(min_date)
                    max_date = self.TimeInverter(max_date)
                except :
                    raise ValueError("timeinverting failed")
                tmp_df = stock.get_market_ohlcv_by_date(min_date, max_date, cd, adjusted=True)
                adj_values = tmp_df['종가'].values
                dt_values = tmp_df.index
                dt_values = [self.TimeInverter(dt, hyphen=True) for dt in dt_values]
                for idx, adjp in enumerate(adj_values):
                    try :
                        if cursor.execute(f"select * from daily_price where adjprice>0 and code='{cd}' and date='{dt_values[idx]}'") == 0 :
                            update_sql = f"update daily_price set adjprice = {int(adjp)} where code = '{cd}' and date = '{dt_values[idx]}'"
                            cursor.execute(update_sql)
                        else :
                            continue
                    except :
                        self.logger.error('Merging is failed. : {}'.format((cd, dt_values[idx], int(adjp))))
                        errors.append((cd,dt_values[idx],int(adjp)))
                        raise ValueError("Exception is occured by merging adjprice error. Check the log.")
                self.conn.commit()
            except :
                errors.append(cd+'T-error')
        """ Genuine None Values -> null to -999 value insertion. """
        cursor.execute("SELECT DISTINCT CODE FROM daily_price WHERE adjprice IS NULL")
        nulls = cursor.fetchall()
        nulls = [x[0] for x in nulls]
        for cd in nulls :
            cursor.execute(f"update daily_price set adjprice=-999 where code='{cd}' and adjprice is null")
        self.conn.commit()
        self.logger.info("Merging Adjprice is well done.")
        return errors
        
    def Crawler(self):
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
            time.sleep(2)
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
            dates = ['None'+str(idx) if e=='' else e for idx, e in enumerate(dates)]
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
            flag, valid_nb = self.DataValidation(values)
            if flag and valid_nb >= 10 :
                pass
            else :
                self.logger.error(f'Not enough *coinfo_cp* about this company -> {self.comps[idx]+"-"+cds}')
                continue
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
            dates = ['None'+str(idx) if e=='' else e for idx, e in enumerate(dates)]
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
            flag, valid_nb = self.DataValidation(values)
            if flag and valid_nb >= 10 :
                pass
            else :
                self.logger.error(f'Not enough *coinfo_cp* about this company -> {self.comps[idx]+"-"+cds}')
                continue
            df = pd.DataFrame(data=values, columns=dates, index=cols)
            self.EmptyChecker(df, cds)
            #last_info_date = self.DataExistChecker(df)
            df.to_hdf(f'./FullCache/Quarter/fs_quarter_{cds}_{self.update_date}.h5',key=cds,mode='w')
        
        self.logger.info(f'Cache generation is well done. : {self.update_date}')
        self.logger.info('=======================================================================')        
        return "Cache generation is well done." 

if __name__ == '__main__':
    print("Starting BS Crawler...")
    argument = sys.argv
    del argument[0]
    print("Mode you requested : {}".format(argument[0]))
    bs_crawler = BSCrawler(argument[0])
    bs_crawler.Crawler()
    bs_crawler.UpdateDataInfo('Annual')
    bs_crawler.UpdateDataInfo('Quarter')