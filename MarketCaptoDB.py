import logging, os, pickle
import requests, glob
from datetime import datetime
from bs4 import BeautifulSoup
from datetime import timedelta, date
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
import pyautogui




def MarcapExtract(codes, td_days, conn):
    errs=[]
    query=[]
    cursor = conn.cursor()
    #year_ls = [str(2010+i) for i in range(11)]
    year_ls = [str(2010)]
    file_path = "./FullCache/marcap/marcap-{}.csv"
    for year in year_ls :
        fpath = file_path.format(year)
        tmp = pd.read_csv(fpath)
        tmp_codes = list(tmp.Code.values)
        #tmp_codes=['005930']
        
        """Trading date for quarter/annual last day"""
        Q1 = ''#year+'-'+'03-31'
        Q2 = ''#year+'-'+'06-31'
        Q3 = ''#year+'-'+'09-31'
        Y = ''#year+'-'+'12-31'
        for td in td_days :
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
                    #raise ValueError("What is this? -> {}".format(td))
        #print(Q1, Q2, Q3, Y)
        """Value Extract"""        
        for cd in tmp_codes :
            if cd not in codes :
                continue
            else :
                for ith, unit in enumerate([Q1,Q2,Q3,Y]):
                    #print(unit)
                    if ith != 3 :
                        sub = tmp[(tmp.Code==cd)&(tmp.Date==unit)]
                        if len(sub) == 0 :
                            errs.append(cd+'|'+unit)
                        else :
                            for idx, row in sub.iterrows():
                                #print(row)
                                #print(f"INSERT INTO finance_info VALUES('{cd}','{unit[:7]}','시가총액','Q',{row.Marcap}")
                                """print(f"INSERT INTO finance_info VALUES('{cd}','{unit[:7]}','시가총액','Q',{row.Marcap}")
                                print(f"INSERT INTO finance_info VALUES('{cd}','{unit[:7]}','시가총액비중(%)','Q',{row.MarcapRatio}")
                                print(f"INSERT INTO finance_info VALUES('{cd}','{unit[:7]}','상장주식수','Q',{row.Stocks}")
                                query.append(f"INSERT INTO finance_info_copy VALUES('{cd}','{unit[:7]}','시가총액','Q',{row.Marcap})")
                                query.append(f"INSERT INTO finance_info_copy VALUES('{cd}','{unit[:7]}','시가총액비중(%)','Q',{row.MarcapRatio})")
                                query.append(f"INSERT INTO finance_info_copy VALUES('{cd}','{unit[:7]}','상장주식수','Q',{row.Stocks})")"""
                                
                                if len(pd.read_sql(f"SELECT * FROM finance_info_copy where code='{cd}' and date='{unit[:7]}' and item='시가총액' and type='Q'",conn)) == 0:
                                    cursor.execute(f"INSERT INTO finance_info_copy VALUES('{cd}','{unit[:7]}','시가총액','Q',{row.Marcap})")
                                else :
                                    errs.append(cd+'|'+unit)
                                if len(pd.read_sql(f"SELECT * FROM finance_info_copy where code='{cd}' and date='{unit[:7]}' and item='시가총액비중(%)' and type='Q'",conn)) == 0:
                                    cursor.execute(f"INSERT INTO finance_info_copy VALUES('{cd}','{unit[:7]}','시가총액비중(%)','Q',{row.MarcapRatio})")
                                else :
                                    errs.append(cd+'|'+unit)
                                if len(pd.read_sql(f"SELECT * FROM finance_info_copy where code='{cd}' and date='{unit[:7]}' and item='상장주식수' and type='Q'",conn)) == 0:
                                    cursor.execute(f"INSERT INTO finance_info_copy VALUES('{cd}','{unit[:7]}','상장주식수','Q',{row.Stocks})")
                                else :
                                    errs.append(cd+'|'+unit)
                                conn.commit()
                    else :
                        sub = tmp[(tmp.Code==cd)&(tmp.Date==unit)]
                        if len(sub) == 0 :
                            errs.append(cd+'|'+unit)
                        else :
                            for idx, row in sub.iterrows():
                                """#print(f"INSERT INTO finance_info VALUES('{cd}','{unit[:7]}','시가총액','Y',{row.Marcap}")
                                query.append(f"INSERT INTO finance_info_copy VALUES('{cd}','{unit[:7]}','시가총액','Y',{row.Marcap})")
                                query.append(f"INSERT INTO finance_info_copy VALUES('{cd}','{unit[:7]}','시가총액비중(%)','Y',{row.MarcapRatio})")
                                query.append(f"INSERT INTO finance_info_copy VALUES('{cd}','{unit[:7]}','상장주식수','Y',{row.Stocks})")
                                print(f"INSERT INTO finance_info VALUES('{cd}','{unit[:7]}','시가총액','Y',{row.Marcap}")
                                print(f"INSERT INTO finance_info VALUES('{cd}','{unit[:7]}','시가총액비중(%)','Y',{row.MarcapRatio}")
                                print(f"INSERT INTO finance_info VALUES('{cd}','{unit[:7]}','상장주식수','Y',{row.Stocks}")"""
                                
                                if len(pd.read_sql(f"SELECT * FROM finance_info_copy where code='{cd}' and date='{unit[:7]}' and item='시가총액' and type='Y'",conn)) == 0:
                                    cursor.execute(f"INSERT INTO finance_info_copy VALUES('{cd}','{unit[:7]}','시가총액','Y',{row.Marcap})")
                                else :
                                    errs.append(cd+'|'+unit)
                                if len(pd.read_sql(f"SELECT * FROM finance_info_copy where code='{cd}' and date='{unit[:7]}' and item='시가총액비중(%)' and type='Y'",conn)) == 0:    
                                    cursor.execute(f"INSERT INTO finance_info_copy VALUES('{cd}','{unit[:7]}','시가총액비중(%)','Y',{row.MarcapRatio})")
                                else :
                                    errs.append(cd+'|'+unit)
                                if len(pd.read_sql(f"SELECT * FROM finance_info_copy where code='{cd}' and date='{unit[:7]}' and item='상장주식수' and type='Y'",conn)) == 0:
                                    cursor.execute(f"INSERT INTO finance_info_copy VALUES('{cd}','{unit[:7]}','상장주식수','Y',{row.Stocks})")
                                else :
                                    errs.append(cd+'|'+unit)
                                conn.commit()
        #conn.commit()
    return errs#, query