import logging, os, pickle
import requests, glob
from datetime import datetime
from bs4 import BeautifulSoup
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
from matplotlib.pyplot import cm
import numpy as np
import scipy.stats as stats
from scipy import stats
import pyautogui

def NaverFinance(code='005930'):
    headers = {"User-Agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.93 Safari/537.36"}
    url = 'https://finance.naver.com/item/board.nhn?code={}&page={}'
    content_url = 'https://finance.naver.com/item/board_read.nhn?code={}&nid={}&st=&sw=&page={}'
    total = pd.DataFrame(columns=['DATE','TITLE','WRITER','FREQ','AGREE','DISAGREE','CONTENT'])
    for p in range(1, 10):
        res = requests.get(url.format(code,p),headers=headers)
        bs = BeautifulSoup(res.text,'html')
        tb = bs.find_all('table', attrs={'class':'type2'})[0]
        tb = tb.find_all('tr')
        for i in range(len(tb)):
            try :
                row = tb[i].find_all('td')
            except :
                raise ValueError('NA')
            if len(row)==6 :
                tmp = pd.DataFrame(columns=['DATE','TITLE','WRITER','FREQ','AGREE','DISAGREE','CONTENT'])
                tmp.loc[0,'DATE'] = row[0].text
                tmp.loc[0,'TITLE'] = row[1].text.replace('\n','').replace('\t','')
                tmp.loc[0,'WRITER'] = row[2].text.replace('\n','').replace('\t','')
                tmp.loc[0,'FREQ'] = row[3].text
                tmp.loc[0,'AGREE'] = row[4].text
                tmp.loc[0,'DISAGREE'] = row[5].text
                
                cnt_id = str(tb[i].find_all('a')[0]).split('nid=')[1].split('&')[0]
                res2 = requests.get(content_url.format(code,cnt_id,p), headers=headers)
                bs2 = BeautifulSoup(res2.text,'html')
                content = bs2.find_all('div',attrs={'id':'body'})[0].text.replace('\n','').replace('\t','').replace('\r','')
                tmp.loc[0,'CONTENT'] = content
                
                total = pd.concat([total, tmp])
    return total.reset_index(drop=True)

