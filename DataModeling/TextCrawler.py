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

def NaverFinance(code_ls=['005930']):
    headers = {"User-Agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.93 Safari/537.36"}
    errs = []
    for code in code_ls :
        try :
            url = 'https://finance.naver.com/item/board.nhn?code={}&page={}'
            content_url = 'https://finance.naver.com/item/board_read.nhn?code={}&nid={}&st=&sw=&page={}'
            total = pd.DataFrame(columns=['DATE','TITLE','WRITER','FREQ','AGREE','DISAGREE','CONTENT'])

            res_init = requests.get(url.format(code,1),headers=headers)
            bs_init = BeautifulSoup(res_init.text, 'html')
            last_page = str(bs_init.find_all('td',attrs={'class':'pgRR'})[0].find('a')).split('page=')[1].split('"')[0]
            print("LAST PAGE : {}".format(last_page))
            for p in range(1, 10):
                res = requests.get(url.format(code,p),headers=headers)
                bs = BeautifulSoup(res.text,'html')
                tb = bs.find_all('table', attrs={'class':'type2'})[0]
                tb = tb.find_all('tr')
                for i in range(len(tb)):
                    try :
                        row = tb[i].find_all('td')
                        if len(row)==6 :
                            tmp = pd.DataFrame(columns=['DATE','TITLE','WRITER','FREQ','AGREE','DISAGREE','CONTENT'])
                            tmp.loc[0,'DATE'] = row[0].text
                            tmp.loc[0,'TITLE'] = row[1].text.replace('\n','').replace('\t','')
                            tmp.loc[0,'WRITER'] = row[2].text.replace('\n','').replace('\t','')
                            tmp.loc[0,'FREQ'] = int(row[3].text)
                            tmp.loc[0,'AGREE'] = int(row[4].text)
                            tmp.loc[0,'DISAGREE'] = int(row[5].text)

                            cnt_id = str(tb[i].find_all('a')[0]).split('nid=')[1].split('&')[0]
                            res2 = requests.get(content_url.format(code,cnt_id,p), headers=headers)
                            bs2 = BeautifulSoup(res2.text,'html')
                            content = bs2.find_all('div',attrs={'id':'body'})[0].text#.replace('\n','').replace('\t','').replace('\r','')
                            tmp.loc[0,'CONTENT'] = content
                            total = pd.concat([total, tmp])                        
                    except Exception as e:
                        print(e)
                        errs.append(code+'|'+str(p))
                        continue
            total = ReplyCount(total)
            total = total.drop_duplicates(['DATE','TITLE','WRITER']).reset_index(drop=True)
            total.to_csv("./data/nf_{}.csv".format(code),index=False)
        except Exception as e:
            print(e)
            errs.append(code+'|'+'NA')
    return total, errs

def ReplyCount(tdf):
    df = tdf.copy()
    cnt_ls = []
    for idx, row in df.iterrows():
        title = row.TITLE
        if title[-1] == ']':
            rep_cnt = title.split('[')[-1].split(']')[0]
            try :
                rep_cnt = int(rep_cnt)
            except :
                rep_cnt = -1
        else :
            rep_cnt = 0
        cnt_ls.append(rep_cnt)
    assert len(cnt_ls) == len(df)
    df['REPLY_CNT'] = cnt_ls
    return df

def NewsCrawler_vMarket(dates = ['2021-05-13']):
    """ Crawling NEWS about Market """
    headers = {"User-Agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.93 Safari/537.36"}
    url_init = "https://finance.naver.com/news/news_list.nhn?mode=LSS3D&section_id=101&section_id2=258&section_id3=401&date={}&page={}"
    total = pd.DataFrame(columns = ['DATE','TITLE','PRESS','CONTENT'])
    for dt in dates:
        res_init = requests.get(url_init.format(dt.replace('-',''),1), headers=headers)
        bs_init = BeautifulSoup(res_init.text, 'html')
        last_page = str(bs_init.find_all('td',attrs={'class':'pgRR'})[0]).split('page=')[-1].split('"')[0]
        print('LAST PAGE : {}'.format(last_page))
        for p in range(1, int(last_page)+1):
            print("Crawling NEWS at {} in page-{}".format(dt,p))
            res = requests.get(url_init.format(dt.replace('-',''),p), headers=headers)
            bs = BeautifulSoup(res.text, 'html')
            news = bs.find_all('dd',attrs={'class':'articleSubject'}) + bs.find_all('dt',attrs={'class':'articleSubject'})
            news_press = bs.find_all('span',attrs={'class':'press'})
            for i in range(len(news)):
                title = news[i].text.replace('\n','')
                press = news_press[i].text
                link = news[i].find('a')
                link = ('https://finance.naver.com' + str(link).split('href=')[-1].split('"')[1]).replace('amp;','')
                res_news = requests.get(link, headers=headers)
                bs_news = BeautifulSoup(res_news.text, 'html')
                content = bs_news.find_all('div',attrs={'id':'content','class':'articleCont'})[0].text
                
                tmp = pd.DataFrame(columns = ['DATE','TITLE','PRESS','CONTENT'])
                tmp.loc[0,'DATE'] = dt.replace('-','')
                tmp.loc[0,'TITLE'] = title
                tmp.loc[0,'PRESS'] = press
                tmp.loc[0,'CONTENT'] = content
                total = pd.concat([total,tmp])
                
    total.reset_index(drop=True, inplace=True)
    #total.drop_duplicates(inplace=True)
    return total