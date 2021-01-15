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

class MarketCapCrawler():
    def __init__(self):
        self.driver_path = "C:/Users/Bae Kyungmo/OneDrive/Desktop/WC_basic/chromedriver.exe"
        self.url = "http://marketdata.krx.co.kr/mdi#document=040402"
        self.download_path = "C:/Users/Bae Kyungmo/Downloads/"
        self.save_path = "C:/Users/Bae Kyungmo/Downloads/"

        return True
    
    def GetMarcap(self, mode='daily'):
        errs = []
        with open("./TradingDates.pickle","rb") as fr:
            td_dates = pickle.load(fr)
        last_update_date = glob.glob("./FullCache/marcap/*.csv").split('marcap-')[-1].split('.csv')[0]
        if len(last_update_date) <= 4:
            raise ValueError("This date is not Y-M-D format.. Check this out. : ", last_update_date)
        browser = Chrome(self.driver_path)
        browser.maximize_window()
        browser.get(self.url)
        WebDriverWait(browser, 10).until(EC.element_to_be_clickable((By.XPATH,'//*[@id="schdate8f14e45fceea167a5a36dedd4bea2543"]')))
        browser.find_elements_by_xpath('//*[@id="schdate8f14e45fceea167a5a36dedd4bea2543"]')[0].click()
        for dt in date_ls:
            try:
                pyautogui.moveTo(617, 404)
                pyautogui.doubleClick()
                browser.find_elements_by_xpath('//*[@id="schdate8f14e45fceea167a5a36dedd4bea2543"]')[0].send_keys(dt[:4]+dt[5:7]+dt[8:])
                pyautogui.click(834, 455,duration=1.0)
                time.sleep(2)
                pyautogui.click(1266, 456,duration=1.0)
                try :
                    for _ in range(1,20):
                        time.sleep(1)
                        try :
                            os.rename(self.download_path + 'data.csv', self.save_path + 'marcap-{}.csv'.format(dt))
                            break
                        except :
                            pass
                except :
                    errs.append(dt)
                    raise ValueError("Renaming a file is failed...")
                time.sleep(1)
            except :
                errs.append(dt)
                raise ValueError("Downloading a file is failed...")
    print("MarketCap Crawling from KRX is successfully finished.")
    return errs

if __name__ == '__main__':
    print("Starting MarketCap Crawler...")
    MCCrawl = MarketCapCrawler()
    MCCrawl.GetMarcap()