import logging, os, pickle
import requests, glob
from datetime import datetime
from bs4 import BeautifulSoup
from datetime import datetime
import pandas as pd
import numpy as np
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

class Loader:
    def __init__(self, pwd):
        self.conn = pymysql.connect(host='localhost',user='root',
                                   password=pwd,db='INVESTAR',charset='utf8')
        with self.conn.cursor() as curs:
            sql_load = """
            SELECT CODE, COMPANY, SECTOR FROM COMPANY_INFO
            """
            curs.execute(sql_load)
            comps_ls = curs.fetchall()
            self.codes = [str(e[0]) for e in comps_ls]
            self.comps = [str(e[1]) for e in comps_ls]
            self.sectors = [str(e[2]) for e in comps_ls]
            
        self.conn.commit()
        self.DataPath = 'C:\\Users\\Bae Kyungmo\\OneDrive\\Desktop\\StockTraidingAutomation\\FullCache\\'
        
        with open(self.DataPath+"TradingDates.pickle", "rb") as fr:
            self.td_days = pickle.load(fr)
        self.items = ['매출액', '영업이익', '영업이익(발표기준)', '세전계속사업이익', '당기순이익', '당기순이익(지배)', '당기순이익(비지배)', '자산총계',
                     '부채총계', '자본총계', '자본총계(지배)', '자본총계(비지배)', '자본금', '영업활동현금흐름', '투자활동현금흐름', '재무활동현금흐름',
                     'CAPEX', 'FCF', '이자발생부채', '영업이익률', '순이익률', 'ROE(%)', 'ROA(%)', '부채비율', '자본유보율', 'EPS(원)', 'PER(배)', 'BPS(원)',
                     'PBR(배)', '현금DPS(원)', '현금배당수익률', '현금배당성향(%)', '발행주식수(보통주)','시가총액','상장주식수','시가총액비중(%)',
                     'PBR','PER','PCR','POR','PSR','ROE','ROA','EPS','BPS']
                
    def __del__(self):
        """Disconnecting MariaDB"""
        self.conn.close()

    def Delist(self):
        with open(self.DataPath+"Delist.pickle", "rb") as fr:
            self.delist = pickle.load(fr)
        return self.delist
        
    def CompanyInfo(self):
        self.company_info = pd.read_sql("select * from company_info",self.conn)
        return self.company_info
    
    def FindCodeByName(self, comp_nm):
        idx = self.comps.index(comp_nm)
        return self.codes[idx]
    
    def FindNameByCode(self, comp_code):
        idx = self.codes.index(comp_code)
        return self.comps[idx]
    
    def FindSectorByCode(self, comp_code):
        idx = self.codes.index(comp_code)
        return self.sectors[idx]

    def ShowItems(self):
        return self.items
    
    def GetTradingDays(self, start=None, end=None):
        if start == None and end==None :
            return self.td_days
        else :
            if start == None :
                start = '2009-01-01'
            if end == None :
                end = '2021-12-30'
            td_days=[]
            for dt in self.td_days:
                if dt >= start and dt <= end :
                    td_days.append(dt)
            return td_days
            
        return self.td_days
    
    def GetPricelv1(self, start, end, code_ls=None):
        if code_ls == None :
            code_ls = self.codes + ['005935','005385','066575']
#            code_ls += ['005935','005385','066575']
        start_year = start[:4]
        end_year = end[:4]
        if start_year == end_year:
            pr = pd.read_hdf(self.DataPath+"Price/price_{}.h5".format(start_year))
        else :
            pr = pd.DataFrame()
            for y in range(int(start_year), int(end_year)+1):
                tmp_pr = pd.read_hdf(self.DataPath+"Price/price_{}.h5".format(str(y)))
                pr = pd.concat([pr, tmp_pr])
        pr = pr[(pr.DATE>=start)&(pr.DATE<=end)&(pr.CODE.isin(code_ls))].sort_values(by=['DATE'])
        return pr
    
    def GetPricelv2(self, start, end, code_ls=None):
        if code_ls == None:
            code_ls = self.codes + ['005935','005385','066575']
#            code_ls += ['005935','005385','066575']
#         start_year = start[:4]
#         end_year = end[:4]
#         if start_year == end_year:
#             pr = pd.read_hdf("FullCache/Price/lv2_price_{}.h5".format(start_year))
#         else :
#             pr = pd.DataFrame()
#             for y in range(int(start_year), int(end_year)+1):
#                 tmp_pr = pd.read_hdf("FullCache/Price/lv2_price_{}.h5".format(str(y)))
#                 pr = pd.concat([pr, tmp_pr])
        pr = pd.read_hdf(self.DataPath+"Price/lv2_price_total.h5")
        for cd in code_ls : 
            if cd not in list(pr.columns):
                pr[cd] = np.nan
        pr = pr[(pr.index>=start)&(pr.index<=end)][code_ls].sort_index()
        return pr
    
    # def GetPricelv2(self, start, end, code_ls=None):
    #     if code_ls == None:
    #         code_ls = self.codes + ['005935','005385','066575']
    #     pr = pd.read_hdf(self.DataPath+"Price/lv2_price_total.h5")
    #     for cd in code_ls : 
    #         if cd not in list(pr.columns):
    #             pr[cd] = np.nan
    #         if cd in self.delist.keys():
    #             pr_tmp = pr[cd].copy()
    #             pr_tmp[pr_tmp.index.isin(self.delist[cd])] = np.nan
    #             pr[cd] = pr_tmp
    #     pr = pr[(pr.index>=start)&(pr.index<=end)][code_ls].sort_index()
    #     return pr

    def GetVolumelv2(self, start, end, code_ls=None):
        if code_ls == None:
            code_ls = self.codes + ['005935','005385','066575']
        volume = pd.read_hdf(self.DataPath+"VOLUME_lv2.h5")
        for cd in code_ls : 
            if cd not in list(volume.columns):
                volume[cd] = np.nan
        volume = volume[(volume.index>=start)&(volume.index<=end)][code_ls].sort_index()
        return volume

    def GetMarcaplv2(self, start, end, code_ls=None):
        if code_ls == None:
            code_ls = self.codes + ['005935','005385','066575']
        marcap = pd.read_hdf(self.DataPath+"marcap/lv2_marcap_total.h5")
        for cd in code_ls : 
            if cd not in list(marcap.columns):
                marcap[cd] = np.nan
        marcap = marcap[(marcap.index>=start)&(marcap.index<=end)][code_ls].sort_index()
        return marcap

    def GetKOSPI(self, start, end):
        kospi = pd.read_hdf(self.DataPath+"KOSPI_lv2.h5")
        kospi = kospi[(kospi.index>=start)&(kospi.index<=end)].sort_index()
        kospi.columns = ['KOSPI']
        return kospi
    
    def GetFunda(self, byDate='2019-12', code_ls=None, itm=None, level=1):
        funda = pd.read_hdf(self.DataPath+"/FUNDA_total.h5")
        if code_ls == None :
            code_ls = self.codes
            code_ls += ['005935','005385','066575']
        if itm == None :
            itm = self.items
        else:
            if type(itm) != list :
                raise ValueError("Please insert itm as list type.")
        if 'main' in itm:
            itm = ['PBR','PER','PCR','POR','PSR','ROE','ROA','EPS','BPS','시가총액']
        funda = funda[(funda.code.isin(code_ls))&(funda.date==byDate)&(funda.itm.isin(itm))].reset_index(drop=True)
        if level == 1:
            return funda
        elif level ==2 :
            lv2_funda = pd.DataFrame(index=code_ls, columns=itm)
            for idx, row in funda.iterrows():
                lv2_funda.loc[row.code, row.itm] = row.value
            return lv2_funda
            
if __name__ == '__main__':
    print("=== Insert MariaDB password ===")
    argument = sys.argv
    del argument[0]
    loader = Loader(argument[0])