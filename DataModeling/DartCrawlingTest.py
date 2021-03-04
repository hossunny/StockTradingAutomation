import pandas as pd
import numpy as np
import glob
import dart-fss
import dart
import OpenDartReader

def finance(company, From, To) : 

    ### Dictionary
    df_info = {}
     
    for year in range(From, To+1) : 

        ### 재무제표
        fs = dart.finstate(company, year, reprt_code='11011') # 재무제표(전체)
        """
        reprt_code : '11013' = 1분기보고서, '11012' = 반기보고서, '11014' = 3분기보고서, '11011' = 사업보고서
        """
        fs_is = fs[fs['sj_div'].str.contains('IS')] # 재무제표(전체) > 손익계산서
        fs_bs = fs[fs['sj_div'].str.contains('BS')] # 재무제표(전체) > 재무제표
        fs_equity = fs_bs[fs_bs['account_nm'].str.contains('자본총계')] # 재무제표(전체) > 재무제표 > 자본총계
        fs_assets = fs_bs[fs_bs['account_nm'].str.contains('자산총계')] # 재무제표(전체) > 재무제표 > 자산총계
        fs_revenue = fs_is[fs_is['account_nm'].str.contains('매출액')] # 재무제표(전체) > 손익계산서 > 매출액
        fs_income = fs_is[fs_is['account_nm'].str.contains('영업이익')] # 재무제표(전체) > 손익계산서 > 영업이익
        fs_profit = fs_is[fs_is['account_nm'].str.contains('당기순이익')] # 재무제표(전체) > 손익계산서 > 당기순이익

        ### 재무제표 전체(재무상태표, 손익계산서, 자본변동표, 현금흐름표 등)
        fs_all = dart.finstate_all(company, year) # 재무제표(전체)
        fs_cf = fs_all[fs_all['sj_div'].str.contains('CF')] # 재무제표(전체) > 현금흐름표
        fs_OCF = fs_cf[fs_cf['account_id'].str.contains('CashFlowsFromUsedInOperatingActivities')] # 재무제표(전체) > 현금흐름표 > 영업활동현금흐름
        fs_aucqusition = fs_cf[fs_cf['account_nm'].str.contains('취득')] # 재무제표(전체) > 현금흐름표 >  '취득'
        fs_CAPEX_P = fs_aucqusition[fs_aucqusition['account_nm'].str.contains('유형자산')] # 재무제표(전체) > 현금흐름표 >  유형자산의취득
        fs_CAPEX_I = fs_aucqusition[fs_aucqusition['account_nm'].str.contains('무형자산')] # 재무제표(전체) > 현금흐름표 >  무형자산의취득

        ### 배당에 관한 사항
        dv = dart.report(company, '배당', year) # 배당에 관한 사항
        dv_dps = dv[ dv['se'] == '주당 현금배당금(원)' ] # 배당에 관한 사항 > 주당 배당금
        dv_eps = dv[ dv['se'].str.contains('주당순이익') ] # 배당에 관한 사항 > 주당 순이익
        dv_yield = dv[ dv['se'].str.contains('현금배당수익률') ] # 배당에 관한 사항 > 배당수익률
        dv_TD = dv[ dv['se'].str.contains('현금배당금총액') ] # 배당에 관한 사항 > 배당금 총액

   
        equity_this = int(fs_equity[['thstrm_amount']].iloc[0,0].replace(',', '').strip()) # 당해년도 자본총계
        equity_last = int(fs_equity[['frmtrm_amount']].iloc[0,0].replace(',', '').strip()) # 직전년도 자본총계
        equity = (equity_this + equity_last ) / 2 # 평균자본총계 
        assets_this = int(fs_assets[['thstrm_amount']].iloc[0,0].replace(',', '').strip()) # 당해년도 자본총계
        assets_last = int(fs_assets[['frmtrm_amount']].iloc[0,0].replace(',', '').strip()) # 직전년도 자본총계
        assets = (assets_this + assets_last ) / 2 # 평균자산총계
        revenue = int(fs_revenue[['thstrm_amount']].iloc[0,0].replace(',', '')) # 매출액
        income = int(fs_income[['thstrm_amount']].iloc[0,0].replace(',', '')) # 영업이익
        profit = int(fs_profit[['thstrm_amount']].iloc[0,0].replace(',', '')) # 순이익
        DPS = int( dv_dps[ ['thstrm'] ].iloc[0, 0].replace(',', '').strip() ) # 주당 배당금
        EPS = int( dv_eps[ ['thstrm'] ].iloc[0, 0].replace(',', '').strip() ) # 주당 순이익
        Yield = int( dv_yield[ ['thstrm'] ].iloc[0, 0].replace('.', '').strip() ) / 10000 # 배당수익률
        TD = int( dv_TD[ ['thstrm'] ].iloc[0, 0].replace(',', '').strip() ) # 배당금 총액
        OCF = int(fs_OCF[['thstrm_amount']].iloc[0,0]) # 당해년도
        CAPEX_P = int(fs_CAPEX_P[['thstrm_amount']].iloc[0,0]) # 유형자산의취득
        CAPEX_I = int(fs_CAPEX_I[['thstrm_amount']].iloc[0,0]) #  무형자산의취득
        CAPEX = CAPEX_P + CAPEX_I
        ROE = profit / equity # 자기자본수익률
        ROA = profit / assets # 자산수익률
        FCF = OCF - CAPEX # 잉여현금흐름
        OPM = income / revenue # 영업이익률
        NPM = profit / revenue # 순이익률
        PR = DPS / EPS # 배당성향
        
        ### dictionary에 담기
        df_info[str(year)] = \
        f'{revenue:,}'[:-8], \
        f'{income:,}'[:-8], \
        f'{profit:,}'[:-8], \
        f'{OCF:,}'[:-8], \
        f'{FCF:,}'[:-8], \
        f'{TD:,}', \
        f'{OPM:.1%}', \
        f'{NPM:.1%}', \
        f'{ROE:.1%}', \
        f'{ROA:.1%}', \
        f'{EPS:,}', \
        f'{DPS:,}', \
        f'{PR:.1%}', \
        f'{Yield:.1%}', \
        # f'{value:,}'[:-8] : ###,, / f'{value:.1%}' : #.#%
        
    ### 출력 ###
    df = pd.DataFrame(data = df_info, \
                      index = ['Revenue', 'Operating Profit', 'Net Profit', 
                               'Operating Cash Flow', 'Free Cash Flow', 'Total Dividend',
                               'Operating Profit Margin', 'Net Profit Margin', 
                               'Return on Equity', 'Return on Assets', 
                              'EPS', 'DPS', 'Payout Ratio', 'Yield'])

    company_nm = dart.company(company)['stock_name'] # 종목명
    company_nm_eng = dart.company(company)['corp_name_eng'] # 회사 영문명
    company_stock_code = dart.company(company)['stock_code'] # 종목코드    
    print(company_nm, company_nm_eng, company_stock_code, '(단위 : %, 백만원, 원)')
    return df


def finance_1Q(company, year) : 

    ### Dictionary
    df_info = {}
    df_info2 = {}

    ### 재무제표
    fs = dart.finstate(company, year, reprt_code='11013') # 재무제표(전체)
    """
    reprt_code : '11013' = 1분기보고서, '11012' = 반기보고서, '11014' = 3분기보고서, '11011' = 사업보고서
    """
    fs_cfs = fs[fs['fs_div'].str.contains('CFS')] # 재무제표(전체) > 연결
    fs_is = fs_cfs[fs_cfs['sj_div'].str.contains('IS')] # 재무제표(전체) > 손익계산서
    fs_bs = fs_cfs[fs_cfs['sj_div'].str.contains('BS')] # 재무제표(전체) > 재무제표
    fs_equity = fs_bs[fs_bs['account_nm'].str.contains('자본총계')] # 재무제표(전체) > 재무제표 > 자본총계
    fs_assets = fs_bs[fs_bs['account_nm'].str.contains('자산총계')] # 재무제표(전체) > 재무제표 > 자산총계
    fs_debt = fs_bs[fs_bs['account_nm'].str.contains('부채총계')] # 재무제표(전체) > 연결 > 재무상태표 > 부채총계
    fs_current_liabilities = fs_bs[fs_bs['account_nm']==('유동부채')]
    fs_current_assets = fs_bs[fs_bs['account_nm']==('유동자산')]
    
    
    ### 재무제표 전체(재무상태표, 손익계산서, 자본변동표, 현금흐름표 등)
    fs_all = dart.finstate_all(company, year, reprt_code = '11013') # 1분기 보고서 재무제표(전체)
    fs_revenue = fs_all[fs_all['account_nm'].str.contains('매출액|영업수익')] # 재무제표(전체) > 손익계산서 > 매출액
    fs_income = fs_all[fs_all['account_id']=='dart_OperatingIncomeLoss'] # 재무제표(전체) > 손익계산서 > 영업이익
    fs_profit = fs_all[fs_all['account_id']=='ifrs-full_ProfitLoss'] # 재무제표(전체) > 손익계산서 > 당기순이익
    fs_CostOfSales = fs_all[fs_all['account_id']=='ifrs-full_CostOfSales']
    fs_CostOfMgnt = fs_all[fs_all['account_nm'].str.contains('판매비와관리비|영업비용')]
    fs_IncomeOfFin = fs_all[fs_all['account_nm'].str.contains('금융수익|금융이익')]
    fs_CostOfFin = fs_all[fs_all['account_nm'].str.contains('금융비용|금융손실|금융원가')]
    fs_IncomeOfEtc = fs_all[fs_all['account_nm'].str.contains('기타수익|기타이익|기타영업외수익')]
    fs_CostOfEtc = fs_all[fs_all['account_nm'].str.contains('기타비용|기타손실|기타원가|기타영업외비용')]
    
    fs_cf = fs_all[fs_all['sj_div'].str.contains('CF')] # 재무제표(전체) > 현금흐름표
    fs_OCF = fs_cf[fs_cf['account_id'].str.contains('CashFlowsFromUsedInOperatingActivities')] # 재무제표(전체) > 현금흐름표 > 영업활동현금흐름
    fs_aucqusition = fs_cf[fs_cf['account_nm'].str.contains('취득')] # 재무제표(전체) > 현금흐름표 >  '취득'
    fs_CAPEX_P = fs_aucqusition[fs_aucqusition['account_nm'].str.contains('유형자산')] # 재무제표(전체) > 현금흐름표 >  유형자산의취득
    fs_CAPEX_I = fs_aucqusition[fs_aucqusition['account_nm'].str.contains('무형자산')] # 재무제표(전체) > 현금흐름표 >  무형자산의취득
    
    for i in (0, 1):
        equity = int(fs_equity[['thstrm_amount', 'frmtrm_amount']].iloc[0,i].replace(',','').strip())
        assets = int(fs_assets[['thstrm_amount', 'frmtrm_amount']].iloc[0,i].replace(',', '').strip())
        debt = int(fs_debt[['thstrm_amount','frmtrm_amount']].iloc[0,i].replace(',','').strip())
        current_liabilities = int(fs_current_liabilities[['thstrm_amount', 'frmtrm_amount']].iloc[0,i].replace(',','').strip())
        current_assets = int(fs_current_assets[['thstrm_amount', 'frmtrm_amount']].iloc[0,i].replace(',','').strip())
        revenue = int(fs_revenue[['thstrm_amount','frmtrm_q_amount']].iloc[0,i].replace(',', '')) # 매출액
        income = int(fs_income[['thstrm_amount','frmtrm_q_amount']].iloc[0,i].replace(',', '')) # 영업이익
        profit = int(fs_profit[['thstrm_amount','frmtrm_q_amount']].iloc[0,i].replace(',', '')) # 순이익
        IncomeOfFin = abs(int(fs_IncomeOfFin[['thstrm_amount','frmtrm_q_amount']].iloc[0,i].replace(',',''))) \
                    - abs(int(fs_CostOfFin[['thstrm_amount','frmtrm_q_amount']].iloc[0,i].replace(',',''))) # 금융손익
        IncomeOfEtc = abs(int(fs_IncomeOfEtc[['thstrm_amount','frmtrm_q_amount']].iloc[0,i].replace(',',''))) \
                    - abs(int(fs_CostOfEtc[['thstrm_amount','frmtrm_q_amount']].iloc[0,i].replace(',',''))) # 금융손익
        
        
        
        try :
            CostOfSales = abs(int(fs_CostOfSales[['thstrm_amount','frmtrm_q_amount']].iloc[0,i].replace(',','')))
            CostOfMgnt = abs(int(fs_CostOfMgnt[['thstrm_amount','frmtrm_q_amount']].iloc[0,i].replace(',','')))
        except :
            CostOfSales = 0
            CostOfMgnt = abs(int(fs_CostOfMgnt[['thstrm_amount','frmtrm_q_amount']].iloc[0,i].replace(',','')))
        
        OCF = int(fs_OCF[['thstrm_amount','frmtrm_q_amount']].iloc[0,i]) # 당해년도
        CAPEX_P = int(fs_CAPEX_P[['thstrm_amount','frmtrm_q_amount']].iloc[0,i]) # 유형자산의취득
        CAPEX_I = int(fs_CAPEX_I[['thstrm_amount','frmtrm_q_amount']].iloc[0,i]) #  무형자산의취득
        CAPEX = CAPEX_P + CAPEX_I
        FCF = OCF - CAPEX # 잉여현금흐름
        OPM = income / revenue # 영업이익률
        NPM = profit / revenue # 순이익률
        DR = debt / equity # 부채비율
        CR = current_assets / current_liabilities # Current Ratio, 유동비율
        ROE = profit / equity # 자기자본수익률
        ROA = profit / assets # 자산수익률
        #EPS = int( dv_eps[ ['thstrm'] ].iloc[0, 0].replace(',', '').strip() )
        
        
        ### dictionary에 담기
        df_info[str(year-i)+' 1Q'] = \
        f'{revenue:,}'[:-8], \
        f'{CostOfSales:,}'[:-8], \
        f'{CostOfMgnt:,}'[:-8], \
        f'{income:,}'[:-8], \
        f'{IncomeOfFin:,}'[:-8], \
        f'{IncomeOfEtc:,}'[:-8], \
        f'{profit:,}'[:-8], \
        f'{OCF:,}'[:-8], \
        f'{FCF:,}'[:-8], \
        f'{OPM:.1%}', \
        f'{NPM:.1%}', \
        f'{DR:,}', \
        f'{CR:.1%}', \
        f'{ROE:.1%}', \
        f'{ROA:.1%}', \
        # f'{value:,}'[:-8] : ###,, / f'{value:.1%}' : #.#%
        
        df_info2[str(year-i)+' 1Q'] = revenue, CostOfSales, CostOfMgnt, income, IncomeOfFin, IncomeOfEtc, profit, OCF, FCF, OPM, NPM, DR, CR, ROE, ROA # YoY 계산 용도
        
    ### 출력 ###
    df = pd.DataFrame(data = df_info, \
                      index = ['Revenue', '(-) Cost Of Revenue', '(-) Cost Of Mgnt', 'Operating Profit',
                               '(+) Financial Profit','(+) Etc. Profit','Net Profit', 
                               'Operating Cash Flow', 'Free Cash Flow',
                               'Operating Profit Margin', 'Net Profit Margin', 
                                'Debt Ratio', 'Current Ratio',
                              'Return on Equity', 'Return on Assets'])

    df2 = pd.DataFrame(data = df_info2, \
                      index = ['Revenue', '(-) Cost Of Revenue', '(-) Cost Of Mgnt','Operating Profit', 
                               '(+) Financial Profit','(+) Etc. Profit','Net Profit',
                              'Operating Cash Flow', 'Free Cash Flow',
                               'Operating Profit Margin', 'Net Profit Margin', 
                                'Debt Ratio', 'Current Ratio',
                              'Return on Equity', 'Return on Assets'])
    
    df['YoY'] = (df2[str(year)+' 1Q'] - df2[str(year-1) +' 1Q']) / abs(df2[str(year-1)+' 1Q']) # YoY
    df['YoY'] = df['YoY'].apply(lambda x: '{:.1%}'.format(float(x))) # column data formatting
    
    
    company_nm = dart.company(company)['stock_name'] # 종목명
    company_nm_eng = dart.company(company)['corp_name_eng'] # 회사 영문명
    company_stock_code = dart.company(company)['stock_code'] # 종목코드    
    print(company_nm, company_nm_eng, company_stock_code, '(단위 : %, 백만원, 원)')
    
    def color_negative_red(val):
        """
        Takes a scalar and returns a string with
        the css property 'color : red' for negative
        strings, black otherwise.
        """
        color = 'red' if val[0:1] == '-' else 'white'
        return 'color: %s' % color
    
    df = df.style.applymap(color_negative_red) # start with '-' then color Red
    df = df.set_table_styles([dict(selector='th', props=[('text-align','left')])])
    
    
    return df

def finance_QQ_v2(company, From, To, byUnit='1Q') : 

    ### Parsing byUnit
    if byUnit == '1Q':
        RptCode = '11013'
    elif byUnit == '2Q':
        RptCode = '11012'
    elif byUnit == '3Q':
        RptCode = '11014'
    else :
        raise ValueError("Invalid Unit type !!!")
    

    total_df = pd.DataFrame()
    for year in range(From, To+1):
        
        ### Dictionary
        df_info = {}

        ### 재무제표
        fs = dart.finstate(company, year, reprt_code=RptCode) # 재무제표(전체)
        """
        reprt_code : '11013' = 1분기보고서, '11012' = 반기보고서, '11014' = 3분기보고서, '11011' = 사업보고서
        """
        fs_cfs = fs[fs['fs_div'].str.contains('CFS')] # 재무제표(전체) > 연결
        fs_is = fs_cfs[fs_cfs['sj_div'].str.contains('IS')] # 재무제표(전체) > 손익계산서
        fs_bs = fs_cfs[fs_cfs['sj_div'].str.contains('BS')] # 재무제표(전체) > 재무제표
        fs_equity = fs_bs[fs_bs['account_nm'].str.contains('자본총계')] # 재무제표(전체) > 재무제표 > 자본총계
        fs_assets = fs_bs[fs_bs['account_nm'].str.contains('자산총계')] # 재무제표(전체) > 재무제표 > 자산총계
        fs_debt = fs_bs[fs_bs['account_nm'].str.contains('부채총계')] # 재무제표(전체) > 연결 > 재무상태표 > 부채총계
        fs_current_liabilities = fs_bs[fs_bs['account_nm']==('유동부채')]
        fs_current_assets = fs_bs[fs_bs['account_nm']==('유동자산')]


        ### 재무제표 전체(재무상태표, 손익계산서, 자본변동표, 현금흐름표 등)
        fs_all = dart.finstate_all(company, year, reprt_code = RptCode) # 1분기 보고서 재무제표(전체)
        fs_revenue = fs_all[fs_all['account_nm'].str.contains('매출액|영업수익')] # 재무제표(전체) > 손익계산서 > 매출액
        fs_income = fs_all[fs_all['account_id']=='dart_OperatingIncomeLoss'] # 재무제표(전체) > 손익계산서 > 영업이익
        fs_profit = fs_all[fs_all['account_id']=='ifrs-full_ProfitLoss'] # 재무제표(전체) > 손익계산서 > 당기순이익
        if len(fs_profit) == 0 :
            fs_profit = fs_all[fs_all['account_nm'].str.contains('당기순이익')]
            if len(fs_profit) == 0 :
                fs_profit = fs_all[fs_all['account_nm'].str.contains('분기순이익')]
                if len(fs_profit) == 0 :
                    raise ValueError('Profit(당기순이익) is not found.')
        
        fs_CostOfSales = fs_all[fs_all['account_id']=='ifrs-full_CostOfSales']
        if len(fs_CostOfSales) == 0 :
            fs_CostOfSales = fs_all[fs_all['account_nm'].str.contains('매출원가')]
            if len(fs_CostOfSales) == 0 :
                raise ValueError("CostOfSales(매출원가) is not found")
                
        fs_CostOfMgnt = fs_all[fs_all['account_nm'].str.contains('판매비와관리비|영업비용')]
        fs_IncomeOfFin = fs_all[fs_all['account_nm'].str.contains('금융수익|금융이익')]
        fs_CostOfFin = fs_all[fs_all['account_nm'].str.contains('금융비용|금융손실|금융원가')]
        fs_IncomeOfEtc = fs_all[fs_all['account_nm'].str.contains('기타수익|기타이익|기타영업외수익')]
        fs_CostOfEtc = fs_all[fs_all['account_nm'].str.contains('기타비용|기타손실|기타원가|기타영업외비용')]

        fs_cf = fs_all[fs_all['sj_div'].str.contains('CF')] # 재무제표(전체) > 현금흐름표
        fs_OCF = fs_cf[fs_cf['account_id'].str.contains('CashFlowsFromUsedInOperatingActivities')] # 재무제표(전체) > 현금흐름표 > 영업활동현금흐름
        fs_aucqusition = fs_cf[fs_cf['account_nm'].str.contains('취득')] # 재무제표(전체) > 현금흐름표 >  '취득'
        fs_CAPEX_P = fs_aucqusition[fs_aucqusition['account_nm'].str.contains('유형자산')] # 재무제표(전체) > 현금흐름표 >  유형자산의취득
        fs_CAPEX_I = fs_aucqusition[fs_aucqusition['account_nm'].str.contains('무형자산')] # 재무제표(전체) > 현금흐름표 >  무형자산의취득
        
        ### 배당 dv2 = dart.report('삼성전자','배당',2020,'11012')
        dv = dart.report(company, '배당', year, RptCode)
        dv_eps = dv[ dv['se'].str.contains('주당순이익') ]

        equity = int(fs_equity[['thstrm_amount']].iloc[0,0].replace(',','').strip())
        assets = int(fs_assets[['thstrm_amount']].iloc[0,0].replace(',', '').strip())
        debt = int(fs_debt[['thstrm_amount']].iloc[0,0].replace(',','').strip())
        current_liabilities = int(fs_current_liabilities[['thstrm_amount']].iloc[0,0].replace(',','').strip())
        current_assets = int(fs_current_assets[['thstrm_amount']].iloc[0,0].replace(',','').strip())
        revenue = int(fs_revenue[['thstrm_amount']].iloc[0,0].replace(',', '')) # 매출액
        income = int(fs_income[['thstrm_amount']].iloc[0,0].replace(',', '')) # 영업이익
        profit = int(fs_profit[['thstrm_amount']].iloc[0,0].replace(',', '')) # 순이익
        IncomeOfFin = abs(int(fs_IncomeOfFin[['thstrm_amount']].iloc[0,0].replace(',',''))) \
                    - abs(int(fs_CostOfFin[['thstrm_amount']].iloc[0,0].replace(',',''))) # 금융손익
        IncomeOfEtc = abs(int(fs_IncomeOfEtc[['thstrm_amount']].iloc[0,0].replace(',',''))) \
                    - abs(int(fs_CostOfEtc[['thstrm_amount']].iloc[0,0].replace(',',''))) # 금융손익

        try :
            CostOfSales = abs(int(fs_CostOfSales[['thstrm_amount']].iloc[0,0].replace(',','')))
            CostOfMgnt = abs(int(fs_CostOfMgnt[['thstrm_amount']].iloc[0,0].replace(',','')))
        except :
            CostOfSales = 0
            CostOfMgnt = abs(int(fs_CostOfMgnt[['thstrm_amount']].iloc[0,0].replace(',','')))

        OCF = int(fs_OCF[['thstrm_amount']].iloc[0,0]) # 당해년도
        CAPEX_P = int(fs_CAPEX_P[['thstrm_amount']].iloc[0,0]) # 유형자산의취득
        CAPEX_I = int(fs_CAPEX_I[['thstrm_amount']].iloc[0,0]) #  무형자산의취득
        CAPEX = CAPEX_P + CAPEX_I
        FCF = OCF - CAPEX # 잉여현금흐름
        OPM = income / revenue # 영업이익률
        NPM = profit / revenue # 순이익률
        DR = debt / equity # 부채비율
        CR = current_assets / current_liabilities # Current Ratio, 유동비율
        ROE = profit / equity # 자기자본수익률
        ROA = profit / assets # 자산수익률
        EPS = int( dv_eps[ ['thstrm'] ].iloc[0, 0].replace(',', '').strip() )


        ### dictionary에 담기
        df_info[str(year)+' 1Q'] = \
        f'{revenue:,}'[:-8], \
        f'{CostOfSales:,}'[:-8], \
        f'{CostOfMgnt:,}'[:-8], \
        f'{income:,}'[:-8], \
        f'{IncomeOfFin:,}'[:-8], \
        f'{IncomeOfEtc:,}'[:-8], \
        f'{profit:,}'[:-8], \
        f'{OCF:,}'[:-8], \
        f'{FCF:,}'[:-8], \
        f'{OPM:.1%}', \
        f'{NPM:.1%}', \
        f'{DR:,}', \
        f'{CR:.1%}', \
        f'{ROE:.1%}', \
        f'{ROA:.1%}', \
        f'{EPS:,}', \
        # f'{value:,}'[:-8] : ###,, / f'{value:.1%}' : #.#%

        ### 출력 ###
        df = pd.DataFrame(data = df_info, \
                          index = ['Revenue', '(-) Cost Of Revenue', '(-) Cost Of Mgnt', 'Operating Profit',
                                   '(+) Financial Profit','(+) Etc. Profit','Net Profit', 
                                   'Operating Cash Flow', 'Free Cash Flow',
                                   'Operating Profit Margin', 'Net Profit Margin', 
                                    'Debt Ratio', 'Current Ratio',
                                  'ROE-Return on Equity', 'ROA-Return on Assets', 'EPS-Earnings Per Share'])
        total_df = pd.concat([total_df, df],axis=1)

    company_nm = dart.company(company)['stock_name'] # 종목명
    company_nm_eng = dart.company(company)['corp_name_eng'] # 회사 영문명
    company_stock_code = dart.company(company)['stock_code'] # 종목코드    
    print(company_nm, company_nm_eng, company_stock_code, '(단위 : %, 백만원, 원)')
    
    return total_df


