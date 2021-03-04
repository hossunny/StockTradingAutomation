from datetime import timedelta, date

def trading_days(start='2009-01-01', end='2021-12-31', mode='r'):
    if mode == 'w':
        files = glob.glob("./FullCache_bk/DateParser/*")
        holidays = []
        for fp in files :
            tmp_holidays = list(pd.read_excel(fp).loc[:,'일자 및 요일'].values)
            tmp_holidays = [dt[:10] for dt in tmp_holidays]
            holidays += tmp_holidays
        def daterange(date1, date2):
            for n in range(int ((date2-date1).days)+1):
                yield date1 + timedelta(n)
        start_dt = date(int(start[:4]),int(start[5:7]),int(start[8:10]))
        end_dt = date(int(end[:4]),int(end[5:7]),int(end[8:10]))
        total_dates = []
        weekdays = [5,6]
        for dt in daterange(start_dt, end_dt):
            if dt.weekday() not in weekdays:
                if dt.strftime("%Y-%m-%d") not in holidays:
                    total_dates.append(dt.strftime("%Y-%m-%d"))
        with open("./TradingDates.pickle","wb") as fw :
            pickle.dump(total_dates, fw)
    elif mode=='r' :
        with open("./TradingDates.pickle","rb") as fr :
            total_dates = pickle.load(fr)
    else :
        raise ValueError("Which mode do you want?")
    return total_dates

def MissingChecker(td_dates):
    miss_dts = []
    allfiles = glob.glob("./FullCache_bk/marcap/*.csv")
    for fp in allfiles:
        print(fp)
        year = fp.split('-')[-1].split('.')[0]
        flag_start = year+'-01-01'
        flag_end = year+'-12-31'
        sub_dates = []
        for e in td_dates :
            if e >= flag_start and e <= flag_end :
                sub_dates.append(e)
        if '2020-10-15' in fp :
            break
        tmp_dates = sorted(list(set(pd.read_csv(fp)['Date'].values)))
        for dt in sub_dates :
            if dt not in tmp_dates :
                miss_dts.append(dt)

    """2019-02-19 & 2019-0531 are excluded!!!!!!!!!!!"""
    return miss_dts
