class BSCrawler:
    def __init__(self, argv):
        """Naver Finance : Financial Summary Crawler"""
        self.driver_path = "C:/Users/Bae Kyungmo/OneDrive/Desktop/WC_basic/chromedriver.exe"
        self.conn = pymysql.connect(host='localhost',user='root',
                                   password='tlqkfdk2',db='INVESTAR',charset='utf8')
        with self.conn.cursor() as curs:
            #sql = """
            #CREATE TABLE IF NOT EXISTS company_info (
            #    code VARCHAR(20),
            #    company VARCHAR(40),
            #    last_update DATE,
            #    PRIMARY KEY (code))
            #"""
            #curs.execute(sql)
            
            sql_load = """
            SELECT CODE, COMPANY FROM COMPANY_INFO
            """
            curs.execute(sql_load)
            comps_ls = curs.fetchall()
            self.codes = [str(e[0]) for e in comps_ls]
            self.comps = [str(e[1]) for e in comps_ls]
            
        self.conn.commit()
        
        self.url = 'https://finance.naver.com/item/coinfo.nhn?code={}&target=finsum_more'
        
    def __del__(self):
        """Disconnecting MariaDB"""
        self.conn.close()
        
    def crawler(self):
        browser = Chrome(self.driver_path)
        t=[]
        for cds in self.codes[:5]:
            url = self.url.format(cds)
            requests.get(url).raise_for_status()
            browser.get(url)
            browser.switch_to_frame(browser.find_element_by_id('coinfo_cp'))
            print("Crwaling *Annual* Financial Summary...")
            browser.find_elements_by_xpath('//*[@id="cns_Tab21"]')[0].click()
            html = BeautifulSoup(browser.page_source, 'html.parser')
            html_Y = html.find('table',{'class':'gHead01 all-width','summary':'주요재무정보를 제공합니다.'}) 
            html_tmp = html_Y.find('thead').find_all('tr')[1].find_all('th',attrs={"class":re.compile("^r03c")})
            dates = [''.join(re.findall('[0-9/]',html_tmp[i].text)) for i in range(len(html_tmp))]
            html_tmp = html_Y.find('tbody').find_all('tr')
            cols = []
            for i in range(len(html_tmp)):
                if '\xa0' in html_tmp[i].find('th').text:
                    tx = re.sub('\xa0','',html_tmp[i].find('th').text)
                else:
                    tx = html_tmp[i].find('th').text
                cols.append(tx)
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
            df = pd.DataFrame(data=values, columns=date, index=col)
            t.append(df)
        return t    

if __name__ == '__main__':
    print("Starting BS Crawler...")
    argument = sys.argv
    del argument[0]
    print("Mode you requested : {}".format(argument[0]))
    bs_crawler = BSCrawler(argument[0])