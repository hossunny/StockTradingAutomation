# StockTradingAutomation
Quantative Methods & Technical Approach for Stock Trading Automation.

### Trading Strategy
* Universe Screener : Fundamentals(Balance Sheet, Financial Summary, Volatility etc)를 통한 종목들 Filtering.
* Pair-Trading : Distance Approach (Correlation/CoIntegration) based Pair-Trading을 통한 Pair-wise Signal Modeling.
https://github.com/hossunny/StockTradingAutomation/blob/main/PairTrading/PairTrading_KOR.ipynb
* Machine Learning 
  * Short-term spread prediction : https://github.com/hossunny/StockTradingAutomation/blob/main/MachineLearning/SpreadPrediction_ML.ipynb
* Technical Indicator
  * OBV (On-Balance Volume) : https://github.com/hossunny/StockTradingAutomation/blob/main/TechnicalIndicator/OBV_ENG.ipynb
  * MAC (Simple/Exponential Moving Average Crossover) : https://github.com/hossunny/StockTradingAutomation/blob/main/TechnicalIndicator/MAC_ENG.ipynb
  * MACD (Moving Average Convergence Divergence) : https://github.com/hossunny/StockTradingAutomation/blob/main/TechnicalIndicator/MACD_ENG.ipynb
  * RSI (Relative Strength Index) : https://github.com/hossunny/StockTradingAutomation/blob/main/TechnicalIndicator/RSI_ENG.ipynb
  * MFI (Money Flow Index) : https://github.com/hossunny/StockTradingAutomation/blob/main/TechnicalIndicator/MFI_ENG.ipynb
  * Stochastic Oscillator : https://github.com/hossunny/StockTradingAutomation/blob/main/TechnicalIndicator/StochasticOscillator_ENG.ipynb
  * Bollinger Bands : https://github.com/hossunny/StockTradingAutomation/blob/main/TechnicalIndicator/BollingerBands_ENG.ipynb
  * Triple Screen Trading System : https://github.com/hossunny/StockTradingAutomation/blob/main/TechnicalIndicator/TripleScreen_ENG.ipynb
* Volatility Breakout : https://github.com/hossunny/StockTradingAutomation/blob/main/VolatilityBreakout/VolatilityBreakout_ENG.ipynb
* Text Mining : (not yet)

### Automation WorkFlow
* DataModeling : DailyUpdate로 StockPrice, MarketCap, KOSPI 등을 크롤링한다. Data Loader 속도 향상을 위해 21/03/02부터는 이전에 MariaDB로 구축한 Data들을 전부 HDF5 Cache로 저장하였다. 그에 맞는 Loader_v2도 구현하였다.
* Trading Logic : Pair-Trading Signal, Short-term trading signal -> HTS (CREON API) -> Slack
* Automation : Scheduler or Daemon Job
* Dash Board : Frontend (not yet)

### Data
* 모든 상장 기업들 목록, 시가 총액, 수정종가 (by KRX 한국거래소)
* 모든 상장 주식 가격 정보 & KOSPI (by Naver Finance)
* 모든 상장 주식 재무 제표 (요약) (by Naver Finance)
* 정기공시 (연결)재무제표, 손익계산서, 현금흐름표 (by DART-FSS)

### References
* https://shinminyong.tistory.com/15
* https://www.youtube.com/watch?v=Y01D2J_7894 (JoCoding)
* 파이썬 증권 데이터 분석 - 김황후 저
* 매트릭 스튜디오 - 문병로 저
* https://github.com/sharebook-kr/pykrx
* https://github.com/josw123/dart-fss
* https://github.com/FinanceData/OpenDartReader
* https://blog.naver.com/PostView.nhn?blogId=freed0om&logNo=221971659619&parentCategoryNo=64&categoryNo=67&viewDate=&isShowPopularPosts=false&from=postList
* https://nbviewer.jupyter.org/github/FinanceData/marcap/blob/master/marcap-tutorial.ipynb
* https://blog.naver.com/PostList.nhn?blogId=chunjein&from=postList&categoryNo=14

### Environments
* python 3.6.11
* MariaDB 10.5
* AWS Lightsail
* CREON API
* Slack
* Dart-Fss, OpenDartReader
* Selenium, BeautifulSoup
* pyautogui
* pykrx
* marcap
