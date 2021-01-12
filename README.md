# StockTradingAutomation
trading automation using machine learning and quant-methodologies.

### 파이썬 주식 자동매매 개발
* 자동화 workflow는 Jocoding님의 틀을 기본으로 하되 매매 기준이 되는 모델링을 ML과 Quant 방법론을 이용해 구현한다.
* 특정 시점 기준으로 상장된 주식들의 주식 가격정보와 (연결)재무제표 데이터를 크롤링 및 DB 구축-관리한다.
* 자동화 스케쥴러를 통해 Daily Update 정보를 받아온다. (단, 분기/연간 데이터는 발표일 기간으로 설정)
* MPT에 따른다.

### Data
* 모든 상장 기업들 목록 (by KRX 한국거래소)
* 모든 상장 주식 가격 정보 (by Naver Finance)
* 모든 상장 주식 재무 제표 (by Naver Finance)
* 특정 주식 (연결)재무제표 (by DART)

### References
* https://shinminyong.tistory.com/15
* https://www.youtube.com/watch?v=Y01D2J_7894
* 파이썬 증권 데이터 분석 - 김황후 저

### Environments
* python 3.6.11
* MariaDB 10.5
* AWS Lightsail :
* CREON API
* Slack
* Dart-Fss, OpenDartReader
* Selenium, BeautifulSoup
