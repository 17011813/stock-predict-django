#######################################################################################################################################
#############################################            뉴스 감성지수 분류 모델            #############################################
#######################################################################################################################################


# KOSPI200 데이터 수집
from pykrx import stock
import pandas as pd

kospi200_data = stock.get_index_ohlcv("20210101", "20220601", "1028") # 1년반 KOSPI200 데이터
kospi200_data = kospi200_data.dropna()

kospi200_data['가격변동'] = 0
for i in range(len(kospi200_data['종가']) - 1):
    if (kospi200_data['종가'][i] < kospi200_data['종가'][i+1]):     # 전날보다 종가가 높으면 1
        kospi200_data['가격변동'][i] = 1
    else:                                                          # 전날보다 종가가 낮거나 같으면 0
        kospi200_data['가격변동'][i] = 0

kospi200_data.to_csv('kospi200_주가데이터.csv')


# 수집날짜 리스트 생성
price_data = pd.read_csv('kospi200_주가데이터.csv')

df_0 = price_data[price_data['가격변동']==0]['날짜']
date_0 = []
for i in range(0, len(df_0)):
    date_0.append(str(df_0.tolist()[i])[:10].replace('-', ''))

df_1 = price_data[price_data['가격변동']==1]['날짜']
date_1 = []
for i in range(0, len(df_1)):
    date_1.append(str(df_1.tolist()[i])[:10].replace('-', ''))

# print(date_0)
# print(date_1)

# 뉴스 타이틀 크롤링
import requests
from bs4 import BeautifulSoup
import pandas as pd
from urllib import parse


# 팍스넷 뉴스 타이틀
result_list = []
error_cnt = 0

def paxnet_news_title(dates):
    base_url = 'http://www.paxnet.co.kr/news/much?newsSetId=4667&currentPageNo={}&genDate={}&objId=N4667'
    headers = {
        'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/84.0.4147.89 Safari/537.36'
    }
    
    for date in dates:
        for page in range(1, 3):
            url = base_url.format(page, date)
            res = requests.get(url, headers=headers)
            if res.status_code == 200:
                soup = BeautifulSoup(res.text)
                title_list = soup.select('ul.thumb-list li')
                for title in title_list:
                    try:
                        news_title = title.select_one('dl.text > dt').text.strip()
                        result_list.append([news_title])
                    except:
                        error_cnt += 1


paxnet_news_title(date_0)

title_df_0 = pd.DataFrame(result_list, columns=['뉴스제목'])
title_df_0['주가변동'] = 0


result_list = []
error_cnt = 0

paxnet_news_title(date_1)

title_df_1 = pd.DataFrame(result_list, columns=['뉴스제목'])
title_df_1['주가변동'] = 1
title_df_1.head()


title_df = pd.concat([title_df_0, title_df_1])
title_df.to_csv('팍스넷_뉴스타이틀.csv', index=False, encoding='utf-8')



# 네이버 경제분야 뉴스 타이틀
result_list = []
error_cnt = 0

def naver_news_title(dates):
    base_url = 'https://news.naver.com/main/ranking/popularDay.nhn?rankingType=popular_day&sectionId=101&date={}'
    headers = {
        'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/84.0.4147.89 Safari/537.36'
    }
    
    for date in dates:
        url = base_url.format(date)
        res = requests.get(url, headers=headers)
        if res.status_code == 200:
            soup = BeautifulSoup(res.text)
            title_list = soup.select('ol.ranking_list li')
            for title in title_list:
                try:
                    news_title = title.select_one('div.ranking_headline').text.strip()
                    result_list.append([news_title])
                except:
                    error_cnt += 1

naver_news_title(date_0)


title_df_2 = pd.DataFrame(result_list, columns=['뉴스제목'])
title_df_2['주가변동'] = 0


result_list = []
error_cnt = 0

naver_news_title(date_1)

title_df_3 = pd.DataFrame(result_list, columns=['뉴스제목'])
title_df_3['주가변동'] = 1


title_df2 = pd.concat([title_df_2, title_df_3])
title_df2.to_csv('네이버_뉴스타이틀.csv', index=False, encoding='utf-8')


all_title = pd.concat([title_df, title_df2])
all_title.to_csv('팍스넷&네이버_뉴스타이틀.csv')

