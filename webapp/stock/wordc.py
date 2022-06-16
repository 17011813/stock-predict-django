import requests
import pandas as pd
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
import nltk 
from konlpy.tag import Kkma
from konlpy.tag import Twitter
from wordcloud import WordCloud

date = '20220613' 
page = ''
naver_url = 'https://finance.naver.com/news/news_list.naver?mode=LSS3D&section_id=101&section_id2=258&section_id3=401&date={}&page={}}'.format(date, page)
headers = {'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/84.0.4147.89 Safari/537.36'}
req = requests.get(naver_url, headers=headers)
soup = BeautifulSoup(req.text)
title_list = soup.select('ul.realtimeNewsList .newsList top')

result_list = []
for title in title_list:
    news_title = title.select_one('.articleSubject').text.strip()
    result_list.append([news_title])

print(result_list)





date='20220613' 

naver_url = 'https://finance.naver.com/news/mainnews.naver?date={}'.format(date)
headers = {'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/84.0.4147.89 Safari/537.36'}
req = requests.get(naver_url, headers=headers)
soup = BeautifulSoup(req.text)
title_list = soup.select('div.mainNewsList .newsList > li')

result_list = []
for title in title_list:
    news_title = title.select_one('.articleSubject').text.strip()
    result_list.append([news_title])

print(result_list)

    #  for date in dates:
#         url = base_url.format(date)
#         res = requests.get(url, headers=headers)
#         if res.status_code == 200:
#             soup = BeautifulSoup(res.text)
#             title_list = soup.select('div.mainNewsList .newsList')
#             for title in title_list:
#                 try:
#                     news_title = title.select_one('.articleSubject').text.strip()
#                     result_list.append([news_title])
#                 except:
#                     error_cnt += 1





# import scipy as sp
# import pandas as pd
# import numpy as np

# from konlpy.tag import Kkma        ; kkma = Kkma()
# from konlpy.tag import Hannanum    ; hannanum = Hannanum()
# from konlpy.tag import Okt         ; t = Okt()     # 구 트위터
# from konlpy.tag import *
# import pickle

# import os
# import matplotlib.pyplot as plt
# from wordcloud import WordCloud, STOPWORDS
# from PIL import Image

# import matplotlib.font_manager as fm
# plt.rc('font', family='NanumGothic')

# import matplotlib as mpl
# mpl.rcParams['axes.unicode_minus'] = False

# import warnings
# warnings.filterwarnings('ignore')

# import requests
# from bs4 import BeautifulSoup
# import pandas as pd
# from urllib import parse


# date = '2022-06-13'

# # 팍스넷 뉴스 타이틀
# result_list = []
# error_cnt = 0

# def paxnet_news_title(dates):
#     base_url = 'http://www.paxnet.co.kr/news/much?newsSetId=4667&currentPageNo={}&genDate={}&objId=N4667'
#     headers = {
#         'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/84.0.4147.89 Safari/537.36'
#     }
    
#     for date in dates:
#         for page in range(1, 3):
#             url = base_url.format(page, date)
#             res = requests.get(url, headers=headers)
#             if res.status_code == 200:
#                 soup = BeautifulSoup(res.text)
#                 title_list = soup.select('ul.thumb-list li')
#                 for title in title_list:
#                     try:
#                         news_title = title.select_one('dl.text > dt').text.strip()
#                         result_list.append([news_title])
#                     except:
#                         error_cnt += 1

# paxnet_news_title(date)
# paxnews = result_list

# result_list = []
# error_cnt = 0

# def naver_news_title(dates):
#     base_url = 'https://finance.naver.com/news/mainnews.naver?date={}'
#     headers = {
#         'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/84.0.4147.89 Safari/537.36'
#     }
    
#     for date in dates:
#         url = base_url.format(date)
#         res = requests.get(url, headers=headers)
#         if res.status_code == 200:
#             soup = BeautifulSoup(res.text)
#             title_list = soup.select('div.mainNewsList .newsList')
#             for title in title_list:
#                 try:
#                     news_title = title.select_one('.articleSubject').text.strip()
#                     result_list.append([news_title])
#                 except:
#                     error_cnt += 1

# naver_news_title(date)

# navernews = result_list


# all_title = pd.concat(paxnews, navernews)
# all_title.to_csv('OneDay.csv')

# news_title = pd.read_csv('OneDay.csv')

