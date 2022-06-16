#######################################################################################################################################
#############################################            뉴스 감성지수 분류 모델            #############################################
#######################################################################################################################################


# 형태소 분석
import scipy as sp
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from konlpy.tag import Kkma
from konlpy.tag import Hannanum
from konlpy.tag import Okt
from konlpy.tag import *
import pickle

kkma = Kkma()
hannanum = Hannanum()
t = Okt()     # 구 트위터
 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline 
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


# 팍스넷 + 네이버 뉴스데이터 연결하기

news_df = pd.read_csv('팍스넷&네이버_뉴스타이틀.csv')
print(news_df.shape)
news_df


title_list = news_df.뉴스제목.values.tolist()
print(len(title_list))

title_text = ''

for each_line in title_list:
    title_text = title_text + each_line + '\n'


# 형태소 분석
tokens_ko = t.morphs(title_text)

import nltk
ko = nltk.Text(tokens_ko)

print(len(ko.tokens))          # 토큰 전체 개수
print(len(set(ko.tokens)))     # 토큰 unique 개수


# 빈도수가 높은 단어 출력
print(ko.vocab().most_common(200))


# 불용어(stopwords) 제거하기

stop_words = ['\n',"'",'…',',','[',']','(',')','"','주','에','코스닥','특징','종목','·','장','코스피','증시','-','적',\
              '도','기술','분석','마감','‘','`','요약','가','’','의','이','오전','★','은','“','대','”','한','B','로',\
              '?','3','선','A','오후','는','5','!','"…','상','들','1','만에','제','2','…"','20','일','서','명',"'…",'기',\
              '···','10','소','등','으로','자','전','률','미','...','50','세','시','안','폭',"…'",'만','9','VI','까지',\
              '눈','더','e','량','고','인','52','성','띄네','1%','부터','다','감','을','지','4','에도','수','7','것','째',\
             '체크','기','···','중','계','관련','왜','1억원','총','내','과','젠','또','연','엔','차','굿모닝','할','8','.',\
             '보다','새','주간','전망','추천','이슈','플러스','사','개월','때','..','임','속','’…','G','나','개','원','에서',\
             '하는','이유','달','→','권','?…','단독','간','배','30','K','저','와','하','/','1조','6','두','해야','분','형',\
             '황','공','&','앞두고','보','문','이번','익','X','1억',']"','치','산','를','오','해','S','우리','그','된','준','▶',\
             '건','재','반','라','10년','초','3분','월','신','p','급','조','줄','경','했다','구','진','이어','올','발','vs','강',\
             '국','9억','1년','난','판','면','"(','`…','살','아','인데','번','텍','팜','8월','Q','메','2년','점','하고','10월',\
             'D','비','됐다','채',"]'",'보니','손','확','종','동','팔','40','타','~','9월','2100','30%','땐','말','한다','요',\
             "',",'스','…`','단','16','길','12','3억','회','될까','호','용','2조','번째','일까','듯','최']

tokens_ko = [each_word for each_word in tokens_ko
           if each_word not in stop_words]

ko = nltk.Text(tokens_ko)
print(ko.vocab().most_common(100))


from wordcloud import WordCloud, STOPWORDS
from PIL import Image
data = ko.vocab().most_common(300)
wordcloud = WordCloud(font_path='c:/Windows/Fonts/malgun.ttf',
                     relative_scaling=0.2,
                     background_color='white').generate_from_frequencies(dict(data))


plt.imshow(wordcloud)
plt.axis('off')
plt.show()



# 형태소 분석을 위한 함수

def tokenizer(text):
    okt = Okt()
    return okt.morphs(text)

    
def data_preprocessing():
    # 수집한 데이터 읽어오기
    # news_df = pd.read_excel()
    
    # 학습셋, 테스트셋 분리
    title_list = news_df['뉴스제목'].tolist()
    price_list = news_df['주가변동'].tolist()
    
    from sklearn.model_selection import train_test_split
    
    # 데이터의 80%는 학습셋, 20%는 테스트셋
    title_train, title_test, price_train, price_test = train_test_split(title_list, price_list, test_size=0.3, random_state=42)
    
    return title_train, title_test, price_train, price_test

def learning(x_train, y_train, x_test, y_test):
    # 전처리가 끝난 데이터를 단어 사전으로 만들고
    # 리뷰별로 나오는 단어를 파악해서 수치화 (벡터화)해서 학습
    # tfidf, 로지스틱 회귀 이용
    
    tfidf = TfidfVectorizer(lowercase=False, tokenizer=tokenizer)
    # 로지스틱
    logistic = LogisticRegression(C=2, penalty='l2', random_state=0)     # C의 숫자가 너무 크면 과적합 (기본 1), penalty로 과적합 방지
    
    pipe = Pipeline([('vect',tfidf),('clf',logistic)])
    
    # 학습
    pipe.fit(x_train, y_train)
    
    # 학습 정확도 측정
    y_pred = pipe.predict(x_test)
    print(accuracy_score(y_test, y_pred))
    
    # 학습한 모델을 저장
    with open('pipe.dat', 'wb') as fp:     # 쓰기, 바탕화면에 저장됨
        pickle.dump(pipe, fp)
        
    print('저장완료')     # 학습된 모델 저장 완료

    
def using():
    # 객체를 복원, 저장된 모델 불러오기
    with open('pipe.dat','rb') as fp:     # 읽기
        pipe = pickle.load(fp)
        
    while True :
        text = input('뉴스 타이틀을 입력해주세요 : ')     # 인풋
        
        str = [text]
        
        # 예측 정확도
        r1 = np.max(pipe.predict_proba(str)*100)     # 확률값을 구해서 *100..?
        
        # 예측 결과
        r2 = pipe.predict(str)[0]     # 긍정('1'), 부정('0')
        
        if r2 == '1':
            print('코스피지수는 상승할 것으로 예상됩니다.')
        else: 
            print('코스피지수는 하락할 것으로 예상됩니다.')
            
        print('정확도 : %.3f' % r1)
        print('------------------------------------------------')

        
# 학습 함수

def model_learning():   # 감성분석 모델 생성
    title_train, title_test, price_train, price_test = data_preprocessing()
    learning(title_train, price_train, title_test, price_test)
    
# 사용 함수

def model_using():   # 감성분석 모델 사용
    using()

    
model_learning()

model_using()