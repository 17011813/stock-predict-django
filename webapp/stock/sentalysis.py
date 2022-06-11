#######################################################################################################################################
#############################################            뉴스 감성지수 분류 모델            #############################################
#######################################################################################################################################


# 형태소 분석
import scipy as sp
import pandas as pd
import numpy as np

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

stop_words = [('\n', 14309), ('(', 9270), (')', 9269), ('종목', 6965), ('코스닥', 6412), ('상위', 5791), ('특징', 5263), 
                ('주', 5162), ('적', 4915), ('기술', 4912), ('분석', 4894), ('코스피', 4501), ('장', 4017), ('50', 3104), 
                ('B', 2529), ('순', 2445), ('A', 2371), ('오전', 2202), ('률', 2169), ('하락', 2135), (',', 2007), ('증시', 1950), 
                ('-', 1780), ('요약', 1720), ('20', 1646), ('매수', 1445), ('도', 1391), ('외국인', 1380), ('기관', 1332), ('오후', 1310), 
                ('/', 1231), ('상', 1166), ('승률', 1153), ('연속', 1082), ('매', 1070), ('주요', 1066), ('기준', 939), ('마감', 852), 
                ('계', 849), ('외국', 848), ('[', 784), (']', 760), ('수량', 719), ('…', 659), ('도일', 563), ('수일', 494), ('후', 483), 
                ('공시', 446), ("'", 441), ('시', 399), ('가', 397), ('황', 390), ('10', 360), ('9', 353), ('8', 350), ('7', 349), ('미국', 332), 
                ('및', 327), ('에', 316), ('"', 278), ('감', 274), ('장마', 263), ('뉴욕증시', 255), ('급등', 252), ('유럽', 236), ('뉴스', 226), 
                ('상한', 225), ('6', 223), ('+', 223), ('·', 212), ('비율', 209), ('신용', 205), ('개장', 199), ('기업', 198), ('등락', 195), 
                ('테마', 170), ('중', 163), ('↑', 162), ('상승', 155), ('시장', 150), ('국내', 150), ('금리', 138), ('동향', 136), ('英', 134), 
                ('투자', 133), ('실적', 130), ('주가', 129), ('환율', 128), ('금액', 126), ('美', 124), ('초반', 122), ('신고', 121), ('전', 121), 
                ('52', 118), ('ETF', 118), ('기', 115), ('의', 114), ('↓', 114), ('株', 108), ('일자', 105), ('발표', 100), ('상승세', 98), ('5', 97), 
                ('증권사', 96), ('신규', 96), ('채권', 94), ('지수', 90), ('유가', 90), ('일제', 86), ('히', 84), ('은', 83), ('국제', 81), ('의견', 77), 
                ('&', 77), ('재송', 77), ('반도체', 77), ('.', 74), ('이', 74), ('폭', 73), ('마켓', 73), ('유지', 72), ('로', 72), ('거래', 69), ('뷰', 69), 
                ('...', 66), ('일정', 66), ('출발', 65), ('연결', 63), ('2', 62), ('상품', 62), ('FTSE', 62), ('선', 61), ('국내외', 61), ('연', 60), 
                ('나스닥', 60), ('이상', 60), ('영업', 60), ('는', 59), ('혼', 59), ('소', 57), ('‘', 56), ('예', 56), ('세', 55), ('이익', 55), ('상치', 55), 
                ('체결', 54), ('준', 52), ('?', 52), ('3', 52), ('1억원', 52), ('Asia', 52), ('우려', 51), ('개미', 50), ('’', 48), ('최고', 47), ('주식', 47), 
                ('종가', 46), ('변동', 46), ('삼성', 46), ('동시', 45), ('호가', 45), ('특이', 45), ('오늘', 45), ('이슈', 44), ('`', 44), ('강세', 43), ('증권', 41), 
                ('전일', 40), ('FOMC', 39), ('지', 39), ('분', 39), ('中', 39), ('SG', 39), ('스', 39), ('에도', 38), ('반등', 38), ('등', 38), ('S', 38), ('한', 38), 
                ('2021년', 38), ('스케줄', 37), ('급락', 37), ('1분', 37), ('휴장', 37), ('관련', 36), ('석유', 36), ('대', 35), ('SK', 35), ('조', 35), ('인상', 35), 
                ('전자', 34), ("'…", 34), ('매도', 34), ('한국', 33), ('해외', 33), ('으로', 32), ('인플레', 32), ('"…', 32), ('조세', 32), ('만에', 32)]

tokens_ko = [each_word for each_word in tokens_ko
           if each_word not in stop_words]

ko = nltk.Text(tokens_ko)
print(ko.vocab().most_common(100))



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