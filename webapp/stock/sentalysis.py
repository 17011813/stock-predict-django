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


