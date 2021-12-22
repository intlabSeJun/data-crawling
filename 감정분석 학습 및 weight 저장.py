import pandas as pd
import os
os.environ["PYTHONIOENCODING"] = "utf-8"
import warnings
warnings.filterwarnings(action='ignore')
from konlpy.tag import Okt # 트위터에서 만든 한글용 형태소 분석기.
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
import re
import joblib
import pickle


"""
감정 txt data : 1-긍정, 0-부정 (영화 평점 1~10점 중 1~4는 0, 9~10은 1로 표시하고 나머지는 제외)
○ rating_train.txt :  150k (용량)
○ rating_test.txt : 50k
"""
#pd.show_versions()
path = '/조별기말/13장'
nsmc_train_df = pd.read_csv(path + '/' + 'ratings_train.txt', encoding='utf8', sep='\t', engine='python') #1:긍정, 0:부정
print(nsmc_train_df.head())#예시. ( 별점에 따라 긍정 부정을 나누었지만 별점은 낮게 주고 한줄평은 좋다고 하는 경우, 비꼬는 경우 등 어렵다)
nsmc_train_df.info() # 학습 데이터 분포

nsmc_train_df = nsmc_train_df[nsmc_train_df['document'].notnull()] #결측치 제거

nsmc_train_df.info()

nsmc_train_df['label'].value_counts() #레이블 개수

# 한글 외의 문자 제거. 한글에서 나올 수 있는 모든 글자가 아니면 제거한다.
nsmc_train_df['document'] = nsmc_train_df['document'].apply(lambda x : re.sub(r'[^ ㄱ-ㅣ가-힣]+', " ", x))
nsmc_train_df.head()

#test data도 동일하게.
nsmc_test_df = pd.read_csv(path + '/' + 'ratings_test.txt', encoding='utf8', sep='\t', engine='python')
nsmc_test_df.head()

nsmc_test_df.info()

nsmc_test_df = nsmc_test_df[nsmc_test_df['document'].notnull()]

print(nsmc_test_df['label'].value_counts())

nsmc_test_df['document'] = nsmc_test_df['document'].apply(lambda x : re.sub(r'[^ ㄱ-ㅣ가-힣]+', "", x))

#konlpy 분석모델.
okt = Okt() #트위터에서 만든 한글 형태소 분석기

def okt_tokenizer(text):
    token = okt.morphs(text)
    return token

# 단어 벡터화 작업 수행 TF-IDF 반식.
tfidf = TfidfVectorizer(tokenizer = okt_tokenizer, ngram_range=(1,2), min_df=3, max_df=0.9)
"""
ngram_range : 토큰의 단어크기 여기서는 1~2개
min_df : 토큰의 최소 출현 빈도도
mad_df : 최대 90% 이하인 것만 사용.
"""
tfidf.fit(nsmc_train_df['document']) #벡터화할 데이터를 넣는다.
nsmc_train_tfidf = tfidf.transform(nsmc_train_df['document'])
pickle.dump(tfidf, open("../tfidf.pickle", "wb"))

#로지스틱 회귀 모델 이용 -> 이진분류 모델 구축. 학습.
SA_lr = LogisticRegression(random_state = 0)
SA_lr.fit(nsmc_train_tfidf, nsmc_train_df['label'])


params = {'C': [1, 3, 3.5, 4, 4.5, 5]} #하이퍼 파라미터 설정
SA_lr_grid_cv = GridSearchCV(SA_lr, param_grid=params, cv=3, scoring='accuracy', verbose=1) # 교차검증 cv = 3 : 조금 더 데이터 평가를 잘하게 하기 위함. 데이터 전체를 n등분해서 따로 검증?

# 언제가 최적인지 여러 모델들을 돌려본다
SA_lr_grid_cv.fit(nsmc_train_tfidf, nsmc_train_df['label'])
print(SA_lr_grid_cv.best_params_, round(SA_lr_grid_cv.best_score_, 4)) # cv:3 일 경우 좋다

# 최적 파라미터의 best 모델 저장
SA_lr_best = SA_lr_grid_cv.best_estimator_
save_model = joblib.dump(SA_lr_best, '../best.pkl')

# 평가용 데이터의 피처 벡터화 : 실행시간 6분 정도 걸립니다 ☺
nsmc_test_tfidf = tfidf.transform(nsmc_test_df['document'])
test_predict = SA_lr_best.predict(nsmc_test_tfidf)

print('감성 분석 정확도 : ', round(accuracy_score(nsmc_test_df['label'], test_predict), 3))

# 실제 문장을 입력하여 이진분류 해본다.
while(1):
    st = input('감성 분석할 문장입력 >> ')

    # 0) 입력 텍스트에 대한 전처리 수행
    st = re.compile(r'[ㄱ-ㅣ가-힣]+').findall(st)
    print(st)
    st = [" ".join(st)]
    print(st)

    # 1) 입력 텍스트의 피처 벡터화
    st_tfidf = tfidf.transform(st)

    # 2) 최적 감성분석 모델에 적용하여 감성 분석 평가
    st_predict = SA_lr_best.predict(st_tfidf)

    # 3) 예측 값 출력하기
    if(st_predict== 0):
        print(st , "->> 부정 감성")
    else :
        print(st , "->> 긍정 감성")






