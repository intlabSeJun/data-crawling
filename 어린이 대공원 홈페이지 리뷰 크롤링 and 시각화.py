from bs4 import  BeautifulSoup
import urllib.request
import os
os.environ["PYTHONIOENCODING"] = "utf-8"
import warnings
warnings.filterwarnings(action='ignore')
from konlpy.tag import Okt # 트위터에서 만든 한글용 형태소 분석기.
import re
import joblib
import pickle
import copy
import matplotlib.pyplot as plt


def _model_inference(sentence):
    st = sentence
    # 0) 입력 텍스트에 대한 전처리 수행
    st = re.compile(r'[ㄱ-ㅣ가-힣]+').findall(st)
    st = [" ".join(st)]

    # 1) 입력 텍스트의 피처 벡터화
    st_tfidf = tfidf.transform(st)

    # 2) 최적 감성분석 모델에 적용하여 감성 분석 평가
    st_predict = linear.predict(st_tfidf)

    return st_predict


def _get_tags():
    tr_tags = []
    for i in range(1,4):
        address_new = address + str(i)
        print(address_new)
        html = urllib.request.urlopen(address_new)
        soup = BeautifulSoup(html, 'html.parser')
        tr_tags.append(soup.find('body').find('tbody').findAll('tr'))

    return tr_tags


def _predict(title, contents):
    value = _model_inference(title)
    value2 = 0
    count = 0
    for content in contents:
        value2 += _model_inference(content)
        count += 1

    value2 = (value*weight+value2)/(count+1*weight)

    if value2 > thres:
        return '긍정', value2
    else:
        return '부정', value2


def _extract_content(tags):
    """
    후기 내용으로 들어가 본문을 각 줄별로 크롤링 함.
    <br>을 포함하고 있는 내용은 약간 다르게 해야함.
     - 해당 객체는 beautifulsoup로 묶여 있고 generator 구조임.
     - 반복문을 사용하여 불러오되 <br>은 제외시키고 문자열만 받아와야함.
     - 만약 soup.string, strings를 사용하면 원하는 결과가 안나옴. (string은 마지막 문자열만 받아옴.)
    """
    contet_address = 'https://www.sisul.or.kr/' + tags.find('a')['href']
    html = urllib.request.urlopen(contet_address)
    soup = BeautifulSoup(html, 'html.parser')
    tbody_td = soup.find('tbody').find('td', attrs={'class':'view_contents'})
    contents = []

    for content in tbody_td:
        len_content = len(content)
        if len_content in [0,1,2]:
            continue
        # 영어 제외하고, 문자열 전환, 공백제거, '.' 기준 split
        sub_content = re.sub('[a-zA-z]', '', content.string.strip()).split('.')

        # 문자열이 너무 긴 것은 n등분. 100개 이상만,
        for i in range(len(sub_content)):
            if sub_content[i] == '' or sub_content[i] == ' ':
                continue
            a = len(sub_content[i])//lenth_limit
            if a == 0 or a == 1:
                contents.append(sub_content[i])
            else:
                for k in range(a):
                    if lenth_limit*(k+1) < len(sub_content[i]):
                        contents.append(sub_content[i][lenth_limit*k:])
                    else:
                        contents.append(sub_content[i][lenth_limit*k:lenth_limit*(k+1)])

    return contents  # 본문 내용이 각 줄별로 담긴 리스트. (줄 수, 문자열)


def _extract_info(tags):
    # '관리자'가 작성한 후기는 제외하고 나머지들을 모은다.
    tags_string = tags.strings
    buf = []
    for tag in tags_string:
        str = tag.strip()
        if str == '':
            continue
        buf.append(tag.strip())
    if '관리자' in buf or '어린이위원회' in buf:
        return
    del buf[2]  # 이름은 제외.
    contents = _extract_content(tags)

    # 각 정보들을 딕셔너리에 담고, 이를 리스트로 모아준다. (*딕셔너리는 주소를 공유하므로 deepcopy)
    info_dic['no'] = buf[0]
    info_dic['title'] = buf[1]
    info_dic['category'] = buf[2]
    info_dic['date'] = buf[3]
    info_dic['contents'] = contents  #후기 내용.
    result, score = _predict(info_dic['title'], info_dic['contents'])  #후기 제목 긍정/부정 이진분류
    info_dic['predict'] = result
    info_dic['score'] = score
    info_total.append(copy.deepcopy(info_dic))


def _parsing(tr_tags):
    # tag를 타고 들어가 정보를 추출함.
    for i in range(len(tr_tags)):
        for j in range(len(tr_tags[i])):
            _extract_info(tr_tags[i][j])


def get_info():
    tr_tags = _get_tags()  # 후기들에 대한 tr_tags 얻음.
    _parsing(tr_tags)  # tr_tags 타고 들어가 필요한 정보 추출해서 전역딕셔너리에 저장.


def save():
    f = open("outputs.txt", "w", encoding="UTF-8")
    for i in range(len(info_total)):
        for code, name in info_total[i].items():
            f.write(f'{code} : {name}, ')
        f.write('\n')


def find_review(year, month):
    select = []
    for i in range(len(info_total)):
        date = info_total[i]['date']
        if date[:date.find('.')] == year and date[date.find('.')+1:date.rfind('.')] == month:
            select.append(info_total[i])

    return select


def _sum():
    data = {}
    for info in info_total:
        date = info['date']
        year, month, predict = date[:date.find('.')], date[date.find('.') + 1:date.rfind('.')], info['predict']
        if year not in data:
            data[year] = [[0 for _ in range(2)] for _ in range(12)]  # 2차원 빈 리스트 생성. (12,2)
        score = 0 if predict == '긍정' else 1
        data[year][int(month) - 1][score] += 1

    return data


def bar_plot():
    data = _sum()

    for key, values in data.items():
        good_score = [values[i][0] for i in range(12)]
        bad_socre = [values[i][1] for i in range(12)]

        plt.title(key + ' year',fontsize=15)
        plt.bar(range(12), good_score, width=0.5)
        plt.bar(range(12), bad_socre, bottom=good_score, width=0.5)
        plt.xlabel('month', fontsize=15)
        plt.ylabel('count', fontsize=15)
        plt.xticks(range(12), range(1,13))
        plt.yticks(range(0,max(good_score)+min(good_score)+2))
        plt.legend(['good', 'bad'], fontsize=15)

        plt.tight_layout()
        plt.show()


"""
서울 어린이 대공원 후기 웹 크롤링 & 이진분류 & 시각화
● 웹 크롤링 & 이진분류
 ○ 후기들은 제목, 내용으로 구성. 모두 크롤링 한 후 미리 학습된 모델을 inference 하여 이진분류 (긍정/부정)
    - 제목만으로는 긍정/부정 나누기가 모호한 경우가 있음 -> 내용을 각 줄마다 크롤링 하여 각 줄에 대한 예측값을 얻음.
    - 얻은 예측값들(제목,내용(각 줄별로)) 평균을 내어 최종 예측값으로 사용.
      - 모델을 통과하면 긍정->1,부정->0 출력. 제목 + 내용(5줄)이면 총 6개의 결과를 평균함.
      - 평균이 0.6 초과하면 긍정, 이하면 부정으로 최종 예측값 사용. (0.6으로 threshold 주었을 때, 예측성능 올라감)
 ○ 데이터 저장
    - 각 후기제목에 대한 정보를 딕셔너리로 저장하고, 이를 리스트에 쌓음.
      - 각 딕셔너리 정보 : {'predict','score', 'no', 'title', 'category', 'date','contents'}
        - 'predict': 최종 예측. 긍정/부정
        - 'score': 최종 예측 값. 
        - 'title': 후기 제목.
        - 'contents': 후기 내용.
        - 'no': 홈페이지 리뷰 번호.
        - 'date': 작성 년/월/일.
        - 'category': 어린이 대공원 행사 프로그램 명
    - outputs.txt 저장.
    
● 시각화
 ○ 년마다 월별로 '긍정/부정' count.
    - 누적 막대 그래프를 사용하여 각 년/월에서의 후기에 대한 예측값들을 합산해서 나타냄.

-> 따라서, 특정 년/월에 '긍정' 혹은 '부정'이 유독히 많은 구간을 알 수 있음. 
-> 또한, 공공API 크롤링을 통해 얻은, 년/월 마다 행사 내용, 년/월 방문객 수를 통한 종합적인 분석을 할 수 있음. 
"""
address = 'https://www.sisul.or.kr/open_content/childrenpark/bbs/bbsMsgList.do?keyfield=title&listsz=50&bcd=reviews&pgno='
info_dic = {'predict':None,'score':None, 'date':None, 'no':None, 'category':None, 'title':None, 'contents':None} #전역딕셔너리
info_total = [] #전역리스트 : 모든 후기들에 대한 정보들이 딕셔너리로 쌓인다.


if __name__ == '__main__':
    # 미리 학습된 한글 형태소 분석기와, 이진 분류 모델 가중치 받아오기. (*무조건 main문에 있어야 함)
    okt = Okt()  # 트위터에서 만든 한글 형태소 분석기
    def okt_tokenizer(text):
        token = okt.morphs(text)
        return token
    tfidf = pickle.load(open("tfidf.pickle", "rb"))
    linear = joblib.load("best.pkl")
    thres = 0.6
    weight = 1
    lenth_limit = 150

    len_mean = []
    # 리뷰 크롤링 ('긍정/부정'예측값, 점수, 번호, 후기제목, 카테고리, 날짜, 후기내용)
    get_info()

    # 수집한 정보 + 예측값 .txt 저장.
    save()

    # 년도와 월을 주고 해당 리뷰 찾기.
    year, month = '2018', '01'
    select = find_review(year, month)
    print(f'YearMonth :{year + month} 리뷰들.')
    for data in select:
        print(data)

    # bar char 시각화
    bar_plot()

