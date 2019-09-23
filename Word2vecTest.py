from konlpy.tag import Okt
from gensim.models import Word2Vec

okt=Okt()
fread = open('C:/PycharmProjects/LSTMTutorial/wikiextractor-master/wiki_data.txt', encoding="utf8")
# 파일을 다시 처음부터 읽음.
n=0
result = []

while True:
    line = fread.readline() #한 줄씩 읽음.
    if not line: break # 모두 읽으면 while문 종료.
    n=n+1
    if n%5000==0: # 5,000의 배수로 While문이 실행될 때마다 몇 번째 While문 실행인지 출력.
        print("%d번째 While문."%n)
    tokenlist = okt.pos(line, stem=True, norm=True) # 단어 토큰화
    temp=[]
    for word in tokenlist:
        if word[1] in ["Noun"]: # 명사일 때만
            temp.append((word[0])) # 해당 단어를 저장함

    if temp: # 만약 이번에 읽은 데이터에 명사가 존재할 경우에만
      result.append(temp) # 결과에 저장
fread.close()

model = Word2Vec(result, size=100, window=5, min_count=5, workers=4, sg=0)
                #size:워드 벡터의 특징값, 즉 임베딩 된 벡터의 차원
                #windows:컨텍스트 윈도우 크기
                #min_count:단어 최소 빈도 수 제한(빈도가 적은 단어들은 학습하지 않는다.)
                #workers:학습을 위한 프로세스 수
                #sg:0은 CBOW, 1은 Skip-gram
                #CBOW: 주변 단어들을 가지고 중간 단어를 예측하는 방법
                #Skip-gram 중간 단어를 가지고 주변 단어들을 예측하는 방법
model.save('Word2Vec_Wiki.model')