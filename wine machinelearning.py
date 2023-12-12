import pandas as pd
import seaborn as sns

df=pd.read_csv("C:/Users/ku001/Desktop/WineQuality.csv",index_col='id')
pd.set_option('display.max_columns',13)
#print(df.head())

#누락 데이터가 있는지 확인
#print(df.info())
#print(df.describe(include='all'))

#분석에 활용할 열 선택
ndf=df[['citric acid','sulphates','alcohol','pH','quality']]

#변수 선택
x=ndf[['citric acid','sulphates','alcohol','pH']]
y=ndf['quality']

#설명 변수 데이터를 정규화
from sklearn import preprocessing
x=preprocessing.StandardScaler().fit(x).transform(x)

#train data와 test data로 구분
from sklearn.model_selection import train_test_split
x_train, x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=10)
print('train data 개수: ', x_train.shape)
print('test data개수: ', x_test.shape)
     

#모형 학습
from sklearn.neighbors import KNeighborsClassifier
model=KNeighborsClassifier(n_neighbors=5)
model.fit(x_train,y_train)

#예측
y_hat=model.predict(x_test)

#모형 성능 평가
from sklearn import metrics
model_matrix=metrics.confusion_matrix(y_test,y_hat)
print(model_matrix)

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

# 정확도 계산
accuracy = accuracy_score(y_test, y_hat)
print(f'Accuracy: {accuracy:.4f}')

# 정밀도 계산
precision = precision_score(y_test, y_hat, average='weighted')
print(f'Precision: {precision:.4f}')

# 재현율 계산
recall = recall_score(y_test, y_hat, average='weighted')
print(f'Recall: {recall:.4f}')

# F1 점수 계산
f1 = f1_score(y_test, y_hat, average='weighted')
print(f'F1 Score: {f1:.4f}')

# 분류 보고서 출력
print(classification_report(y_test, y_hat))
