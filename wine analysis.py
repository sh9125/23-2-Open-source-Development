import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

df = pd.read_csv("C:/Users/ku001/Desktop/WineQuality.csv",index_col='id')
pd.set_option('display.max_columns',None)

# 타입(Red, White)을 기준으로 분류
grouped = df.groupby(['Type'])

# Red 와인인 것과 White 와인인 것을 따로 선택하여 추출
group1 = grouped.get_group('Red Wine')
group2 = grouped.get_group('White Wine')
group1.drop('Type',axis=1,inplace=True)
group2.drop('Type',axis=1,inplace=True)

# Red 와인과 White 와인의 특성들의 평균 값을 구함
# Red 와인의 고정산도가 White 와인의 고정산도보다 높고 White 와인의 자유 이산화황과 총 이산화황이 Red 와인의 자유 이산화황과 총 이산화황보다 월등히 높다는 것을 알 수 있다.
average1 = group1.mean()
average2 = group2.mean()
#print('group1 average')
#print(average1)
#print('\n')
#print('group2 average')
#print(average2)

# Red 와인을 품질별로 구분한 뒤 각 품질별 특성들의 표준편차를 구함.
# 품질 점수가 3인 와인을 제외하곤 품질이 올라갈수록 고정산도의 표준편차가 커지고 구연산의 표준편차도 커진다.
group1Q = group1.groupby(['quality'])
std_all1 = group1Q.std()
#print('group1 std')
#print(std_all1)
#print('\n')

# White 와인을 품질별로 구분한 뒤 각 품질별 특성들의 표준편차를 구함.
# 품질 점수가 커질수록 총 이산화황의 표준편차는 작아진다. 
group2Q = group2.groupby(['quality'])
std_all2 = group2Q.std()
#print('group2 std')
#print(std_all2)

# 타입별 와인을 품질별로 묶어 각 특성의 최댓값과 최솟값의 차이를 구함. 
def min_max(x):
    return x.max()-x.min()
grouped2=group1.groupby(['quality'])
agg_all2=grouped2.apply(min_max)
#print('group1 min_max')
#print(agg_all2)
#print('\n')

grouped3=group2.groupby(['quality'])
agg_all3=grouped3.apply(min_max)
#print('group2 min_max')
#print(agg_all3)

# Red 와인과 White 와인의 각 특성들의 상관계수를 히트맵으로 표현
sns.set_style('whitegrid')

fig1=plt.figure(figsize=(15,5))
ax1=fig1.add_subplot(1,2,1)
ax2=fig1.add_subplot(1,2,2)
sns.heatmap(group1.corr(),annot=True, cmap='coolwarm',fmt='.4f',annot_kws={"size": 6},ax=ax1)
sns.heatmap(group2.corr(),annot=True, cmap='coolwarm',fmt='.4f',annot_kws={"size": 6},ax=ax2)
ax1.set_title('<Red Wine Feature Heatmap>')
ax1.set_xticklabels(ax1.get_xticklabels(),fontsize=6)
ax1.set_yticklabels(ax1.get_yticklabels(),fontsize=6)
ax2.set_title('<White Wine Feature Heatmap>')
ax2.set_xticklabels(ax2.get_xticklabels(),fontsize=6)
ax2.set_yticklabels(ax2.get_yticklabels(),fontsize=6)
plt.show()

# Red 와인과 White 와인별로 품질에 따른 평균 알코올 값을 막대 그래프로 표현
fig2=plt.figure(figsize=(15,5))
ax1=fig2.add_subplot(1,2,1)
ax2=fig2.add_subplot(1,2,2)
sns.barplot(x='quality',y='alcohol',data=group1,ax=ax1,estimator=np.mean,hue='quality')
sns.barplot(x='quality',y='alcohol',data=group2,ax=ax2,estimator=np.mean,hue='quality')
ax1.set_title('<Red Wine Quality-Alcohol Graph>')
ax2.set_title('<White Wine Quality-Alcohol Graph>')
ax1.legend(title='Quality')
ax2.legend(title='Quality')
plt.show()

# Red 와인과 White 와인별로 품질에 따른 황산염의 값을 회귀선이 있는 산점도로 표현
fig3=plt.figure(figsize=(15,15))
ax1=fig3.add_subplot(1,2,1)
ax2=fig3.add_subplot(1,2,2)
sns.regplot(x='quality',y='sulphates',data=group1,ax=ax1,color='orange', label='Red Wine', scatter_kws={'s': 15})
sns.regplot(x='quality',y='sulphates',data=group2,ax=ax2,color='green', label='White Wine', scatter_kws={'s': 15})
ax1.set_title('<Red Wine Quality-Sulphates Scatter plot>')
ax2.set_title('<White Wine Quality-Sulphates Scatter plot>')
ax1.legend(title='Quality')
ax2.legend(title='Quality')
plt.show()
