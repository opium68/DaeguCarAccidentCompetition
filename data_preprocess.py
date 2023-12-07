#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np

#df = pd.read_csv('./data/external_open/대구 CCTV 정보.csv', encoding='cp949')
df = pd.read_csv('./data/train.csv')


# In[2]:


print(df.dtypes)


# In[3]:


df_ori = df.copy()


# In[4]:


# '사고일시' 열에서 날짜 부분만 추출
df['사고일자'] = pd.to_datetime(df['사고일시']).dt.date

# 각 날짜의 사고 횟수 계산
daily_accidents = df.groupby('사고일자').size()

# '일별사고횟수' 열 생성
df['일별사고횟수'] = df['사고일자'].map(daily_accidents)


# In[5]:


df = df.drop('사고일자', axis=1)


# In[6]:


df['사고일시'] = pd.to_datetime(df['사고일시'])

# '월'만 남기기
df['사고일시'] = df['사고일시'].dt.month

df = df.rename(columns={'사고일시': '사고월'})


# In[7]:


# 요일을 숫자로 매핑
day_to_num = {'월요일': 1, '화요일': 2, '수요일': 3, '목요일': 4, '금요일': 5, '토요일': 6, '일요일': 7}

# '요일' 열의 데이터 변경
df['요일'] = df['요일'].replace(day_to_num)

print(df)


# In[8]:


# for column in df.columns:
#     print(f"{column}:\n{df['가해운전자 연령'].value_counts()}\n")
pd.set_option('display.max_rows', None)
for column in df.columns:
    print(f"{column}:\n{df['가해운전자 연령'].value_counts(dropna=False).sort_index()}\n")


# In[9]:


# 연령대를 반환하는 함수
def age_to_decade(age):
    return (age // 10) * 10

# 연령대 분류를 반환하는 함수
def age_to_category(age):
    if age < 20:
        return '청소년'
    elif age < 40:
        return '청년'
    elif age < 60:
        return '중년'
    else:
        return '노년'


# In[10]:


# '세', '이상', '미분류' 제거 및 int 형식으로 변환
df['피해운전자 연령'] = df['피해운전자 연령'].str.replace('세', '')
df['피해운전자 연령'] = df['피해운전자 연령'].str.replace('이상', '')
# 가장 빈도가 높은 나이 값 얻기
most_frequent_age = df['피해운전자 연령'].mode()[0]

# NaN 값을 가장 빈도가 높은 나이 값으로 변경
df['피해운전자 연령'] = df['피해운전자 연령'].replace('미분류', most_frequent_age)
df['피해운전자 연령'] = df['피해운전자 연령'].replace('nan', np.nan)
df['피해운전자 연령'] = df['피해운전자 연령'].fillna(most_frequent_age).astype(int)


# float 형식을 int 형식으로 변환
df['피해운전자 연령'] = df['피해운전자 연령'].astype(int)

# '가해운전자 연령' 열의 데이터 변경 및 '나이대' 열 추가
df['피해운전자 연령'] = df['피해운전자 연령'].apply(age_to_decade)
df['피해 나이대'] = df['피해운전자 연령'].apply(age_to_category)


# In[11]:


print(df['피해운전자 연령'])
#print(df['피해 나이대'])


# In[12]:


# '세', '이상', '미분류' 제거 및 int 형식으로 변환
df['가해운전자 연령'] = df['가해운전자 연령'].str.replace('세', '')
df['가해운전자 연령'] = df['가해운전자 연령'].str.replace('이상', '')
# 가장 빈도가 높은 나이 값 얻기
most_frequent_age = df['가해운전자 연령'].mode()[0]

# NaN 값을 가장 빈도가 높은 나이 값으로 변경
df['가해운전자 연령'] = df['가해운전자 연령'].replace('미분류', most_frequent_age)
df['가해운전자 연령'] = df['가해운전자 연령'].replace('nan', np.nan)
df['가해운전자 연령'] = df['가해운전자 연령'].fillna(most_frequent_age).astype(int)


# float 형식을 int 형식으로 변환
df['가해운전자 연령'] = df['가해운전자 연령'].astype(int)

# '가해운전자 연령' 열의 데이터 변경 및 '나이대' 열 추가
df['가해운전자 연령'] = df['가해운전자 연령'].apply(age_to_decade)
df['가해 나이대'] = df['가해운전자 연령'].apply(age_to_category)


# In[13]:


print(df['가해운전자 연령'] )


# In[14]:


# 성별 남자 : 0, 여자 : 1

df['가해운전자 성별'] = df['가해운전자 성별'].map({'남': 0, '여': 1})
df['피해운전자 성별'] = df['피해운전자 성별'].map({'남': 0, '여': 1})


# In[15]:


# 남녀 중 남자가 더 많아, 신원미상은 남자로 통일

df['피해운전자 성별'] = df['피해운전자 성별'].fillna(0)
df['가해운전자 성별'] = df['가해운전자 성별'].fillna(0)

for column in df.columns:
    print(f"{column}:\n{df['피해운전자 성별'].value_counts(dropna=False).sort_index()}\n")
    print(f"{column}:\n{df['가해운전자 성별'].value_counts(dropna=False).sort_index()}\n")


# 
# print(df)

# In[16]:


pd.set_option('display.max_columns', None)
print(df.columns)


# In[17]:


pd.set_option('display.max_columns', None)
# 더미 변수로 변환
df = pd.get_dummies(df, columns=['사고월', '요일', '기상상태', '시군구', '도로형태', '노면상태', '사고유형',
                                 '사고유형 - 세부분류', '법규위반', '가해운전자 차종','가해운전자 상해정도',
                                 '피해운전자 차종', '피해운전자 상해정도', '피해 나이대', '가해 나이대'])
print(df)


# In[18]:


pd.set_option('display.max_columns', None)
print(df.columns)


# In[24]:


# csv로 저장
df.to_csv('data_preprocess.csv', index=False)

# excel로 저장
df.to_excel('data_preprocess.xlsx', index=False) 


# In[22]:


# 각 칼럼의 상관계수 저장 후 출력
correlation = df.corr()
print(correlation)


# In[26]:


# excel로 저장
# correlation.to_excel('correlation.xlsx', index=False) 

df_ori.to_excel('df_ori.xlsx', index=False)


# In[ ]:


# 히트맵 사용하여 상관계수 시각화
plt.figure(figsize=(10, 10))  # 그래프 크기 설정
sns.heatmap(data = correlation, annot=True, fmt = '.2f', linewidths=.5, cmap='Blues')  # 히트맵 그리기

plt.show()  # 그래프 출력

