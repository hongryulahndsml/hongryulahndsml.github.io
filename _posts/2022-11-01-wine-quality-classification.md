---
layout: post
title: "와인 품질 분류 머신러닝"
author: "Hongryul Ahn"
tags: MachineLearning
---

https://dacon.io/competitions/open/235610/codeshare/4221

Goal - seperate Red wine and White wine!

# 개요

와인의 품질을 예측하는 분류 지도학습 인공지능 모델링입니다.

https://dacon.io/competitions/open/235610/codeshare/4221
를 참고하여 변형하였습니다.

# LIBRARY


```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
%matplotlib inline

plt.style.use('fivethirtyeight')
```

# DATA 불러오기


```python
#Load Data!
train= pd.read_csv("train.csv", index_col=None)
test = pd.read_csv("test.csv", index_col=None)
submission= pd.read_csv("sample_submission.csv")
```


```python
train
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>index</th>
      <th>quality</th>
      <th>fixed acidity</th>
      <th>volatile acidity</th>
      <th>citric acid</th>
      <th>residual sugar</th>
      <th>chlorides</th>
      <th>free sulfur dioxide</th>
      <th>total sulfur dioxide</th>
      <th>density</th>
      <th>pH</th>
      <th>sulphates</th>
      <th>alcohol</th>
      <th>type</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>5</td>
      <td>5.6</td>
      <td>0.695</td>
      <td>0.06</td>
      <td>6.8</td>
      <td>0.042</td>
      <td>9.0</td>
      <td>84.0</td>
      <td>0.99432</td>
      <td>3.44</td>
      <td>0.44</td>
      <td>10.2</td>
      <td>white</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>5</td>
      <td>8.8</td>
      <td>0.610</td>
      <td>0.14</td>
      <td>2.4</td>
      <td>0.067</td>
      <td>10.0</td>
      <td>42.0</td>
      <td>0.99690</td>
      <td>3.19</td>
      <td>0.59</td>
      <td>9.5</td>
      <td>red</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>5</td>
      <td>7.9</td>
      <td>0.210</td>
      <td>0.39</td>
      <td>2.0</td>
      <td>0.057</td>
      <td>21.0</td>
      <td>138.0</td>
      <td>0.99176</td>
      <td>3.05</td>
      <td>0.52</td>
      <td>10.9</td>
      <td>white</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>6</td>
      <td>7.0</td>
      <td>0.210</td>
      <td>0.31</td>
      <td>6.0</td>
      <td>0.046</td>
      <td>29.0</td>
      <td>108.0</td>
      <td>0.99390</td>
      <td>3.26</td>
      <td>0.50</td>
      <td>10.8</td>
      <td>white</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>6</td>
      <td>7.8</td>
      <td>0.400</td>
      <td>0.26</td>
      <td>9.5</td>
      <td>0.059</td>
      <td>32.0</td>
      <td>178.0</td>
      <td>0.99550</td>
      <td>3.04</td>
      <td>0.43</td>
      <td>10.9</td>
      <td>white</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>5492</th>
      <td>5492</td>
      <td>5</td>
      <td>7.7</td>
      <td>0.150</td>
      <td>0.29</td>
      <td>1.3</td>
      <td>0.029</td>
      <td>10.0</td>
      <td>64.0</td>
      <td>0.99320</td>
      <td>3.35</td>
      <td>0.39</td>
      <td>10.1</td>
      <td>white</td>
    </tr>
    <tr>
      <th>5493</th>
      <td>5493</td>
      <td>6</td>
      <td>6.3</td>
      <td>0.180</td>
      <td>0.36</td>
      <td>1.2</td>
      <td>0.034</td>
      <td>26.0</td>
      <td>111.0</td>
      <td>0.99074</td>
      <td>3.16</td>
      <td>0.51</td>
      <td>11.0</td>
      <td>white</td>
    </tr>
    <tr>
      <th>5494</th>
      <td>5494</td>
      <td>7</td>
      <td>7.8</td>
      <td>0.150</td>
      <td>0.34</td>
      <td>1.1</td>
      <td>0.035</td>
      <td>31.0</td>
      <td>93.0</td>
      <td>0.99096</td>
      <td>3.07</td>
      <td>0.72</td>
      <td>11.3</td>
      <td>white</td>
    </tr>
    <tr>
      <th>5495</th>
      <td>5495</td>
      <td>5</td>
      <td>6.6</td>
      <td>0.410</td>
      <td>0.31</td>
      <td>1.6</td>
      <td>0.042</td>
      <td>18.0</td>
      <td>101.0</td>
      <td>0.99195</td>
      <td>3.13</td>
      <td>0.41</td>
      <td>10.5</td>
      <td>white</td>
    </tr>
    <tr>
      <th>5496</th>
      <td>5496</td>
      <td>6</td>
      <td>7.0</td>
      <td>0.350</td>
      <td>0.17</td>
      <td>1.1</td>
      <td>0.049</td>
      <td>7.0</td>
      <td>119.0</td>
      <td>0.99297</td>
      <td>3.13</td>
      <td>0.36</td>
      <td>9.7</td>
      <td>white</td>
    </tr>
  </tbody>
</table>
<p>5497 rows × 14 columns</p>
</div>



학습 데이터셋 변수 설명 

(https://dacon.io/competitions/open/235610/data 참고)

* index 구분자
* quality 품질
* fixed acidity 산도
* volatile acidity 휘발성산
* citric acid 시트르산
* residual sugar 잔당 : 발효 후 와인 속에 남아있는 당분
* chlorides 염화물
* free sulfur dioxide 독립 이산화황
* total sulfur dioxide 총 이산화황
* density 밀도
* pH 수소이온농도
* sulphates 황산염
* alcohol 도수
* type 종류


```python
test
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>index</th>
      <th>fixed acidity</th>
      <th>volatile acidity</th>
      <th>citric acid</th>
      <th>residual sugar</th>
      <th>chlorides</th>
      <th>free sulfur dioxide</th>
      <th>total sulfur dioxide</th>
      <th>density</th>
      <th>pH</th>
      <th>sulphates</th>
      <th>alcohol</th>
      <th>type</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>9.0</td>
      <td>0.31</td>
      <td>0.48</td>
      <td>6.60</td>
      <td>0.043</td>
      <td>11.0</td>
      <td>73.0</td>
      <td>0.99380</td>
      <td>2.90</td>
      <td>0.38</td>
      <td>11.6</td>
      <td>white</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>13.3</td>
      <td>0.43</td>
      <td>0.58</td>
      <td>1.90</td>
      <td>0.070</td>
      <td>15.0</td>
      <td>40.0</td>
      <td>1.00040</td>
      <td>3.06</td>
      <td>0.49</td>
      <td>9.0</td>
      <td>red</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>6.5</td>
      <td>0.28</td>
      <td>0.27</td>
      <td>5.20</td>
      <td>0.040</td>
      <td>44.0</td>
      <td>179.0</td>
      <td>0.99480</td>
      <td>3.19</td>
      <td>0.69</td>
      <td>9.4</td>
      <td>white</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>7.2</td>
      <td>0.15</td>
      <td>0.39</td>
      <td>1.80</td>
      <td>0.043</td>
      <td>21.0</td>
      <td>159.0</td>
      <td>0.99480</td>
      <td>3.52</td>
      <td>0.47</td>
      <td>10.0</td>
      <td>white</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>6.8</td>
      <td>0.26</td>
      <td>0.26</td>
      <td>2.00</td>
      <td>0.019</td>
      <td>23.5</td>
      <td>72.0</td>
      <td>0.99041</td>
      <td>3.16</td>
      <td>0.47</td>
      <td>11.8</td>
      <td>white</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>995</th>
      <td>995</td>
      <td>7.1</td>
      <td>0.59</td>
      <td>0.02</td>
      <td>2.30</td>
      <td>0.082</td>
      <td>24.0</td>
      <td>94.0</td>
      <td>0.99744</td>
      <td>3.55</td>
      <td>0.53</td>
      <td>9.7</td>
      <td>red</td>
    </tr>
    <tr>
      <th>996</th>
      <td>996</td>
      <td>8.7</td>
      <td>0.15</td>
      <td>0.30</td>
      <td>1.60</td>
      <td>0.046</td>
      <td>29.0</td>
      <td>130.0</td>
      <td>0.99420</td>
      <td>3.22</td>
      <td>0.38</td>
      <td>9.8</td>
      <td>white</td>
    </tr>
    <tr>
      <th>997</th>
      <td>997</td>
      <td>8.8</td>
      <td>0.66</td>
      <td>0.26</td>
      <td>1.70</td>
      <td>0.074</td>
      <td>4.0</td>
      <td>23.0</td>
      <td>0.99710</td>
      <td>3.15</td>
      <td>0.74</td>
      <td>9.2</td>
      <td>red</td>
    </tr>
    <tr>
      <th>998</th>
      <td>998</td>
      <td>7.0</td>
      <td>0.42</td>
      <td>0.19</td>
      <td>2.30</td>
      <td>0.071</td>
      <td>18.0</td>
      <td>36.0</td>
      <td>0.99476</td>
      <td>3.39</td>
      <td>0.56</td>
      <td>10.9</td>
      <td>red</td>
    </tr>
    <tr>
      <th>999</th>
      <td>999</td>
      <td>8.5</td>
      <td>0.21</td>
      <td>0.26</td>
      <td>9.25</td>
      <td>0.034</td>
      <td>73.0</td>
      <td>142.0</td>
      <td>0.99450</td>
      <td>3.05</td>
      <td>0.37</td>
      <td>11.4</td>
      <td>white</td>
    </tr>
  </tbody>
</table>
<p>1000 rows × 13 columns</p>
</div>



테스트 데이터셋은 quality 변수가 제외되어 있음


```python
submission
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>index</th>
      <th>quality</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>995</th>
      <td>995</td>
      <td>0</td>
    </tr>
    <tr>
      <th>996</th>
      <td>996</td>
      <td>0</td>
    </tr>
    <tr>
      <th>997</th>
      <td>997</td>
      <td>0</td>
    </tr>
    <tr>
      <th>998</th>
      <td>998</td>
      <td>0</td>
    </tr>
    <tr>
      <th>999</th>
      <td>999</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>1000 rows × 2 columns</p>
</div>



서브미션 파일은 테스트 데이터셋에 대한 quality를 예측하여 제출해야 함


```python
#drop index column
train= train.drop(['index'],axis=1)
test= test.drop(['index'],axis=1)
train.shape, test.shape, submission.shape
```




    ((5497, 13), (1000, 12), (1000, 2))



# 탐색적 데이터 분석(EDA)

## pandas의 기본 요약 함수 활용 확인하기


```python
train.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 5497 entries, 0 to 5496
    Data columns (total 13 columns):
     #   Column                Non-Null Count  Dtype  
    ---  ------                --------------  -----  
     0   quality               5497 non-null   int64  
     1   fixed acidity         5497 non-null   float64
     2   volatile acidity      5497 non-null   float64
     3   citric acid           5497 non-null   float64
     4   residual sugar        5497 non-null   float64
     5   chlorides             5497 non-null   float64
     6   free sulfur dioxide   5497 non-null   float64
     7   total sulfur dioxide  5497 non-null   float64
     8   density               5497 non-null   float64
     9   pH                    5497 non-null   float64
     10  sulphates             5497 non-null   float64
     11  alcohol               5497 non-null   float64
     12  type                  5497 non-null   object 
    dtypes: float64(11), int64(1), object(1)
    memory usage: 558.4+ KB
    


```python
train.describe(include='all')
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>quality</th>
      <th>fixed acidity</th>
      <th>volatile acidity</th>
      <th>citric acid</th>
      <th>residual sugar</th>
      <th>chlorides</th>
      <th>free sulfur dioxide</th>
      <th>total sulfur dioxide</th>
      <th>density</th>
      <th>pH</th>
      <th>sulphates</th>
      <th>alcohol</th>
      <th>type</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>5497.000000</td>
      <td>5497.000000</td>
      <td>5497.000000</td>
      <td>5497.000000</td>
      <td>5497.000000</td>
      <td>5497.000000</td>
      <td>5497.000000</td>
      <td>5497.000000</td>
      <td>5497.000000</td>
      <td>5497.000000</td>
      <td>5497.000000</td>
      <td>5497.000000</td>
      <td>5497</td>
    </tr>
    <tr>
      <th>unique</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2</td>
    </tr>
    <tr>
      <th>top</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>white</td>
    </tr>
    <tr>
      <th>freq</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>4159</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>5.818992</td>
      <td>7.210115</td>
      <td>0.338163</td>
      <td>0.318543</td>
      <td>5.438075</td>
      <td>0.055808</td>
      <td>30.417682</td>
      <td>115.566491</td>
      <td>0.994673</td>
      <td>3.219502</td>
      <td>0.530524</td>
      <td>10.504918</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.870311</td>
      <td>1.287579</td>
      <td>0.163224</td>
      <td>0.145104</td>
      <td>4.756676</td>
      <td>0.034653</td>
      <td>17.673881</td>
      <td>56.288223</td>
      <td>0.003014</td>
      <td>0.160713</td>
      <td>0.149396</td>
      <td>1.194524</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>min</th>
      <td>3.000000</td>
      <td>3.800000</td>
      <td>0.080000</td>
      <td>0.000000</td>
      <td>0.600000</td>
      <td>0.009000</td>
      <td>1.000000</td>
      <td>6.000000</td>
      <td>0.987110</td>
      <td>2.740000</td>
      <td>0.220000</td>
      <td>8.000000</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>5.000000</td>
      <td>6.400000</td>
      <td>0.230000</td>
      <td>0.250000</td>
      <td>1.800000</td>
      <td>0.038000</td>
      <td>17.000000</td>
      <td>78.000000</td>
      <td>0.992300</td>
      <td>3.110000</td>
      <td>0.430000</td>
      <td>9.500000</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>6.000000</td>
      <td>7.000000</td>
      <td>0.290000</td>
      <td>0.310000</td>
      <td>3.000000</td>
      <td>0.047000</td>
      <td>29.000000</td>
      <td>118.000000</td>
      <td>0.994800</td>
      <td>3.210000</td>
      <td>0.510000</td>
      <td>10.300000</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>6.000000</td>
      <td>7.700000</td>
      <td>0.400000</td>
      <td>0.390000</td>
      <td>8.100000</td>
      <td>0.064000</td>
      <td>41.000000</td>
      <td>155.000000</td>
      <td>0.996930</td>
      <td>3.320000</td>
      <td>0.600000</td>
      <td>11.300000</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>max</th>
      <td>9.000000</td>
      <td>15.900000</td>
      <td>1.580000</td>
      <td>1.660000</td>
      <td>65.800000</td>
      <td>0.610000</td>
      <td>289.000000</td>
      <td>440.000000</td>
      <td>1.038980</td>
      <td>4.010000</td>
      <td>2.000000</td>
      <td>14.900000</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



피쳐들의 스케일(단위)가 다르기 때문에 추후에'표준화' 진행합니다. 

e.g. ```density```와 ```pH```는 단위가 다르죠 :)

## pandas_profiling 라이브러리 활용하여 확인하기


```python
#import pandas_profiling as pp
#report = train.profile_report()
#report.to_file("profiling.html")
```

pandas profiling 으로 간단하게 훑어봅니다.

* Overview에서 각 피쳐, 상관관계, 결측값 등의 정보를 얻을 수 있습니다. 
    * 각 피쳐의 Toggle detais를 눌러 상세히 봅니다.
* Warnings(28) 탭에서 상관관계에 대한 정보를 얻을 수 있습니다.

## 직접 시각화해서 확인하기


```python
print(train['quality'].value_counts())
sns.countplot(x=train['quality']);
plt.title("frequency of quality", fontsize=20);
```

    6    2416
    5    1788
    7     924
    4     186
    8     152
    3      26
    9       5
    Name: quality, dtype: int64
    


    
![png](https://raw.githubusercontent.com/hongryulahndsml/hongryulahndsml.github.io/master/_images/2022-11-01-wine-quality-classification/output_23_1.png)
    


6등급의 와인이 가장 많이 있네요!


```python
#distribution by 'quality'
numerical_columns = train.select_dtypes(exclude='object').columns.tolist()
numerical_columns.remove('quality')
def show_dist_plot(df, columns):
    for column in columns:
        f, ax = plt.subplots(1,2,figsize=(16,4))
        sns.stripplot(x=df['quality'],y=df[column], ax=ax[0],hue=df['quality'])
        sns.violinplot(data=df, x='quality', y=column, ax=ax[1])
        
show_dist_plot(train, numerical_columns)
```


    
![png](https://raw.githubusercontent.com/hongryulahndsml/hongryulahndsml.github.io/master/_images/2022-11-01-wine-quality-classification/output_25_0.png)
    



    
![png](https://raw.githubusercontent.com/hongryulahndsml/hongryulahndsml.github.io/master/_images/2022-11-01-wine-quality-classification/output_25_1.png)
    



    
![png](https://raw.githubusercontent.com/hongryulahndsml/hongryulahndsml.github.io/master/_images/2022-11-01-wine-quality-classification/output_25_2.png)
    



    
![png](https://raw.githubusercontent.com/hongryulahndsml/hongryulahndsml.github.io/master/_images/2022-11-01-wine-quality-classification/output_25_3.png)
    



    
![png](https://raw.githubusercontent.com/hongryulahndsml/hongryulahndsml.github.io/master/_images/2022-11-01-wine-quality-classification/output_25_4.png)
    



    
![png](https://raw.githubusercontent.com/hongryulahndsml/hongryulahndsml.github.io/master/_images/2022-11-01-wine-quality-classification/output_25_5.png)
    



    
![png](https://raw.githubusercontent.com/hongryulahndsml/hongryulahndsml.github.io/master/_images/2022-11-01-wine-quality-classification/output_25_6.png)
    



    
![png](https://raw.githubusercontent.com/hongryulahndsml/hongryulahndsml.github.io/master/_images/2022-11-01-wine-quality-classification/output_25_7.png)
    



    
![png](https://raw.githubusercontent.com/hongryulahndsml/hongryulahndsml.github.io/master/_images/2022-11-01-wine-quality-classification/output_25_8.png)
    



    
![png](https://raw.githubusercontent.com/hongryulahndsml/hongryulahndsml.github.io/master/_images/2022-11-01-wine-quality-classification/output_25_9.png)
    



    
![png](https://raw.githubusercontent.com/hongryulahndsml/hongryulahndsml.github.io/master/_images/2022-11-01-wine-quality-classification/output_25_10.png)
    


모든 피쳐들의 의미를 헤아리고, 유의미한 피쳐를 찾는 것은 분석에 있어 큰 도움이 됩니다.<br>
그러나, 피쳐의 수가 50개가 넘는다면? 모든 피쳐들을 헤아리기 힘들겠죠! <br>
그래서 피쳐들이 많을 때 ```train.corr()``` 상관관계를 통해서 힌트를 얻곤 합니다.


```python
plt.figure(figsize=(18,8))
corr= train.corr()
sns.heatmap(corr, annot=True, square=False, vmin=-.6, vmax=1.0);
```


    
![png](https://raw.githubusercontent.com/hongryulahndsml/hongryulahndsml.github.io/master/_images/2022-11-01-wine-quality-classification/output_27_0.png)
    


**주의! 상관관계와 인관관계는 다릅니다 **  <br>

분포에서의 관계가 있음을 알려주는 것이지, 원인과 결과의 관계는 아니라는 것! 기억해주세요 :)

관련 문서: https://ko.wikipedia.org/wiki/상관관계와_인과관계

# 데이터 전처리


```python
from sklearn.preprocessing import LabelEncoder, StandardScaler

#Standardscaler
ss= StandardScaler()
ss.fit(train[numerical_columns])
train[numerical_columns] = ss.transform(train[numerical_columns])
test[numerical_columns] = ss.transform(test[numerical_columns])

#factorize
le = LabelEncoder()
le.fit(train['type'])
train['type'] = le.transform(train['type'])
test['type'] = le.transform(test['type'])
```


```python
train.head(3)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>quality</th>
      <th>fixed acidity</th>
      <th>volatile acidity</th>
      <th>citric acid</th>
      <th>residual sugar</th>
      <th>chlorides</th>
      <th>free sulfur dioxide</th>
      <th>total sulfur dioxide</th>
      <th>density</th>
      <th>pH</th>
      <th>sulphates</th>
      <th>alcohol</th>
      <th>type</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5</td>
      <td>-1.250611</td>
      <td>2.186377</td>
      <td>-1.78194</td>
      <td>0.286345</td>
      <td>-0.398500</td>
      <td>-1.211937</td>
      <td>-0.560852</td>
      <td>-0.117252</td>
      <td>1.372128</td>
      <td>-0.605988</td>
      <td>-0.255287</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>5</td>
      <td>1.234899</td>
      <td>1.665574</td>
      <td>-1.23056</td>
      <td>-0.638755</td>
      <td>0.322998</td>
      <td>-1.155351</td>
      <td>-1.307080</td>
      <td>0.738864</td>
      <td>-0.183584</td>
      <td>0.398147</td>
      <td>-0.841348</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>5</td>
      <td>0.535849</td>
      <td>-0.785265</td>
      <td>0.49250</td>
      <td>-0.722855</td>
      <td>0.034399</td>
      <td>-0.532907</td>
      <td>0.398583</td>
      <td>-0.966732</td>
      <td>-1.054782</td>
      <td>-0.070450</td>
      <td>0.330774</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
test.head(3)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>fixed acidity</th>
      <th>volatile acidity</th>
      <th>citric acid</th>
      <th>residual sugar</th>
      <th>chlorides</th>
      <th>free sulfur dioxide</th>
      <th>total sulfur dioxide</th>
      <th>density</th>
      <th>pH</th>
      <th>sulphates</th>
      <th>alcohol</th>
      <th>type</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.390243</td>
      <td>-0.172555</td>
      <td>1.112802</td>
      <td>0.244295</td>
      <td>-0.369640</td>
      <td>-1.098765</td>
      <td>-0.756293</td>
      <td>-0.289803</td>
      <td>-1.988209</td>
      <td>-1.007643</td>
      <td>0.916835</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4.730148</td>
      <td>0.562696</td>
      <td>1.802026</td>
      <td>-0.743880</td>
      <td>0.409577</td>
      <td>-0.872422</td>
      <td>-1.342614</td>
      <td>1.900261</td>
      <td>-0.992553</td>
      <td>-0.271277</td>
      <td>-1.259963</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-0.551562</td>
      <td>-0.356368</td>
      <td>-0.334569</td>
      <td>-0.050055</td>
      <td>-0.456220</td>
      <td>0.768567</td>
      <td>1.127044</td>
      <td>0.042025</td>
      <td>-0.183584</td>
      <td>1.067570</td>
      <td>-0.925071</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



StandardScaler를 통해서 **표준화** 작업을 진행하였고, <br>
type을 0과 1로 변환해주었습니다. ML에서는 str은 들어가지 않으니 변환은 꼭! <br>
이외에도 encoding에는 ```pd.get_dummies()```, ```labelEncoder``` 등이 있습니다.

# 모델링(학습)

## 라이브러리


```python
#Library
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import plot_roc_curve,accuracy_score, confusion_matrix, plot_confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import accuracy_score
```

## 데이터셋 구조화(X, Y 나누기)


```python
X = train.drop(['quality'],axis=1)
y = train.quality
```


```python
X
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>fixed acidity</th>
      <th>volatile acidity</th>
      <th>citric acid</th>
      <th>residual sugar</th>
      <th>chlorides</th>
      <th>free sulfur dioxide</th>
      <th>total sulfur dioxide</th>
      <th>density</th>
      <th>pH</th>
      <th>sulphates</th>
      <th>alcohol</th>
      <th>type</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-1.250611</td>
      <td>2.186377</td>
      <td>-1.781940</td>
      <td>0.286345</td>
      <td>-0.398500</td>
      <td>-1.211937</td>
      <td>-0.560852</td>
      <td>-0.117252</td>
      <td>1.372128</td>
      <td>-0.605988</td>
      <td>-0.255287</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.234899</td>
      <td>1.665574</td>
      <td>-1.230560</td>
      <td>-0.638755</td>
      <td>0.322998</td>
      <td>-1.155351</td>
      <td>-1.307080</td>
      <td>0.738864</td>
      <td>-0.183584</td>
      <td>0.398147</td>
      <td>-0.841348</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.535849</td>
      <td>-0.785265</td>
      <td>0.492500</td>
      <td>-0.722855</td>
      <td>0.034399</td>
      <td>-0.532907</td>
      <td>0.398583</td>
      <td>-0.966732</td>
      <td>-1.054782</td>
      <td>-0.070450</td>
      <td>0.330774</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-0.163201</td>
      <td>-0.785265</td>
      <td>-0.058879</td>
      <td>0.118145</td>
      <td>-0.283060</td>
      <td>-0.080221</td>
      <td>-0.134436</td>
      <td>-0.256620</td>
      <td>0.252016</td>
      <td>-0.204334</td>
      <td>0.247051</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.458177</td>
      <td>0.378883</td>
      <td>-0.403491</td>
      <td>0.854020</td>
      <td>0.092119</td>
      <td>0.089537</td>
      <td>1.109276</td>
      <td>0.274305</td>
      <td>-1.117010</td>
      <td>-0.672931</td>
      <td>0.330774</td>
      <td>1</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>5492</th>
      <td>0.380505</td>
      <td>-1.152890</td>
      <td>-0.196724</td>
      <td>-0.870030</td>
      <td>-0.773678</td>
      <td>-1.155351</td>
      <td>-0.916198</td>
      <td>-0.488900</td>
      <td>0.812072</td>
      <td>-0.940700</td>
      <td>-0.339010</td>
      <td>1</td>
    </tr>
    <tr>
      <th>5493</th>
      <td>-0.706906</td>
      <td>-0.969078</td>
      <td>0.285733</td>
      <td>-0.891055</td>
      <td>-0.629379</td>
      <td>-0.249978</td>
      <td>-0.081134</td>
      <td>-1.305196</td>
      <td>-0.370269</td>
      <td>-0.137392</td>
      <td>0.414497</td>
      <td>1</td>
    </tr>
    <tr>
      <th>5494</th>
      <td>0.458177</td>
      <td>-1.152890</td>
      <td>0.147888</td>
      <td>-0.912080</td>
      <td>-0.600519</td>
      <td>0.032951</td>
      <td>-0.400946</td>
      <td>-1.232194</td>
      <td>-0.930325</td>
      <td>1.268397</td>
      <td>0.665666</td>
      <td>1</td>
    </tr>
    <tr>
      <th>5495</th>
      <td>-0.473889</td>
      <td>0.440154</td>
      <td>-0.058879</td>
      <td>-0.806955</td>
      <td>-0.398500</td>
      <td>-0.702665</td>
      <td>-0.258808</td>
      <td>-0.903684</td>
      <td>-0.556954</td>
      <td>-0.806815</td>
      <td>-0.004118</td>
      <td>1</td>
    </tr>
    <tr>
      <th>5496</th>
      <td>-0.163201</td>
      <td>0.072529</td>
      <td>-1.023793</td>
      <td>-0.912080</td>
      <td>-0.196480</td>
      <td>-1.325109</td>
      <td>0.061004</td>
      <td>-0.565220</td>
      <td>-0.556954</td>
      <td>-1.141527</td>
      <td>-0.673902</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>5497 rows × 12 columns</p>
</div>




```python
y
```




    0       5
    1       5
    2       5
    3       6
    4       6
           ..
    5492    5
    5493    6
    5494    7
    5495    5
    5496    6
    Name: quality, Length: 5497, dtype: int64



## 데이터셋 구조화(훈련/검증 데이터셋 나누기)


```python
X_train, X_validation, y_train, y_validation = train_test_split(X,y,test_size=.2, random_state=42, stratify=y)
X_train.shape, y_train.shape, X_validation.shape, y_validation.shape
```




    ((4397, 12), (4397,), (1100, 12), (1100,))



* ```test_size : 0.2```   train과 test를 8:2로 구분한다는 의미!
* ```random_state : 42``` 같은 값으로 나오게 하기 위한 Seed 설정! 

X_train과 X_validation에서 type을 제외한 12개의 Feature이 있는 것을 볼 수 있습니다.

## 모델 학습(훈련 데이터셋)


```python
#RandomForest
model = RandomForestClassifier()
model.fit(X_train,y_train)
```




    RandomForestClassifier()



## 모델 검증(검증 데이터셋)


```python
# 예측 결과 시각화
plot_confusion_matrix(model,X_validation,y_validation,cmap='OrRd')
```

    /package/anaconda3.9/lib/python3.9/site-packages/sklearn/utils/deprecation.py:87: FutureWarning: Function plot_confusion_matrix is deprecated; Function `plot_confusion_matrix` is deprecated in 1.0 and will be removed in 1.2. Use one of the class methods: ConfusionMatrixDisplay.from_predictions or ConfusionMatrixDisplay.from_estimator.
      warnings.warn(msg, category=FutureWarning)
    




    <sklearn.metrics._plot.confusion_matrix.ConfusionMatrixDisplay at 0x72c93416bb80>




    
![png](https://raw.githubusercontent.com/hongryulahndsml/hongryulahndsml.github.io/master/_images/2022-11-01-wine-quality-classification/output_48_2.png)
    


5~7등급을 예측하는데 실제로 그러한 것이 높은 것을 볼 수 있네요.


```python
# 예측 결과 점수 평가
import sklearn

ypred_validation = model.predict(X_validation)
print("Validation Accuracy Score : ", accuracy_score(y_validation,ypred_validation))
```

    Validation Accuracy Score :  0.6563636363636364
    

## 모델 예측(시험 데이터셋)


```python
ypred_test = model.predict(test)
```


```python
# submission 파일에 저장
submission['quality'] = ypred_test
submission.to_csv("submission.csv",index=False)
```

이상 기초적인 와인 품질 분류였습니다. 감사합니다.

---

추가)앙상블 학습

<b>정형 데이터를 다루는 데 가장 뛰어난 성과를 내는 알고리즘

> RandomForest

![](https://miro.medium.com/max/1678/1*Wf91XObaX2zwow7mMwDmGw.png)

* bootstrap sample : 복원추출로 인한 <b>증복된 샘플 추출
    
기본적으로 bootstrap sample은 훈련 세트의 크기와 같게 만듭니다.

1. RandomForestClassifier : 전체 특성 개수의 제곱근만큼의 특성 선택이 '최선의 분할' <br>
    - 그러나, 회귀모델 RandomForestRegressor은 전체 특성 사용

# 특성 중요도 분석


```python
#분류에 있어 각 피쳐에 대한 중요도 출력
print(model.feature_importances_)
```

    [0.07325887 0.10038532 0.07734029 0.08518867 0.08647417 0.08604095
     0.09318203 0.10286569 0.08203673 0.08375554 0.12573259 0.00373917]
    

하나의 특성에 과도하게 집중하지 않고 더 많은 특성이 훈련에 기여할 기회를 얻는다면 일반화 가능성을 높혀줍니다.


> 이외에, 엑스트라 트리 (ExtraTreesClassifier)

1. 기본적으로 100개의 결정 트리 훈련
2. 부트스트랩 샘플을 사용하지 않습니다. = 전체 훈련 세트를 사용
3. 노드 분할 시, 가장 좋은 분할을 찾는 것이 아닌, 무작위로 분할 = DecisionTreeClassifier 내 splitter = 'random' 지정
