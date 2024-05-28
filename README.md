# Data_Mining_Project
24-1 데이터마이닝 수업 프로젝트

## 배경/ 필요성
30대 여성의 경제활동 참가율이 증가하며 일, 가정 양립에 대한 지원을 지속할 필요성이 대두되었다.

하지만 현재 시행 중인 정책들은 실질적으로 돌봄 서비스가 필요한 초등학생에겐 미비한 수준이며 이마저 돌봄 교사의 부재, 예산 문제, 돌봄 공간 확대의 필요성 등의 문제들이 있어 정책의 실효성에 대한 의문이 제기되고 있다.

## 분석 목적
클러스터링을 통해 지역을 구분하고 해당 지역이 공통으로 가지고 있는 지역적 특성을 판단한다. 

이를 통해 현재 돌봄 서비스 수요의 원인을 고려하여 우선적으로 돌봄 서비스가 확충되어야 할 지역들을 선별하고 교사 채용, 예산 분배 등의 문제에 의사결정을 지원하고자 한다.

## 데이터
데이터의 범위는 전국으로 수집하며 구분은 총 229개의 시/군/구 단위로 한다.
데이터 수집은 전국적인 데이터가 모여있는 한국통계청, 국가통계포털을 이용하고 필요시 각 도청의 개별 데이터셋을 취합하여 전국 데이터로 만들며 2020년 통계를 기준으로 한다.

데이터는 각 지역의 소득 수준을 파악하기 위해 평균 과세/아파트 비율을 사용하고
초등학생들이 머무를 수 있는 공간을 파악하기 위해 학원 수/도서관 수/PC방 수/학교 숲 면적을 선정하였으며
학부모의 돌봄 여부를 판단하기 위해 맞벌이 비율을 사용하였다.
또한 초등학생에게 집중된 인프라라는 측면에서 소아과 수를 선정하였다.

각 지역의 면적과 인구수에 의해 데이터가 왜곡하지 않도록 초등학생 수로 나누어 비율을 사용하였으며
이상점 또한 해당 지역의 지역적 특성으로 인지하고 포함하여 사용하였다.

전처리
결측치 처리 및 변수 변환을 통해 분석에 적합한 형태로 데이터를 전처리했습니다.

## 분석
###1. EDA
   
        1.1 heatmap 출력
        1.2 feature distribution
        1.3 scatter plot
        1.4 box plot
        1.5 scaling

#### 1.1 heatmap 출력
```python
from IPython.display import display, HTML
import seaborn as sns
import random
import matplotlib.pyplot as plt
import plotly.express as px
import matplotlib.font_manager as fm
import pandas as pd
import numpy as np

font_path = 'C:\Windows\Fonts\MALGUNBD.TTF'
font_prop = fm.FontProperties(fname=font_path)
plt.rc('font', family=font_prop.get_name())
plt.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 깨짐 방지

def eda(df):
    print("==================================================================")
    print("1. Dataframe Shape: ", df.shape)
    print("==================================================================")
    print("2. Explore the Data: ")
    display(HTML(df.head(5).to_html()))
    print("==================================================================")
    print("3. Information on the Data: ")
    data_info_df = pd.DataFrame(df.dtypes, columns=['data type'])
    data_info_df['Duplicated_Values'] = df.duplicated().sum()
    data_info_df['Missing_Values'] = df.isnull().sum().values
    data_info_df['%Missing'] = df.isnull().sum().values / len(df) * 100
    data_info_df['Unique_Values'] = df.nunique().values
    df_desc = df.describe(include='all').transpose()
    data_info_df['Count'] = df_desc['count'].values
    data_info_df['Mean'] = df_desc['mean'].values
    data_info_df['STD'] = df_desc['std'].values
    data_info_df['Min'] = df_desc['min'].values
    data_info_df['Max'] = df_desc['max'].values
    data_info_df = data_info_df[['Count', 'Mean', 'STD', 'Min', 'Max', 'Duplicated_Values', 'Missing_Values',
                                 '%Missing', 'Unique_Values']]
    display(HTML(data_info_df.to_html()))
    print("==================================================================")
    print("4. Correlation Matrix Heatmap - For Numeric Variables:")
    num_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    correlation_matrix = df[num_cols].corr()
    plt.figure(figsize=(12, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5, fmt='.2f', annot_kws={"size": 10, "color": "black"})
    plt.show()
    print("==================================================================")
```
### 1.2 feature distribution
```import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 원본 데이터프레임의 컬럼 분포 확인
columns_to_visualize = df.columns

plt.figure(figsize=(15, 10))
for i, col in enumerate(columns_to_visualize, 1):
    plt.subplot(3, 3, i)
    sns.histplot(df[col], bins=30, kde=True)
    plt.title(f'{col}의 분포')
    plt.tight_layout()

plt.show()

# 로그 변환
df_log_transformed = df.apply(lambda x: np.log1p(x))

# 로그 변환된 데이터프레임의 컬럼 분포 확인
plt.figure(figsize=(15, 10))
for i, col in enumerate(df_log_transformed.columns, 1):
    plt.subplot(3, 3, i)
    sns.histplot(df_log_transformed[col], bins=30, kde=True)
    plt.title(f'{col}의 로그 변환 후 분포')
    plt.tight_layout()

plt.show()
```
### 1.3 scatter plot
```# 산점도 행렬 그리기
sns.pairplot(df)
plt.suptitle('Scatter Plot Matrix', y=1.02)
plt.show()
```

### 1.4 box plot
```# 박스플롯 그리기
plt.figure(figsize=(16, 10))

for i, column in enumerate(df_log_transformed.columns, 1):
    plt.subplot(2, 4, i)  # 2행 4열의 서브플롯에 그래프를 배치
    sns.boxplot(data=df[column])
    plt.title(column)

plt.tight_layout()
plt.show()
```

### 1.5 scaling
```from sklearn.preprocessing import RobustScaler, MinMaxScaler, StandardScaler

standard_scaler = StandardScaler()
standard_scaled = standard_scaler.fit_transform(df)

robust_scaler = RobustScaler()
robust_scaled = robust_scaler.fit_transform(df)

# MinMaxScaler를 사용한 스케일링
minmax_scaler = MinMaxScaler()
minmax_scaled = minmax_scaler.fit_transform(df)
```

2. clustering 알고리즘(K- means, K-medoids, DBSCAN, agglomerative, Mean shift)별  각 feature pair 
### K-Means
```
import itertools

standard_scaler = StandardScaler()
df_scaled = standard_scaler.fit_transform(df)

# 모든 피처 쌍 생성
feature_combinations = list(itertools.combinations(df.columns, 2))

# 클러스터링 및 시각화
plots_per_page = 4  # 한 페이지에 표시할 플롯 수
num_pages = (len(feature_combinations) + plots_per_page - 1) // plots_per_page

for page in range(num_pages):
    plt.figure(figsize=(15, 15))
    start_index = page * plots_per_page
    end_index = min(start_index + plots_per_page, len(feature_combinations))
    for i, (feat1, feat2) in enumerate(feature_combinations[start_index:end_index], 1):
        # 피처 쌍 선택
        X = df_scaled[:, [df.columns.get_loc(feat1), df.columns.get_loc(feat2)]]
        
        # KMeans 클러스터링 수행
        kmeans = KMeans(n_clusters=5, random_state=0).fit(X)
        labels = kmeans.labels_
        
        # 클러스터링 결과 시각화
        plt.subplot(2, 2, i)
        sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=labels, palette='viridis')
        plt.xlabel(feat1)
        plt.ylabel(feat2)
        plt.title(f'Clustering of {feat1} and {feat2}')
        plt.tight_layout()
    
    plt.suptitle(f'Page {page + 1}')
    plt.show()
```
### K-Medoids
```from sklearn_extra.cluster import KMedoids
# 모든 피처 쌍 생성
feature_combinations = list(itertools.combinations(df.columns, 2))

# 클러스터링 및 시각화
plots_per_page = 4  # 한 페이지에 표시할 플롯 수
num_pages = (len(feature_combinations) + plots_per_page - 1) // plots_per_page

for page in range(num_pages):
    plt.figure(figsize=(15, 15))
    start_index = page * plots_per_page
    end_index = min(start_index + plots_per_page, len(feature_combinations))
    for i, (feat1, feat2) in enumerate(feature_combinations[start_index:end_index], 1):
        # 피처 쌍 선택
        X = df_scaled[:, [df.columns.get_loc(feat1), df.columns.get_loc(feat2)]]
        
        # KMedoids 클러스터링 수행
        kmedoids = KMedoids(n_clusters=5, random_state=0).fit(X)
        labels = kmedoids.labels_
        
        # 클러스터링 결과 시각화
        plt.subplot(2, 2, i)
        sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=labels, palette='viridis')
        plt.xlabel(feat1)
        plt.ylabel(feat2)
        plt.title(f'Clustering of {feat1} and {feat2}')
        plt.tight_layout()
    
    plt.suptitle(f'Page {page + 1}')
    plt.show()
```
### DBSCAN
```from sklearn.cluster import DBSCAN
from sklearn.model_selection import ParameterGrid

# 모든 피처 쌍 생성
feature_combinations = list(itertools.combinations(df.columns, 2))

# 데이터 스케일링
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df)

# 모든 피처 쌍 생성
feature_combinations = list(itertools.combinations(df.columns, 2))

# 클러스터링 및 시각화
plots_per_page = 4  # 한 페이지에 표시할 플롯 수
num_pages = (len(feature_combinations) + plots_per_page - 1) // plots_per_page

for page in range(num_pages):
    plt.figure(figsize=(15, 15))
    start_index = page * plots_per_page
    end_index = min(start_index + plots_per_page, len(feature_combinations))
    for i, (feat1, feat2) in enumerate(feature_combinations[start_index:end_index], 1):
        # 피처 쌍 선택
        X = df_scaled[:, [df.columns.get_loc(feat1), df.columns.get_loc(feat2)]]
        
        # DBSCAN 클러스터링 수행
        dbscan = DBSCAN(eps=0.5, min_samples=5).fit(X)
        labels = dbscan.labels_
        
        # 클러스터링 결과 시각화
        plt.subplot(2, 2, i)
        sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=labels, palette='viridis', legend=None)
        plt.xlabel(feat1)
        plt.ylabel(feat2)
        plt.title(f'Clustering of {feat1} and {feat2} (DBSCAN)')
        plt.tight_layout()
    
    plt.suptitle(f'Page {page + 1}')
    plt.show()
```
### Agglomerative
```from sklearn.cluster import AgglomerativeClustering
# 모든 피처 쌍 생성
feature_combinations = list(itertools.combinations(df.columns, 2))

# 클러스터링 및 시각화
plots_per_page = 4  # 한 페이지에 표시할 플롯 수
num_pages = (len(feature_combinations) + plots_per_page - 1) // plots_per_page

for page in range(num_pages):
    plt.figure(figsize=(15, 15))
    start_index = page * plots_per_page
    end_index = min(start_index + plots_per_page, len(feature_combinations))
    for i, (feat1, feat2) in enumerate(feature_combinations[start_index:end_index], 1):
        # 피처 쌍 선택
        X = df_scaled[:, [df.columns.get_loc(feat1), df.columns.get_loc(feat2)]]
        
        # Agglomerative 클러스터링 수행
        agglomerative = AgglomerativeClustering(n_clusters=5)
        labels = agglomerative.fit_predict(X)
        
        # 클러스터링 결과 시각화
        plt.subplot(2, 2, i)
        sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=labels, palette='viridis')
        plt.xlabel(feat1)
        plt.ylabel(feat2)
        plt.title(f'Clustering of {feat1} and {feat2}')
        plt.tight_layout()
    
    plt.suptitle(f'Page {page + 1}')
    plt.show()
```
### MeanShift
```from sklearn.cluster import MeanShift, estimate_bandwidth
# 모든 피처 쌍 생성
feature_combinations = list(itertools.combinations(df.columns, 2))

# 그리드 서치를 위한 파라미터 그리드 설정
param_grid = {
    'bandwidth': np.linspace(0.1, 1.0, 10)
}

# 최적의 파라미터를 찾는 함수
def find_best_params(X, param_grid):
    best_score = -1
    best_params = None
    for params in ParameterGrid(param_grid):
        mean_shift = MeanShift(bandwidth=params['bandwidth'])
        labels = mean_shift.fit_predict(X)
        if len(set(labels)) > 1:
            score = silhouette_score(X, labels)
            if score > best_score:
                best_score = score
                best_params = params
    return best_params

# 클러스터링 및 시각화
plots_per_page = 4  # 한 페이지에 표시할 플롯 수
num_pages = (len(feature_combinations) + plots_per_page - 1) // plots_per_page

for page in range(num_pages):
    plt.figure(figsize=(15, 15))
    start_index = page * plots_per_page
    end_index = min(start_index + plots_per_page, len(feature_combinations))
    for i, (feat1, feat2) in enumerate(feature_combinations[start_index:end_index], 1):
        # 피처 쌍 선택
        X = df_scaled[:, [df.columns.get_loc(feat1), df.columns.get_loc(feat2)]]
        
        # 최적의 하이퍼파라미터 찾기
        best_params = find_best_params(X, param_grid)
        mean_shift = MeanShift(bandwidth=best_params['bandwidth'])
        labels = mean_shift.fit_predict(X)
        
        # 클러스터링 결과 시각화
        plt.subplot(2, 2, i)
        sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=labels, palette='viridis', legend=None)
        plt.xlabel(feat1)
        plt.ylabel(feat2)
        plt.title(f'Clustering of {feat1} and {feat2} (Mean Shift)')
        plt.tight_layout()
    
    plt.suptitle(f'Page {page + 1}')
    plt.show()
```
3. 전체 데이터로 clustering
  
       3.1 K-means 알고리즘의 scaler 별 실루엣 계수, dunn index 파악
           3.1.1 K-means , standard scaler 사용해 optimal k 로 clustering 2차원, 3차원(PCA 이용) plotting
           3.1.2  K-means , Robust scaler 사용해 optimal k 로 clustering 2차원, 3차원(PCA 이용) plotting
           3.1.3 K-means , Minmax scaler 사용해 optimal k 로 clustering 2차원, 3차원(PCA 이용) plotting

        3.2 K-Medoids 알고리즘 사용 시 scaler 별 실루엣 계수, dunn index 파악
           3.2.1 K-Medoids, standard scaler 사용해 optimal k 로 clustering 2차원, 3차원(PCA 이용) plotting
           3.2.2 K-Medoids, Robust scaler 사용해 optimal k 로 clustering 2차원, 3차원(PCA 이용) plotting
           3.2.3 K-Medoids, Minmax scaler 사용해 optimal k 로 clustering 2차원, 3차원(PCA 이용) plotting

       3.3 DBSCAN 사용 시 알고리즘 사용 시 scaler 별 실루엣 계수, dunn index 파악
           3.3.1 DBSCAN , standard scaler 사용해 optimal eps와 min_samples 로 cluster plotting (with PCA)
           3.3.2 DBSCAN , Robust scaler 사용해 optimal eps와 min_samples 로 cluster plotting (with PCA)
           3.3.3 DBSCAN, Minmax scaler 사용해 optimal eps와 min_samples 로 cluster plotting (with PCA)

        3.4 Mean Shifts 사용 시 scaler 별 실루엣 계수, dunn index 파악
           3.4.1 Mean Shifts, standard scaler 사용해 optimal band_width로 cluster plotting (with PCA)
           3.4.2 Mean Shifts, Robust scaler 사용해 optimal band_width로 cluster plotting (with PCA)
           3.4.3 Mean Shifts, Minmax scaler 사용해 optimal band_width로 cluster plotting (with PCA)

        3.5 Agglomerative clustering 사용 시 scaler 별 실루엣 계수, 던 인덱스, dendrogram -> k도출 및 pca
           3.5.1 Agglomerative, standard scaler 사용, dendrogram 통해 optimal한 distance로 k 값 설정해서 cluster plotting (with PCA)
           3.5.2  Agglomerative, Robust scaler 사용, dendrogram 통해 optimal한 distance로 k 값 설정해서 cluster plotting (with PCA)
           3.5.3  Agglomerative, Minmax scaler 사용, dendrogram 통해 optimal한 distance로 k 값 설정해서 cluster plotting (with PCA)

4. 최종 모델 선택
   
       4.1 standard scaler 이용한 Agglomerative clustering 선택
           4.1.1실루엣계수, 던 인덱스가 다른 모델들과 비교 했을 때 비슷한 점수를 가지면서 PCA를 통해 봤을 때 가장 적절하게 나뉘었다고 판단 + tableaur를 통해 본 cluster 결과가 가장 납득되는 결과였음
       4.2 최종 모델 학습 및 결과 저장
   

## 결과
시군구별로 각 특징에 대해 상/중상/중/하 의 등급을 매겼다.

등급은 각 특징에 대해 백분위를 사용하여 25% 50% 75%로 나누었으며

각 클러스터에 포함된 지역을 모아 특징마다 등급의 빈도수를 측정하여 해당 클러스터의 대표 등급을 선정하였다.


* 군집 별 해석

| 항목     | 클러스터 1 | 클러스터 2 | 클러스터 3 | 클러스터 4 |
|----------|-------------|-------------|-------------|-------------|
| 아파트   | 중          | 상          | 하          | 상          |
| 과세     | 하          | 상          | 중          | 상          |
| 맞벌이   | 중상        | 하          | 상          | 하          |
| PC방     | 상          | 중          | 하          | 하          |
| 도서관   | 중상        | 하          | 상          | 중          |
| 소아과   | 하          | 중상        | 하          | 상          |
| 학교 숲 | 상          | 하          | 상          | 중상        |
| 학원     | 중상        | 중상        | 하          | 상          |

<img width="300" alt="클러스터2" src="https://github.com/koojahyeob/Data_Mining_Project/assets/155933613/cb3865c3-7049-4b6b-ae99-388de603065a">
<img width="300" alt="클러스터4" src="https://github.com/koojahyeob/Data_Mining_Project/assets/155933613/f830acd2-ff6a-4f5d-b7c2-372457022dc8">

**Cluster 2, 4 - 가장 잘 사는 지역 과세**
- 맞벌이 비율은 낮지만 과세(소득), 아파트 등이 매우 높다. → 소득 수준이 높다고 판단된다.
- 높은 소아과의 비율로 보아 인프라 구축이 잘 되어 있다.
- pc방 도서관 학원등 편의시설 → 방과후 시간 이후 보낼 수 있는 상호보완적 관계이며 다 합해서 어느 정도 적당히 충족되어있다고 판단된다.
  → 따라서 돌봄 교실 확대 우선순위 대상이 아니다.

<img width="300" alt="클러스터3" src="https://github.com/koojahyeob/Data_Mining_Project/assets/155933613/6e51d7b9-e3fc-4b7b-8a0a-169439258e44">

**Cluster 3**
- 맞벌이 비율은 높지만, 회사보단 1차 산업에 종사하며 맞벌이 하는 경우가 많아 맞벌이로 인한 돌봄공백이 적용되지 않는다.
- 그러나 아파트 비율, 소득이 상대적으로 낮다.
- 모든 구성 지역이 군 단위이며 지역적 특성에 따라 초등학생의 수가 현저히 적다.
  → 따라서 늘봄 교실 확대 대상의 후보로 생각할 수 있다.

<img width="300" alt="클러스터1" src="https://github.com/koojahyeob/Data_Mining_Project/assets/155933613/dab67b5f-0863-4d47-8d2a-521c37175ca6">

**Cluster 1**
- 맞벌이 비율(중상)은 높지만 과세는 낮다 (하) → 소득 수준이 낮다고 판단된다.
- 소아과가 아예 없는 지역들도 있다 → 인프라가 낙후되어 있다.
- 학원 비율은 높은 반면 소득이 낮으므로 사교육비 부담이 클 것이다.
- 한 학년에 초등학생 수도 100명 200명씩 있고, 이에 따라서 돌봄 교실 확대 니즈가 많이 필요하다고 판단된다.
  → 따라서 늘봄 교실 확대 대상의 후보로 생각할 수 있다.

**Cluster 1 VS Cluster3**
- 클러스터 3은 1에 비해 높은 소득을 가지고 있다.
- 클러스터 1은 3에 비해 근소하게 인프라 부분 점수가 앞선다.
- 해석 부분에서 클러스터 3의 높은 맞벌이 비율을 무효화하였으며, 클러스터 1이 앞서는 인프라 부분 중 높은 학원 비율은 오히려 사교육비 부담을 초래함.
- 맞벌이로 인한 돌봄 공백을 인프라 부분보다 중요하게 생각하므로, 실질적으로 높은 맞벌이 + 낮은 소득을 보이는
  → 클러스터 1을 늘봄교실 확대 우선 지역으로 선정한다.

**현황과의 비교 판단**

교육부에서 제공한 2020학년도 온종일돌봄(초등돌봄) 시설 현황에 따르면
Cluster 1지역에 대한 돌봄 교실 이용률은 최소 6.27% 최대 40.6% 평균 22.78% 이다.

또한 2020 돌봄교실 학부모 수요 현황을 보면 수요조사 돌봄교실 이용 희망 비율이 전국 단위 기준 40% 정도이다.
이에 따라 가장 낮은 이용률을 가진 지역에 대해서 가장 큰 우선순위 배정 도달할 수 있도록 각 지자체는 적절한 자원 배분 정책을 실시하여 돌봄 서비스 공급 부족 문제를 줄여나갈 수 있다.

## 한계점 및 추후 개선 방안
- 현황 확보의 어려움
2020년 돌봄교실 이용 현황을 기준으로 데이터를 수집하려 했으나, 소아과 및 PC방과 같은 일부 데이터의 연도를 일치시키지 못했다.
- 돌봄 -> 늘봄 교실로의 전환
정부 정책의 패러다임이 변화하면서 기존의 현황과는 다른 양상으로 공급이 확대될 것이므로 현황이 미래를 잘 반영하지 못 할 가능성이 존재한다.
- 다른 클러스터의 이상치 고려 X
최종적으로 선정된 클러스터에서 현황과 비교했을 때, 높은 이용률을 보이고 있는 지역들을 다음 우선순위 클러스터의 가장 시급한 지역보다 시급하다고 단정지을 수 없다.
