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
1. EDA
   
        1.1 heatmap 출력
        1.2 feature distribution
        1.3 scatter plot
        1.4 box plot
        1.5 scaling

- EDA를 진행하여 feature들 간의 상관관계, 분포 및 이상치를 확인해 보았다.
- 분포의 비대칭성을 해결하기 위해 로그 변환을 수행하였으나 차이가 미미하고 Clustering 결과의 차이가 없어 기존의 데이터를 활용하기로 하였다.
- 초등학생 수 대비 학교 숲 비율에 진천군 데이터가 이상치를 보였으나 학교 근처의 숲과 논이 많은 즉, 지역적 특성임을 고려하여 제거하지 않고 사용하였다.
- 데이터들의 분포 및 값의 범위가 다르기 때문에 scaling을 진행하였고 다양한 scaler 중에 실루엣 스코어를 척도한 결과 최종적으로 standard scaler 사용하기로 하였다.

2. clustering 알고리즘(K- means, K-medoids, DBSCAN, agglomerative, Mean shift)별  각 feature pair 

- 데이터를 다각도로 분석하고 가장 적합한 클러스터링 알고리즘과 feature 조합을 예상해볼 수 있도록 K-means, K-medoids, DBSCAN, Mean Shift, Agglomerative 클러스터링을 비교하였다.

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

- 다양한 알고리즘에 대해 다양한 스케일링 방법을 사용하여 클러스터링 결과를 비교함으로써, 최적의 데이터 전처리 방법을 결정하고, 이를 통해 가장 효과적인 클러스터링 결과를 도출한다. 실루엣 계수와 Dunn Index를 사용하여 클러스터링 품질을 정량적으로 평가하고, 최적의 k 값을 찾는 과정을 통해 데이터의 특성에 가장 적합한 클러스터링 설정을 선택한다.

4. 최종 모델 선택
Standard Scaler로 스케일링한 후, k=4인 Agglomerative Clustering 모델을 선택 하였다.

**이유**
- PCA를 통해 3차원으로 축소하여 본 클러스터가 가장 잘 묶임.
- 실루엣 스코어에서 가장 좋은 점수를 얻음.
- 덴드로그램을 통해 시각적으로 클러스터가 잘 묶였음을 확인.
- 태블로를 통해 클러스터 결과를 시각화한 결과 가장 납득할만 했음.

**다른 모델과의 비교**
1. standard Scaling
- K-means와 K-medoids: 실루엣, Dunn Index 점수가 Agglomerative에 비해 낮음.
- DBSCAN, Mean Shift: PCA로 3차원 축소 후 클러스터링 결과가 좋지 않음.

2. Robust Scaling
- 전체적으로 모든 모델이 PCA 통해서 3차원 축소해본 결과가 별로 좋지 않았음.

3. MinMax Scaling
- K-medoids가 PCA로 본 결과 좋았으나, 태블로로 시각화한 결과 Agglomerative만큼 설명력이 높지 않음.


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

######      Cluster 2 - 고소득 중형 도시군        Cluster 4 - 메트로폴리스 군집

**Cluster 2, 4 - 가장 잘 사는 지역 과세**
- 맞벌이 비율은 낮지만 과세(소득), 아파트 등이 매우 높다. → 소득 수준이 높다고 판단된다.
- 높은 소아과의 비율로 보아 인프라 구축이 잘 되어 있다.
- pc방 도서관 학원등 편의시설 → 방과후 시간 이후 보낼 수 있는 상호보완적 관계이며 다 합해서 어느 정도 적당히 충족되어있다고 판단된다.
  → 따라서 돌봄 교실 확대 우선순위 대상이 아니다.

<img width="300" alt="클러스터3" src="https://github.com/koojahyeob/Data_Mining_Project/assets/155933613/6e51d7b9-e3fc-4b7b-8a0a-169439258e44">

######     Cluster 3 - 평균 소득 농업 군집
**Cluster 3**
- 맞벌이 비율은 높지만, 회사보단 1차 산업에 종사하며 맞벌이 하는 경우가 많아 맞벌이로 인한 돌봄공백이 적용되지 않는다.
- 그러나 아파트 비율, 소득이 상대적으로 낮다.
- 모든 구성 지역이 군 단위이며 지역적 특성에 따라 초등학생의 수가 현저히 적다.
  → 따라서 늘봄 교실 확대 대상의 후보로 생각할 수 있다.

<img width="300" alt="클러스터1" src="https://github.com/koojahyeob/Data_Mining_Project/assets/155933613/dab67b5f-0863-4d47-8d2a-521c37175ca6">

######     Cluster 1 - 도농 혼합 저소득 군집
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

## 한계점
1. 현황 확보의 어려움
2020년 돌봄교실 이용 현황을 기준으로 데이터를 수집하려 했으나, 소아과 및 PC방과 같은 일부 데이터의 연도를 일치시키지 못했다.
2. 돌봄 -> 늘봄 교실로의 전환
정부 정책의 패러다임이 변화하면서 기존의 현황과는 다른 양상으로 공급이 확대될 것이므로 현황이 미래를 잘 반영하지 못 할 가능성이 존재한다.
3. 클러스터링 해석의 어려움
클러스터 알고리즘 특성 상 적절한 군집 수를 설정하는 게 어렵고 또한 명확한 기준을 가지고 해석하기가 어렵다.

## 추후 개선 방안
- 최신 자료의 데이터를 통해서 새롭게 분석을 진행함으로써 현재 시행되는 정책들에 대해서 발 빠르게 대처할 수 있다.
- 해석력을 높이기 위한 도메인 전문가와의 협업함으로써 합리적인 결과를 도출해낼 수 있다.
