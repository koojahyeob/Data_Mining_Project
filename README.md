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

## 분석(code 부분)(일단 보류)
1. EDA 
    1. heatmap 출력
    2. feature distribution
    3. scatter plot
    4. box plot
    5. 스케일링

2. 알고리즘 별 clustering(K- means, K-medoids, DBSCAN, agglomerative, Mean shift) 각 feature pair 

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
        3.5.1 Agglomerative, standard scaler 사용해 optimal한 distance로 k 값 설정해서 cluster plotting (with PCA)
        3.5.2  Agglomerative, Robust scaler 사용해 optimal한 distance로 k 값 설정해서 cluster plotting (with PCA)
        3.5.3  Agglomerative, Minmax scaler 사용해 optimal한 distance로 k 값 설정해서 cluster plotting (with PCA)

4. 최종 모델 선택
   4.1 standard scaler 이용한 Agglomerative clustering 선택
        <실루엣계수, 던 인덱스가 다른 모델들과 비교 했을 때 비슷한 점수를 가지면서 PCA를 통해 봤을 때 가장 적절하게 나뉘었다고 판단 + tableaur를 통해 본 cluster 결과가 가장 납득되는 결과였음>
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
- Cluster 1과 비교
- 맞벌이 비율은 높지만, 회사보단 1차 산업에 종사하며 맞벌이 하는 경우가 많아 맞벌이로 인한 돌봄공백이 적용되지 않는다.
- 지역 자체의 특성이 시골이며 초등학생의 수가 현저히 적다.
- 돌봄 서비스 자체가 적은 규모로 이루어지고 있으며 돌봄 인력 추가시 얻을 효용이 적다
  → 따라서 돌봄 교실 확대 우선순위 대상이 아니다.

<img width="300" alt="클러스터1" src="https://github.com/koojahyeob/Data_Mining_Project/assets/155933613/dab67b5f-0863-4d47-8d2a-521c37175ca6">

**Cluster 1**
- 맞벌이 비율(중상)은 높지만 과세는 낮다 (하) → 소득 수준이 낮다고 판단된다.
- 소아과가 아예 없는 지역들도 있다 → 인프라가 낙후되어 있다.
- 한 학년에 초등학생 수도 100명 200명씩 있고, 이에 따라서 돌봄 교실 확대 니즈가 많이 필요하다고 판단된다.
- pc방(상), 도서관(중상), 학원(중상)
  → 비록 방과후 시간 이후 보낼 수 있는 시설이 많지만 맞벌이로 인한 돌봄 공백을 더 주요한 특징으로 보고 있으며 이는 Cluster 3에 비해 높다.
  → 따라서 돌봄 교실 확대 우선순위 지역이다.

**현황과의 비교 판단**

교육부에서 제공한 2020학년도 온종일돌봄(초등돌봄) 시설 현황에 따르면
Cluster 1지역에 대한 돌봄 교실 이용률은 최소 6.27% 최대 40.6% 평균 22.78% 이다.

또한 2020 돌봄교실 학부모 수요 현황을 보면 수요조사 돌봄교실 이용 희망 비율이 전국 단위 기준 40% 정도이다.
이에 따라 가장 낮은 이용률을 가진 지역에 대해서 가장 큰 우선순위 배정 도달할 수 있도록 각 지자체는 적절한 자원 배분 정책을 실시하여 돌봄 서비스 공급 부족 문제를 줄여나갈 수 있다.

## 한계점 및 추후 개선 방안(일단 보류)
