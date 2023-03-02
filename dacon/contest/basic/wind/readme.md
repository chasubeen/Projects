# **0. 대회 소개**
- 기후 데이터로부터 풍력 발전량을 예측하는 AI 알고리즘 개발
- 풍력 발전량을 **회귀** 예측
- 다양한 회귀 모형 활용
- 예측 성능 평가 기준 학습

# **1. 데이터 준비**
- id: 샘플 별 고유 id
- temperature: 기온(°C)
- pressure: 기압(hPa)
- humidity: 습도(%)
- wind_speed: 풍속(m/s)
- wind_direction: 풍향(degree)
- precipitation: 1시간 강수량(mm)
- snowing: 눈 오는 상태 여부(False, True)
- cloudiness: 흐림 정도(%)
- target: 풍력 발전량(GW) **(목표 예측값)**  

# **2. EDA & 데이터 전처리**
### **2-0. 데이터 전처리란?**
- 결측치 처리, 이상치 제거, 데이터 단위 변환, 데이터 분포 변환 등 
  - 데이터를 정확하게 분석하기 위해 먼저 데이터에 여러 가지 처리를 해주는 것
- 전처리를 함으로써 데이터 분석이 가능하도록 하며, 데이터를 합치거나 나눠서 더 정확한 정보를 갖도록 해줌
  - 전처리 과정은 데이터 분석에 있어서 반드시 필요한 부분
- 데이터를 계산하는 컴퓨터는 오로지 숫자(정수, 실수)만을 인식
  - 한국어나 영어와 같은 문자나 비어있는 값(결측치) 등을 숫자로 변경해 주어야 함 => **Encoding**  

### **2-1. 랜덤 시드 고정**
- 매번 고정된 결과를 얻기 위해 사용함
- seed를 고정하지 않는 경우 같은 코드임에도 매번 다른 결과가 도출됨
  - 항상 동일한 결과를 얻기 위해 사용
```Python
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

seed_everything(42) # Seed 고정
```

### **2-2. 양적 변수 vs 질적 변수**
- 양적(quantitative) 변수
  - 변수의 값을 숫자로 나타낼 수 있는 변수 ex> 키, 몸무게, 소득 등
  - 이산변수와 연속변수로 구분
    - 이산(discrete) 변수: 하나하나 셀 수 있는 변수 ex> 정수
    - 연속(continuous) 변수: 이산변수와 다르게 변수의 값 사이에 무수히 많은 또 다른 값들이 존재하는 경우 ex> 실수
- 질적(qualitative) 변수
  - 변수의 값이 비수치적 특정 카테고리에 포함시키도록 하는 변수 ex> 색상, 성별, 종교
    - 명목(nomial) 변수: 변수의 값이 특정한 범주에 들어가지만 해당 범주간 순위는 존재하지 않는 경우
    - 순위(ordinal) 변수: 변수의 값이 특정 범주에 들어가면서 변수의 값이 순위를 가지는 경우 

- 현재 프로젝트에서는 'snowing' 컬럼을 제외한 컬럼은 모두 **양적** 변수  

### **2-3. EDA(Exploratory Data Analysis)**
- ```df.info()```를 통해 데이터의 정보 확인
- ```df.describe()```를 통해 양적 변수들의 기술통계량(빈도, 평균, 최대/최소, 사분위값) 파악
- ```df.hist()```를 통해 양적 변수들의 분포를 히스토그램으로 파악
- ```sns.countplot(x = df[columns])```를 통해 질적 변수들의 빈도 시각화
- ```sns.boxpllot(y = df[columns])```를 통해 이상치 탐지(상자 수염 그림)
- ```sns.heatmap(df.corr())```를 통해 변수들 간의 상관계수 시각화
- ```scipy.stats.skew()```를 통해 데이터의 왜곡 정도 

### **2-4. 데이터 전처리 기법들**
**1️⃣ 범주형 변수 전처리**  
  - **라벨 인코딩(Label Encoding)**  
    - 카테고리 피처를 코드형 숫자 값으로 변환하는 것
    - sklearn의 ```LabelEncoder``` 클래스로 구현
    ```Python
    from sklearn.preprocessing import LabelEncoder
    
    items = ['사과', '배', '바나나']
    
    encoder = LabelEncoder() # 객체 생성
    encoder.fit(items) # 학습
    labels = encoder.transform(items) # 라벨 변환(인코딩)
    ```
  
    - ```classes_``` 속성을 통해 문자열 값이 어떤 숫자 값으로 인코딩됐는지 확인할 수 있음
    - ```inverse_transform()```을 통해 인코딩된 값을 다시 디코딩할 수 있음
    
    - 숫자 값의 크고 작음에 대한 특성이 작용할 위험성이 존재 -> 선형 회귀 등의 ML 알고리즘에는 적용을 지양해야 함
      - 숫자의 이러한 특성 반영 해결을 위해 **원-핫 인코딩** 활용
       
  - **원-핫 인코딩(One-hot Encoding)**  
    - 피처 값의 유형에 따라 새로운 피처를 추가해 고유 값에 해당하는 칼럼에만 1을 표시하고 나머지 컬럼에는 0을 표시
    - sklearn의 ```OneHotEncoder``` 클래스로 구현
    - 주의사항
      - OneHotEncoder로 변환 전 모든 문자열 값이 숫자형 값으로 변환돼야 한다는 것
      - 입력 값으로 **2차원** 데이터가 필요
    ```Python
    from sklearn.preprocessing import OneHotEncoder
    import numpy as np
    
    items = ['사과', '배', '바나나']
    
    ### 문자열 값 -> 숫자형 값
    # LabelEncoder 활용
    encoder = LabelEncoder() # 객체 생성
    encoder.fit(items) # 학습
    labels = encoder.transform(items) # 라벨 변환(인코딩)
    
    ### 2차원 데이터로 변환
    labels = labels.reshape(-1, 1)
    
    ### 원-핫 인코딩 적용
    oh_encoder = OneHotEncoder()
    oh_encoder.fit(labels)
    oh_labels = oh_encoder.transform(labels)
    ```
    
    - pandas의 ```get_dummies()```를 활용해 원-핫 인코딩을 간단히 구현할 수 있음
      - 숫자형 값으로의 변환 없이도 바로 변환 가능
       
    ```Python
    import pandas as pd
    
    df = pd.DataFrame({'item':['사과', '배', '바나나']})
    pd.get_dummies(df)
    ```
    
**2️⃣ 이상치/ 데이터 분포에 대한 전처리**  
  - **피처 스케일링(Feature Scaling)**  
    - 서로 다른 변수의 값 범위를 일정한 수준으로 맞추는 작업
    - 종류
      - 표준화(StandardScaler): 개별 피처를 평균이 0이고, 분산이 1인 값으로 변환
      - 정규화(MinMaxScaler): 데이터 값을 0과 1 사이의 범위 값으로 변환
      - 표준정규화(RobustScaler)
    - 주의: 학습 데이터로 fit()이 적용된 스케일링 기준 정보를 그대로 테스트 데이터에 적용해야 함
      - 가능하다면 전체 데이터의 스케일링 변환 후 train, test 데이터 분리
      - 또는 train data에 대해서는 ```fit_transform()```, test data에 대해서는 ```transform()```  
     
  - **데이터 변환(Data Transformation)**  
    - 데이터가 **왜곡된** 분포를 가지는 경우 데이터 변환을 통해 왜곡 정도를 완화할 수 있음
    - 종류
      - 로그 변환(Log Transformation): 좌로 치우쳐진(Positive skew, Left skew) 데이터에 활용 가능
      - 제곱(Square): 우로 치우쳐진(Negative skew, Right skew) 데이터에 활용 가능
         
# **3. 평가 지표**
- **MAE(Mean Absolute Error)**  
  - 예측값과 실제값의 차이에 대한 절대값에 대하여 평균을 낸 값
  - $(\frac{1}{n})\sum_{i=1}^{n}\left | y_{i} - x_{i} \right |$ 
  - 작을수록 좋다.
  ```Python
  from sklearn.metrics import mean_absolute_error
  
  mae = mean_absolute_error(pred, actual)
  ```
# **4. 회귀 모형(Regression Model)**

# **5. 결과 정리**
