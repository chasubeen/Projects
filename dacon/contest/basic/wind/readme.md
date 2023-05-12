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
- ```sns.boxplot(y = df[columns])```를 통해 이상치 탐지(상자 수염 그림)
- ```sns.heatmap(df.corr())```를 통해 변수들 간의 상관계수 시각화
- ```scipy.stats.skew()```를 통해 데이터의 왜곡 정도 파악
  - 카테고리형 변수(범주형 변수)의 경우 원-핫 인코딩 시 데이터가 왜곡될 가능성이 높음 => ```skew()``` 함수 적용 시에는 원-핫 인코딩이 적용되지 않은 데이터로 

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
### **4-0. 회귀 분석 모델링**
- 단순 선형 회귀(LinearRegression)
- 규제 선형 회귀
  - 릿지(Ridge)
  - 라쏘(Lasso)
  - 엘라스틱넷(ElasticNet): 릿지 + 라쏘
- 회귀 트리
  - 결정 트리
  - 랜덤 포레스트
  - GBM
  - XGBoost
  - LightGBM
- 스태킹 모델
- AutoML

### **4-1. 선형 회귀(LinearRegression)**
- 실제값과 예측값의 RSS(Residual Sum of Squares)를 최소화 해 OLS(Ordinary Least Squares) 추정 방식으로 구현
- 규제를 적용하지 않은 모델
- 코드
```Python
from sklearn.linear_model import linearRegression

model = LinearRegression(n_jobs = -1) # CPU Core를 있는 대로 모두 사용하겠다.
model.fit(X_train, y_train) # 학습
pred = model.predict(X_test) # 예측
mean_absolute_error(pred, y_test) # 평가
```

### **4-2. 규제(Regularization)**
- 학습이 과적합되는 것을 방지하고자 일종의 penalty를 부여하는 것

#### **1) L1 규제**  
- 가중치의 합을 더한 값에 규제 강도를 곱하여 오차에 더한 값($Error=MAE+α|w|$)
- 어떤 가중치는 실제로 0이 됨 -> 모델에서 완전히 제외되는 특성이 발생할 수 있음
- **라쏘(Lasso)** 모델에 적용됨
- 코드
```Python
from sklearn.linear_model import Lasso

model = Lasso(alpha = alpha) # alpha: 규제 강도
model.fit(X_train, y_train) # 학습
pred = model.predict(X_test) # 예측
mean_absolute_error(pred, y_test) # 평가
```

#### **2) L2 규제**  
- 각 가중치 제곱의 합에 규제 강도를 곱한 값($Error=MAE+αw^2$)
- 규제 강도를 크게 하면 가중치가 더 많이 감소되고(규제를 중요시함), 규제 강도를 작게 하면 가중치가 증가함(규제를 중요시하지 않음)
- **릿지(Ridge)** 모델에 적용됨
- 코드
```Python
from sklearn.linear_model import Ridge

model = Ridge(alpha = alpha) # 규제 강도
model.fit(X_train, y_train) # 학습
pred = model.predict(X_test) # 예측
mean_squared_error(pred, y_test) # 평가
```

#### **3) 엘라스틱넷(ElasticNet)**
- L1 규제 + L2 규제
- l1_ratio(default: 0.5) 속성 -> 규제 강도 조정
  - l1_ratio = 0: L2 규제만
  - l1_ratio = 1: L1 규제만
  - 0 < l1_ratio < 1: L1 and L2 규제(혼합 사용)
- 코드
```Python
from sklearn.linear_model import ElasticNet

model = ElasticNet(alpha = alpha,l1_ratio = l1_ratio) # 규제 강도
model.fit(X_train, y_train) # 학습
pred = model.predict(X_test) # 예측
mean_absolute_error(pred, y_test) # 평가
```

### **4-3. 다항 회귀**
- 다항식의 계수 간 상호작용을 통해 새로운 feature를 생성
- 데이터가 단순한 직선의 형태가 아닌 비선형 형태여도 선형 모델을 사용하여 비선형 데이터를 학습할 수 있음
  - 원본 데이터: 선형 관계가 x
  - 새로운 feature: 선형 관계
- 특성의 거듭제곱을 새로운 특성으로 추가하고 확장된 특성을 포함한 데이터 셋에 선형 모델을 학습
- 코드
```Python
from sklearn.preprocessing import PolynomialFeatures

poly = PolyNomialFeatures(degree = 2, include_bias = False)
# degree: 차수(몇 제곱까지 갈 것인가)
# include_bias: 절편 포함 여부 선택
```

#### **파이프라인(PipeLine)**
- 여러 가지 방법들을 융합하는 기법
- 코드
```Python
from sklearn.pipeline import make_pipeline

# 파이프라인 생성(모델 객체 생성)
pipeline = make_pipeline(
  StandarsScaler(),
  ElasticNet(alpha = 0.1, l1_ratio = 0.2)
)

pipeline_pred = pipeline.fit(X_train, y_train).predict(X_test) # 학습, 예측
mean_squared_error(pipeline_pred,y_test) # 평가
```

### **4-4. 회귀 트리**
- 회귀 함수를 기반으로 하지 않고 결정 트리와 같이 **트리**를 기반으로 하는 회귀 방식
  -  회귀를 위한 트리를 생성하고 이를 기반으로 회귀 예측을 하는 것
  -  리프 노드에 속한 데이터 값의 평균값을 구해 회귀 예측값을 계산
- sklearn에서는 결정 트리, 랜덤 포레스트, GBM에서 회귀 수행을 할 수 있는 Estimator 클래스 제공
  - XGBoost, LightGBM도 사이킷런 래퍼 클래스를 통해 제공됨
- 코드
```Python
### 모델 객체 생성
dt_reg = DecisionTreeRegressor(random_state = 0, max_depth = 4)
rf_reg = RandomForestRegressor(n_estimators = 100)
gbm_reg = GradientBoostingRegressor(n_estimators = 100)
xgb_reg = XGBRegressor(n_estimators = 100)
lgbm_reg = LGBMRegressor(n_estimators = 100)


### 학습/예측/평가
models = [rf_reg,gbm_reg,xgb_reg,lgbm_reg]
names = ['RandomForest', 'GradientBoosting', 'XGB','LGBM']
for i in range(len(models)):
    model = models[i]
    model.fit(X_train.values,y_train.values) # 학습
    pred = np.expm1(model.predict(X_valid.values)) # 예측
    mae_eval('{}'.format(names[i]),pred,y_valid) # 평가 & 시각화
```
- ```feature_importances_``` 속성을 통해 피처별 중요도를 파악할 수 있음
- 선형 회귀가 직선으로 예측 회귀선을 표현하는 데 반해, 회귀 트리의 경우 분할되는 데이터 지점에 따라 브랜치를 만들면서 **계단 형태**로 회귀선을 생성

### **4-5. 예측 결과 혼합을 통한 최종 예측**
- A 모델과 B 모델, 두 모델의 예측값이 있다면 A 모델 예측값의 40%, B 모델 예측값의 60%를 더해서 최종 회귀 값으로 예측하는 등의 방법
- 혼합하는 모델들 중 성능이 조금 **좋은** 쪽에 높은 가중치 부여

### **4-6. 스태킹 앙상블**
- 스태킹 모델의 구현 방법  
  1️⃣.개별적인 기반 모델 학습 -> 예측 데이터 생성 -> 각각 스태킹 형태로 결합해 최종 메타 모델의 **feature data** 세트 생성  
  2️⃣. 최종 메타 모델을 통해 최종 회귀 예측값 도출  
- 코드
```Python
### 1. 개별 기반 모델에서 최종 메타 모델이 사용할 학습 및 테스트용 데이터를 생성하기 위한 함수

### 개별 기반 모델에서 최종 메타 모델이 사용할 학습 및 테스트용 데이터를 생성하기 위한 함수 
def get_stacking_base_datasets(model, X_train_n, y_train_n, X_test_n, n_folds):
    # 지정된 n_folds값으로 KFold 생성
    kf = KFold(n_splits = n_folds, shuffle = False)

    #추후에 메타 모델이 사용할 학습 데이터 반환을 위한 넘파이 배열 초기화 
    train_fold_pred = np.zeros((X_train_n.shape[0],1))
    test_pred = np.zeros((X_test_n.shape[0],n_folds))
    print(model.__class__.__name__, ' model 시작 ')
    
    for folder_counter, (train_index, valid_index) in enumerate(kf.split(X_train_n)):

        # 입력된 학습 데이터에서 기반 모델이 학습/ 예측할 폴드 데이터 셋 추출 
        print('\t 폴드 세트: ', folder_counter,' 시작 ')
        X_tr = X_train_n[train_index] 
        y_tr = y_train_n[train_index] 
        X_te = X_train_n[valid_index]  
        
        # 폴드 세트 내부에서 다시 만들어진 학습 데이터로 기반 모델의 학습 수행
        model.fit(X_tr, y_tr)       

        # 폴드 세트 내부에서 다시 만들어진 검증 데이터로 기반 모델 예측 후 데이터 저장
        train_fold_pred[valid_index, :] = model.predict(X_te).reshape(-1,1)

        # 입력된 원본 테스트 데이터를 폴드 세트 내 학습된 기반 모델에서 예측 후 데이터 저장
        test_pred[:, folder_counter] = model.predict(X_test_n)
            
    # 폴드 세트 내에서 원본 테스트 데이터를 예측한 데이터를 평균하여 테스트 데이터로 생성 
    test_pred_mean = np.mean(test_pred, axis = 1).reshape(-1,1)    
    
    # train_fold_pred는 최종 메타 모델이 사용하는 학습 데이터, test_pred_mean은 테스트 데이터
    return train_fold_pred , test_pred_mean
```
```Python
### 각 모델별로 메타 모델이 사용할 데이터 세트 추출

X_train_n = X_train.values
X_valid_n = X_valid.values
y_train_n = y_train.values

poly_ridge = make_pipeline(
        PolynomialFeatures(degree = 2,include_bias = False),
        Ridge(alpha = 0.0001)
)

# 각 개별 기반(Base) 모델이 생성한 학습용/테스트용 데이터 반환
rf_train, rf_valid = get_stacking_base_datasets(rf_reg, X_train_n, y_train_n, X_valid_n, 5)
lgbm_train, lgbm_valid = get_stacking_base_datasets(lgbm_reg, X_train_n, y_train_n, X_valid_n, 5)
xgb_poly_train, xgb_poly_valid = get_stacking_base_datasets(xgb_poly, X_train_n, y_train_n, X_valid_n, 5)
gbm_train, gbm_valid = get_stacking_base_datasets(gbm_reg, X_train_n, y_train_n, X_valid_n, 5)
lr_poly_train, lr_poly_valid = get_stacking_base_datasets(poly_linear, X_train_n, y_train_n, X_valid_n, 5)
ridge_poly_train, ridge_poly_valid = get_stacking_base_datasets(poly_ridge, X_train_n, y_trai
```

```Python
### 2. 최종 메타 모델에 적용

# 개별 모델이 반환한 학습 및 테스트용 데이터 세트를 스태킹 형태로 결합
stack_final_X_train = np.concatenate((rf_train, lgbm_train, xgb_poly_train, 
                                      gbm_train,lr_poly_train,ridge_poly_train ), axis = 1)
stack_final_X_valid = np.concatenate((rf_valid, lgbm_valid, xgb_poly_valid, 
                                      gbm_valid, lr_poly_valid, ridge_poly_valid), axis = 1)

# 최종 메타 모델로 RandomForest 모델 적용
meta_model_rf = RandomForestRegressor(n_estimators = 500)

# 개별 모델 예측값을 기반으로 새롭게 만들어진 학습/테스트 데이터로 메타 모델 예측 및 성능 평가
meta_model_rf.fit(stack_final_X_train,y_train)
final = meta_model_rf.predict(stack_final_X_valid)
mae = mean_absolute_error(y_valid,final)
print("스태킹 회귀 모델의 최종 성능은: ", mae)
```

### **4-7. AutoML(Automated Machine Learning)**
- 현재의 머신러닝 모델링은 Machine Learning Process 동안 많은 시간과 노력을 요구
  - Machine Learning Process: 문제 정의 과정, 데이터 수집, 전처리, 모델 학습 및 평가, 서비스 적용 
- AutoML은 기계 학습 파이프라인에서 수작업과 반복되는 작업을 자동화하는 프로세스
  - 머신러닝을 자동화하는 AI 기술
- AutoML systems
  - AutoWEKA
  - Auto-sklearn
  - Auto-Pytorch

#### **▶ PyCaret**
- AutoML 구현을 가능하게 해주는 파이썬 라이브러리
- 적은 코드로 머신 러닝을 구현할 수 있음 => Low-code machine learning
- scikit-learn 패키지 기반
- 분류/회귀/군집화 등 다양한 모델 지원
- 단계
  - 각 반복마다
    - setup
    - model 비교
    - 다양한 방법으로 모델 생성/저장
    - 모델 혼합
    - 모델 stacking  
- [PyCaret API](https://pycaret.readthedocs.io/en/latest/index.html)

# **5. 결과 정리**
- 범주형 변수의 경우 인코딩 진행(LabelEncoding, One-hot Encoding)
- target 변수의 경우 데이터 왜곡 정도가 심함 => 모두 로그 변환 수행

### **✅ Case 1**
- 피처 데이터에 대한 처리
  - 피처 스케일링
  - 데이터 변환(로그 변환)
  
| |ver 1|ver 2|ver 3|
|-------|-------|-------|-------|
|**범주형 변수**|LabelEncoding|LabelEncoding|LabelEncoding|
|**피처 변환**|로그 변환|변환 x|로그 변환|
|**피처 스케일링**|RobustScaler|RobustScaler|StandardScaler|
|**최종 선택 모델**|Poly LinearRegression|Poly ElasticNet|Poly ElasticNet|
|**MAE**|2.3460|2.3417|2.3271|

### **✅ Case 2**
- 범주형 변수의 인코딩 방식을 Label Encoding에서 One-hot Encoding으로 변경
  - 선형 회귀 등의 ML 알고리즘에서는 숫자 값의 크고 작음이 하나의 특성으로 작용할 위험성 존재
  
| |ver 4|ver 5|
|-------|-------|-------|
|**범주형 변수**|One-hot Encoding|One-hot Encoding|
|**피처 변환**|로그 변환|변환 x|
|**피처 스케일링**|RobustScaler|RobustScaler|
|**최종 선택 모델**|Poly LinearRegression|Poly ElasticNet|
|**MAE**|0.3867|2.3361|

### **✅ Case 3**
- 상관도가 높은 feature의 이상치 제거

| |ver 6|
|-------|-------|
|**범주형 변수**|One-hot Encoding|
|**피처 변환**|변환 x|
|**피처 스케일링**|StandardScaler|
|**최종 선택 모델**|Poly ElasticNet|
|**MAE**|2.2805|

### **✅ Case 4**
- 회귀 트리 적용

- 범주형 변수: One-hot Encoding 진행
- 이상치 제거
- 피처 스케일링의 경우 모두 StandardScaler 적용
- target 데이터의 경우 모두 로그 변환 적용
- 최종 회귀 모형 선택 시 성능이 좋은 두 개의 모델을 선택하여 혼합 모델 생성

| |ver 7|ver 8|ver 9|
|-------|-------|-------|-------|
|**이상치 처리**|모든 feature|상관도가 높은 feature|상관도가 높은 feature|
|**피처 변환**|로그 변환|로그 변환|변환 x|
|**최종 선택 모델**|Poly RandomForest + Poly LGBM|RandomForest + Poly LGBM|RandomForest + LGBM|
|**MAE**|1.9692|2.0590|2.0777|

### **✅ Case 5**
- 컬럼 제거
  - 왜곡 정도가 높은 컬럼 중 target 변수와 상관도가 낮은 컬럼 제거
- 상관도가 높은 feature의 이상치 제거
- 피처 데이터의 경우 왜곡 정도가 심하면 로그 변환 수행
- target 데이터의 경우 로그 변환 수행

| |ver 10|ver 11|ver 12|
|-------|-------|-------|-------|
|**제외 컬럼**|pressure|pressure, snowing|pressure, snowing|
|**피처 스케일링**|StandardScaler|Standard|RobustScaler|
|**최종 선택 모델**|RandomForest|RandomForest|Poly RandomForest + Poly LGBM|
|**MAE**|2.1720|0.3643|0.3633|

### **✅ Case 6**
- 스태킹 앙상블 적용

- 컬럼 제거
  - 왜곡 정도가 높은 컬럼 중 target 변수와 상관도가 낮은 컬럼 제거
- 상관도가 높은 feature의 이상치 제거
- 피처 데이터의 경우 왜곡 정도가 심하면 로그 변환 수행
- target 데이터의 경우 로그 변환 수행

| |ver 13|
|-------|-------|
|**제외 컬럼**|pressure, snowing|
|**피처 스케일링**|StandardScaler|
|**MAE**|0.3741|

### **✅ Case 7**
- AutoML 적용
  - PyCaret 적용
 - 컬럼 제거
  - 왜곡 정도가 높은 컬럼 중 target 변수와 상관도가 낮은 컬럼 제거
- 상관도가 높은 feature의 이상치 제거
- 피처 데이터의 경우 왜곡 정도가 심하면 로그 변환 수행
- target 데이터의 경우 로그 변환 수행

| |ver 14|
|-------|-------|
|**제외 컬럼**|pressure, snowing|
|**피처 스케일링**|StandardScaler|
|**최종 선택 모델**|ExtraTreesRegressor|
|**MAE**|2.1058|
