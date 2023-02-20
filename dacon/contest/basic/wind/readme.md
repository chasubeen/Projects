# **0. 대회 소개**

# **회귀 모델 적용 시 고려 사항**
- target 데이터가 정규 분포 형태인지 -> 로그 변환을 통해 개선
- 범주형(categorical) feature의 경우 One-hot Encoding으로 피처를 인코딩
- feature 데이터의 데이터 분포
  - skew() 함수를 이용해 칼럼의 데이터 세트의 왜곡된 정도 추출
  - 반환값이 1 잇항인 경우 왜곡 정도가 높다고 판단
  - skew() 함수 적용 시 One-hot Encoding된 변수는 피해야 한다.

# **Version**
- ver1
  - categorical 변수: LabelEncoding 적용
  - feature: 로그 변환, RobustScaler 적용
  - target: 로그 변환
- ver2
  - categorical 변수: LabelEncoding 적용
  - feature: RobustScaler 적용
  - target: 로그 변환
- ver3
  - categorical 변수: One-hot Encoding 적용
  - feature: 로그 변환, RobustScaler 적용
  - target: 로그 변환
- ver4
  - categorical 변수: One-hot Encoding 적용
  - feature: RobustScaler 적용
  - target: 로그 변환
- ver5
  - categorical 변수: LabelEncoding 적용
  - feature: 로그 변환, StandardScaler 적용
  - target: 로그 변환
- ver6
  - categorical 변수: LabelEncoding,One-hot Encoding 적용
  - feature: StandardScaler 적용
  - target: 로그 변환
