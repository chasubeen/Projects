# **0. 대회 소개**

# **회귀 모델 적용 시 고려 사항**
- target 데이터가 정규 분포 형태인지 -> 로그 변환을 통해 개선
- 범주형(categorical) feature의 경우 One-hot Encoding으로 피처를 인코딩
- feature 데이터의 데이터 분포
  - skew() 함수를 이용해 칼럼의 데이터 세트의 왜곡된 정도 추출
  - 반환값이 1 이상인 경우 왜곡 정도가 높다고 판단
  - skew() 함수 적용 시 One-hot Encoding된 변수는 피해야 한다.
- 이상치(Outlier) 처리 -> target 변수와 상관관계가 높은 feature에 대해서만 이상치 제거
  - 참고자료
    - 파이썬 머신러닝 완벽가이드(4-9. 분류 실습_캐글 신용카드 사기 검출)
    - [기술블로그](https://hungryap.tistory.com/69)
- feature 변수들 간의 다중공선성(multicollinearity)
  - 참고자료: 회귀분석 강의록, [기술블로그](https://ysyblog.tistory.com/171)
  

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

---
### **이상치 제거 작업 진행**

- ver6
  - 상관도가 높은 feature의 이상치만 제거
  - categorical 변수: LabelEncoding,One-hot Encoding 적용
  - feature: StandardScaler 적용
  - target: 로그 변환

### **회귀 트리 적용**
- ver7
  - 모든 feature의 이상치 제거
  - categorical 변수: LabelEncoding,One-hot Encoding 적용
  - feature: 로그 변환, StandardScaler 적용
  - target: 로그 변환 

- ver8
  - 상관도가 높은 feature의 이상치만 제거
  - categorical 변수: LabelEncoding,One-hot Encoding 적용
  - feature: 로그 변환, StandardScaler 적용
  - target: 로그 변환 
  
- ver9
  - 상관도가 높은 feature의 이상치만 제거
  - categorical 변수: LabelEncoding,One-hot Encoding 적용
  - feature: StandardScaler 적용
  - target: 로그 변환 
  
- ver10
  - 왜곡 정도가 높은 pressure 컬럼 제거(상관계수: 0.0)
  - 상관도가 높은 feature의 이상치만 제거
  - categorical 변수: LabelEncoding,One-hot Encoding 적용
  - feature: 로그 변환, StandardScaler 적용
  - target: 로그 변환 
---
**여기서부터 해봐야 함**  
- ver11
  - 왜곡 정도가 높은 pressure 컬럼, snowing 제거(상관계수: 0.0)
  - 상관도가 높은 feature의 이상치만 제거
  - categorical 변수: LabelEncoding,One-hot Encoding 적용
  - feature: 로그 변환, StandardScaler 적용
  - target: 로그 변환 
