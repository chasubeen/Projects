### ***️⃣ 출처**
1. 상권정보 데이터: https://www.data.go.kr/data/15083033/fileData.do
2. 생활인구 데이터: https://data.seoul.go.kr/dataList/OA-14991/S/1/datasetView.do
3. 대학정보 데이터: https://data.seoul.go.kr/dataList/OA-12974/S/1/datasetView.do
4. 초중고학교정보 데이터 https://www.data.go.kr/data/15099519/fileData.do
5. 소득정보 데이터: https://www.bigdata-environment.kr/user/data_market/detail.do?id=8cee0160-2dff-11ea-9713-eb3e5186fb38
6. 상권변화지표 데이터: https://data.seoul.go.kr/dataList/OA-15575/S/1/datasetView.do#
7. 주민등록인구 데이터: https://data.seoul.go.kr/dataList/10727/S/2/datasetView.do
8. 지하철역사정보 데이터: https://data.kric.go.kr/rips/M_01_01/detail.do?id=32

### ***️⃣ 전처리**
- 각각의 데이터에서 필요한 변수들만을 추출
- 행정동 or 행정구 단위로 집계(합계 or 평균)
- 상권정보 데이터와 7개의 데이터를 병합하여 최종 데이터셋 생성

----------------------------------------------------------------------
### ***️⃣ 최종 데이터**
- 분석 목적에 맞게 활용하기 위해 **행정구** 단위와 **행정동** 단위로 각각 집계  

**1) 행정구 단위**  
  - 71개의 변수(행정구 + 34개의 업종 + 37개의 요소들) x 25개의 행(데이터/ 행정구 개수)  
  <img src = "https://user-images.githubusercontent.com/98953721/219932638-9b10c62c-3624-4d95-befe-89f755f010c4.png" width = 750 height = 200>  
  
**2) 행정동 단위**   
  - 49개의 변수(행정구, 행정동 + 상위 10개 업종 + 37개의 요소들) x 426개의 행(데이터/ 행정동 개수)
  <img src = "https://user-images.githubusercontent.com/98953721/219932704-383a1feb-38e5-423a-9c75-41b2e69cf201.png" width = 750 height = 200>
