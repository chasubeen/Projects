# **ResNet50 모델링 결과**

## **1️⃣Baseline**  
### **- ResNet?**  
- 일반적으로는 신경망의 깊이가 깊어질수록 딥러닝 성능이 좋아짐
  - [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385) 논문에 의하면 신경망은 깊이가 깊어질수록 성능이 좋아지다가 일정한 단계에 다다르면 오히려 성능이 나빠진다고 함
- 깊어진 신경망을 효과적으로 학습하기 위한 방법으로 **레지듀얼(residual, 잔차)** 개념 도입
- Residual block을 이용해 기울기가 잘 전파될 수 있도록 일종의 **숏컷**을 만들어 줌 -> 기울기 소멸 문제 방지
- VGG19 구조를 뼈대로 하며, 거기에 합성곱 층들을 추가해서 깊게 만든 후 숏컷들을 추가한 모델  
  
<img src = "https://user-images.githubusercontent.com/98953721/220263791-69b1bdfc-56e3-4293-81d3-61ad2a21ac9c.png" width = 250 height = 250> <img src = "https://user-images.githubusercontent.com/98953721/220263979-2c5d7298-96ba-437c-acaa-b6b108328bb6.png" width = 250 height = 250>  
<img src = "https://user-images.githubusercontent.com/98953721/220264178-12a39f1b-93a5-495d-8e94-f29faa654545.png" width = 350 height = 350> 

### **- 구조**

<img src = "https://user-images.githubusercontent.com/98953721/220259257-0074e59d-1b4b-4d9c-8af8-2beb3a10b782.png" width = 700 height = 200>

<img src = "https://user-images.githubusercontent.com/98953721/220260021-579b1f5f-a28e-4379-a9ce-20c1a4f85dce.png" width = 700 height = 100>
<img src = "https://user-images.githubusercontent.com/98953721/220261298-fa159988-f9e1-4d61-9567-13ba1e6f3fe9.png" width = 700 height = 700>
<img src = "https://user-images.githubusercontent.com/98953721/220261377-3c9d2861-df47-4b3e-9a68-ec913cf1e417.png" width = 700 height = 700>
<img src = "https://user-images.githubusercontent.com/98953721/220261615-2cecac1e-9fd9-4559-b999-5fee6c74c9e5.png" width = 700 height = 950>
<img src = "https://user-images.githubusercontent.com/98953721/220261989-c674f291-579b-4d07-b7b7-8d52dd0d0356.png" width = 700 height = 950>
<img src = "https://user-images.githubusercontent.com/98953721/220262064-537e55fe-ff1b-4bb2-83e1-fc1c5aa4abda.png" width = 700 height = 70>

### **- 데이터 구조에 맞게 head 부분 수정**
  - 7개 클래스로 분류하는 모델
<img src = "https://user-images.githubusercontent.com/98953721/220263554-daef77e6-ecf2-46a4-9b55-7cddb47da69c.png" width = 700 height = 70>

## **2️⃣ 성능 최적화 결과**
### **✅ Case 1**
- Optimizer: SGD(Stochastic Gradient Descent/ 확률적 경사 하강법)
- batch size: 64
- Epoch: 100
- 손실함수: CrossEntropyLoss

- 목표: 적절한 **learning rate** 찾기
  - lr_scheduler, early stopping 적용

|   |**sgd_ver1**|**sgd_ver2**|**sgd_ver3**|
|------|-------|-------|-------|
|**초기 learning rate**|1e-2|1e-3|1e-4|
|**min_lr**|1e-10|1e-12|1e-12|
|**Best Acc**|**0.6639**|0.6420|0.5601|

### **✅ Case 2**
- 목표: 적절한 **batch size** 찾기
- batch size가 **클** 때
  - 한 번 학습할 때 많은 데이터로 학습
  - 빠른 학습/수렴 속도 -> local optima에 빠질 확률이 작음
  - 작은 배치에 비해 과적합 위험성 증가(batch가 크면 계산되는 loss값의 편차가 작으므로)
- batch size가 **작을** 때
  - 1 epoch 당 iteration이 크기 때문에 step이 많아짐
  - 작은 데이터로 학습 -> loss의 분산이 커서 정규화 효과가 있음, 조금 더 다양하고 예리하게 학습할 수 있음
  - 긴 학습시간, 많은 step 수로 인해 local minima에 빠질 위험성 증가

- 일반적으로 learning rate와 batch size는 **양의 상관관계**를 보임 -> 동시 조정이 요구됨
  - 각각의 경우에 대해 batch size와 learning rate만을 조정/ 나머지 조건은 **Case 1**과 동일

|   |**sgd_ver4**|**sgd_ver5**|**sgd_ver6**|**sgd_ver7**|
|------|-------|-------|-------|-------|
|**batch**|128|128|256|256|
|**초기 lr**|1e-2|1e-3|1e-1|1e-2|
|**min_lr**|1e-10|1e-12|1e-10|1e-12|
|**Best Acc**|**0.6642**|0.6104|0.6183|0.6628|

- - -

### **✅ Case 3**
- Optimizer 변경(SGD -> **Adam**)
- batch size: 64
- Epoch: 100
- 손실함수: CrossEntropyLoss

- 목표: 적절한 **learning rate** 찾기

|   |**adam_ver1**|**adam_ver2**|**adam_ver3**|
|------|-------|-------|-------|
|**초기 learning rate**|1e-3|1e-4|1e-5|
|**min_lr**|1e-13|1e-14|1e-15|
|**Best Acc**|0.6442|**0.6626**|0.5712|

### **✅ Case 4**
- 목표: 적절한 **batch size** 찾기
  - 각각의 경우에 대해 batch size와 learning rate만을 조정
  - 나머지 조건은 **Case 3**과 동일 

|   |**adam_ver4**|**adam_ver5**|**adam_ver6**|**adam_ver7**|
|------|-------|-------|-------|-------|
|**batch**|128|128|256|256|
|**초기 lr**|1e-3|1e-4|1e-2|1e-3|
|**min_lr**|1e-13|1e-14|1e-12|1e-13|
|**Best Acc**|0.6521|0.6531|0.5909|**0.6654**|

### **✅ Case 5**
- 목표: **L2 정규화(L2 규제)** 적용
  - 규제(Regularization): 학습이 과대적합되는 것을 방지하고자 일종의 penalty를 부여하는 것
  - 각 가중치 제곱의 합에 규제 강도를 곱한 값($Error = MSE + α𝑤^2$)
  - 원형의 경계를 만들어서 학습 데이터셋의 최적 지점인 w* 에 도달하지 못하게 하고 경계 내부의 v* 까지만 도달할 수 있도록 하는 방식
  - Optimizer로 Adam을 사용할 경우 **weight_decay** 파라미터를 추가할 수 있음
    - 값이 클수록 규제 강도가 강한 것을 의미 -> 가중치가 더 많이 감소됨
  ```Python
  optimizer = optim.Adam(model.parameters(), lr = 1e-3, weight_decay = 1e-3)
  ```
- batch size: 128, 256
- Epoch: 100
- 손실함수: CrossEntropyLoss
- learning rate
  - 각각의 batch size에 대해 성능이 가장 **좋았던** 모델들의 learning rate 활용
  - batch = 128: 초기(1e-4), min_lr(1e-14)
  - batch = 256: 초기(1e-3), min_lr(1e-13)

|   |**adam_ver8**|**adam_ver9**|**adam_ver10**|**adam_ver11**|
|------|-------|-------|-------|-------|
|**batch**|128|128|256|256|
|**초기 lr**|1e-4|1e-4|1e-3|1e-3|
|**min_lr**|1e-14|1e-14|1e-13|1e-13|
|**L2 규제 강도**|1e-4|1e-3|1e-4|1e-3|
|**Best Acc**|0.6571|**0.6658**|0.6653|0.6525|

## **✅ Case 6**
- 목표: **교란 라벨링** 적용
  - 분류(classification) 문제에서 일정 비율의 라벨을 의도적으로 잘못된 라벨로 만들어서 학습을 방해하는 방식
    - 단순한 방식이지만 과적합을 효과적으로 막을 수 있음
  - 손실층(loss layer)에 규제를 두는 방식
    - 학습 과정에서 교란 라벨을 추가한 후 학습 진행
- 이전의 모델들 중 성능이 가장 좋았던 모델 구조로 모델링
  - Optimizer: Adam
  - batch size: 128
  - Epoch: 100
  - learning rate
    - 초기: 1e-4
    - min_lr: 1e-14
  - 손실함수: CrossEntropyLoss
  - weight_decay: 1e-3

|   |**adam_ver12**|**adam_ver13**|**adam_ver14**|
|------|-------|-------|-------|
|**alpha(교란 %)**|10|15|20|
|**Best Acc**|**0.6177**|0.5902|0.5610|

- 오히려 성능이 저하되었음
---

## **#️⃣ References**
- [기술블로그_과적합 방지를 통한 모델 성능 개선](https://yeong-jin-data-blog.tistory.com/entry/%ED%8C%8C%EC%9D%B4%ED%86%A0%EC%B9%98-%EC%8A%A4%ED%84%B0%EB%94%94-%EA%B3%BC%EC%A0%81%ED%95%A9-%EB%B0%A9%EC%A7%80%EB%A5%BC-%ED%86%B5%ED%95%9C-%EB%AA%A8%EB%8D%B8-%EC%84%B1%EB%8A%A5-%EA%B0%9C%EC%84%A0-%EB%B0%A9%EB%B2%95)  
- [교란 라벨링 관련 논문](https://arxiv.org/abs/1605.00055)  
- [교란 라벨링 구현](https://github.com/amirhfarzaneh/disturblabel-pytorch/blob/master/main.py)
