# **ResNet 모델링 결과**

## **1️⃣Baseline**  
### **- ResNet50?**  
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
```Python
optimizer = torch.optim.SGD(model.parameters(), lr = 1e-2, momentum = 0.9)
```
- batch size: 64
- Epoch: 100
- 손실함수: CrossEntropyLoss
- **learning rate** 조정
  - lr_scheduler, early stopping 적용
```Python
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode = 'min', patience = 3, factor = 0.1, 
                                                       min_lr = min_lr,verbose = True)  # lr scheduling
early_stopping = EarlyStopping(patience = 20, verbose = False) # 조기 종료(사용자 정의 모듈)
```
|   |**sgd_ver1**|**sgd_ver2**|**sgd_ver3**|
|------|-------|-------|-------|
|**초기 learning rate**|1e-2|1e-3|1e-4|
|**min_lr**|1e-10|1e-12|1e-13|
|**Best Acc**|**0.6587**|0.6285|0.5710|


✅ **Case 2**
- Case 1에서 **batch size**만 변경(64 -> 128)
- batch size가 **클** 때
  - 한 번 학습할 때 많은 데이터로 학습
  - 빠른 학습/수렴 속도 -> local optima에 빠질 확률이 작음
  - 작은 배치에 비해 과적합 위험성 증가(batch가 크면 계산되는 loss값의 편차가 작으므로)
- batch size가 **작을** 때
  - 1 epoch 당 iteration이 크기 때문에 step이 많아짐
  - 작은 데이터로 학습 -> loss의 분산이 커서 정규화 효과가 있음, 조금 더 다양하고 예리하게 학습할 수 있음
  - 긴 학습시간, 많은 step 수로 인해 local minima에 빠질 위험성 증가

|   |**sgd_ver4**|**sgd_ver5**|**sgd_ver6**|
|------|-------|-------|-------|
|**손실 함수**|CrossEntropyLoss|**가중** CrossEntropyLoss|**가중** CrossEntropyLoss|
|**활성화 함수**|softmax|softmax|**log** softmax|
|**Best Acc**|0.6427|**0.6622**|0.6606|

- - -

✅ **Case 3**
- Optimizer 변경(SGD -> **Adam**)
- 실패한 모델들 
  - 손실함수: CrossEntropyLoss
  - 활성화 함수: softmax
  - lr_scheduler, early stopping 적용
 
|   |**adam_ver1**|**adam_ver2**|**adam_ver4**|
|------|-------|-------|-------|
|**batch**|64|128|64|
|**초기 lr**|0.0005|0.001|0.0001|
|**min_lr**|1e-7|1e-8|1e-10|
|**Epoch**|100|200|200|
|**Best Acc**|0.24xx(중단)|(중단)|(중단)|

- 실패 원인 분석: batch size와 learning rate의 관계
<img src = "https://user-images.githubusercontent.com/98953721/209615216-0c5679ab-11db-438d-846d-06bc2f2d6a98.png" width = 300 height = 300>

-> 일반적으로 learning rate와 batch size는 **양의 상관관계**를 보인다.  
-> learning rate를 **줄인** 후 다시 모델링 진행  

✅ **Case 4**
- Case 3에서 **learning rate & batch size** 튜닝 -> 적절한 조합 탐색
- 손실함수: CrossEntropyLoss
- 활성화 함수: softmax
- lr_scheduler, early stopping 적용
- Epoch: 200

|   |**adam_ver3**|**adam_ver5**|**adam_ver6**|
|------|-------|-------|-------|
|**batch**|128|128|64|
|**초기 lr**|1e-4|1e-5|1e-5|
|**min_lr**|1e-10|1e-12|1e-12|
|**Best Acc**|0.6137|**0.6344**|0.6249|

✅ **Case 5**
- Case 4에서 **손실 함수 & 활성화 함수** 튜닝
- batch size: 128
- learning rate
  - 초기: 1e-5
  - lr_scheduler, early stopping 적용(min_lr: 1e-12)
- Epoch: 200 

|   |**adam_ver7**|**adam_ver8**|
|------|-------|-------|
|**손실 함수**|**가중** CrossEntropyLoss|**가중** CrossEntropyLoss|
|**활성화 함수**|softmax|**log** softmax|
|**Best Acc**|0.6238|**0.6444**|


### **#️⃣ Reference**
- [VGG19 관련 논문(Very Deep Convolutional)](https://arxiv.org/abs/1409.1556)  
- [Learning rate & batch size best 조합 찾기(기술 블로그)](https://inhovation97.tistory.com/32)  
- [learning rate& batch size 관련 논문](https://www.sciencedirect.com/science/article/pii/S2405959519303455#fig2)  

