# **1. CNN이란**
- 합성곱 인공 신경망(Convolutional Neural Network)
- 전통적인 Neural Network 앞에 Convolutional 계층(주로 for 이미지 처리)을 붙인 형태
- 합성곱(Convolution)
  - 밀집층과 비슷하게 입력과 가중치를 곱하고 절편을 더하는 선형 계산
  - 입력 전체를 사용하여 선형 계산을 수행하는 밀집층과 달리 입력의 일부만을 사용하여 선형 계산을 수행 
- Convolutional 계층을 통해 입력받은 이미지에 대한 특징(feature)을 추출하고, 추출한 특징을 기반으로 기존의 뉴럴 네트워크를 이용하여 분류

# **2. Convolutional Layer**
- 입력 데이터로부터 특징을 추출하는 역할
- 특징을 추출하는 필터(Filter) 사용
- 필터의 값을 비선형 값으로 바꿔주는 활성화 함수(activation function)을 사용
  - torch.nn.ReLU
    - 비선형 활성화(activation)는 모델의 입력과 출력 사이에 복잡한 관계(mapping)를 생성
    - 선형 변환 후에 적용 -> 비선형성(nonlinearity) 도입, 신경망이 다양한 현상을 학습할 수 있도록 함
    - 코드
    ```Python
    torch.nn.ReLU(inplace = False)
    ```
    - 입력: (*), 여기서 *는 임의의 차원 수
    - 출력: (*), 입력과 동일한 차원
    
  - torch.nn.Softmax
    - n차원 출력 텐서의 요소가 [0,1] 범위에 있고 합이 1이 되도록 n차원 입력 텐서에 softmax 함수를 적용
    - 세 개 이상으로 분류하는 다중 클래스 분류에서 사용되는 활성화 함수
    - 분류될 클래스가 n개라 할 때, n차원의 벡터를 입력받아 각 클래스에 속할 확률을 추정
    - 코드
    ```Python
    torch.nn.Softmax(dim = 1) # 결과가 1개로 출력된다.
    ```
  
  - torch.nn.Sigmoid
    - S자형 곡선 또는 시그모이드 곡선을 갖는 수학 함수
    - 반환값은 단조증가하는 것이 일반적이지만 단조감소할 수도 있음
    - 반환값(y축)은 흔히 0에서 1까지의 범위를 가짐, 또는 -1부터 1까지의 범위를 가지기도 함
    - 코드
    ```Python
    torch.nn.Sigmoid()
    ```

- torch.nn.Conv2d
  - 입력의 너비와 높이 방향의 합성곱 연산을 구현한 Layer
  - 여러 개의 입력 평면으로 구성된 입력 신호에 2D 컨볼루션을 적용
  - 코드
  ```Python
  torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, 
                  groups=1, bias=True, padding_mode='zeros', device=None, dtype=None)
  ```

### **2-1. 필터(Filter)**
- 특징이 데이터에 있는지 검출하는 함수
- 밀집층의 뉴런에 해당됨
- 필터는 구현 후 행렬로 정의
- 입력받는 이미지를 모두 행렬 형태로 변환
- 입력받은 데이터에서 해당 특징을 가지고 있으면 결과값이 크게, 특징을 가지고 있지 않다면 결과값으로 0에 가까운 값이 나온다.
- 필터의 가중치/ 절편을 종종 **커널**이라고 부름
  cf> 뉴런 = 필터 = 커널
- 커널의 깊이 == 입력의 깊이

### **2-2. 패딩(Padding)**
- CNN 네트워크는 여러 단계에 걸쳐서 계속 필터를 연속적으로 적용함
  => 필터 적용 후 결과값이 작아지게 되면 처음에 비해 특징이 유실될 수 있음
- 필터 적용 후 특징의 유실에 대응하기 위해 이용
- 충분히 특징이 추출되기 전에 결과값이 작아지면 특징이 유실되므로 이를 방지하기 위해 패딩 기법을 활용
  - 입력값 주위로 0 값을 삽입해 입력값의 크기를 인위적으로 키우는 방식 -> 결과값이 작아지는 것 방지
- 원본 데이터에 0 값 삽입 시 원래의 특징이 희석됨 -> 머신러닝 모델이 학습 데이터에만 정확하게 맞아 들어가는 'overfitting'도 방지할 수 있음
- 종류
  - padding = 'same': 자동으로 패딩을 삽입해 입력값과 출력값의 크기를 맞춰주는 것
  - padding = 'valid': 패딩을 적용하지 않고 필터를 적용 -> 출력값의 크기가 작아진다.

### **2-3. 특성 맵(Feature map/Activation map)**
- 필터를 적용하여 얻어낸 결과
- 합성곱 층이나 풀링 층의 출력 배열
- 하나의 필터가 하나의 특성 맵 생성
- 필터의 개수 == 특성 맵의 개수

### **2-4. 스트라이드(Stride)**
- 필터를 적용하는 간격 ex> 우측으로 한 칸씩, 아래로 한 칸씩 적용
- 일반적으로 스트라이드는 1 pixel을 사용

### **2-5. 풀링(Pooling)**
- Feature map의 사이즈를 줄이는 방법
- Max Pooling을 가장 많이 사용함
- average pooling, L2-norm pooling 등이 있음
- 데이터의 크기를 줄이고 싶을 때 선택적으로 사용함
- torch.nn.MaxPool2d
  - 여러 입력 평면으로 구성된 입력 신호에 2D max pooling을 적용
  - 코드
  ```Python
  torch.nn.MaxPool2d(kernel_size, stride=None, padding=0, dilation=1, 
                   return_indices=False, ceil_mode=False)
  ```
- torch.nn.AdaptiveAvgPool2d
  - 여러 입력 평면으로 구성된 입력 신호에 2D adaptive 평균 풀링을 적용(각 구역의 평균값 추출)
  - 코드
  ```Python
  torch.nn.AdaptiveAvgPool2d(output_size)
  ```
### **2-6. FC Layer(Fully Connected Layer)**
- 기존의 뉴럴 네트워크

### **2-7. Flatten Layer**
- 연속된 범위의 차원을 텐서로 평평하게 만듦(다차원 -> 1차원)
- torch.nn.Flatten
  - 코드
  ```Python
  torch.nn.Flatten(start_dim=1, end_dim=- 1)
  ```
  - Parameters>
    - start_dim(int): 평평하게 할 첫 번째 차원(default = 1)
    - end_dim(int): 평평하게 할 마지막 차원(default = -1)
    
### **2-8. Linear Layer**
- 저장된 가중치(weight)와 편향(bias)을 사용하여 입력에 선형 변환(linear transformation)을 적용하는 모듈
- torch.nn.Linear
  - 들어오는 데이터에 선형 변환을 적용: y = xA^T + b
  - 코드
  ```Python
  torch.nn.Linear(in_features, out_features, bias=True, device=None, dtype=None)
  ```

### **2-9. Dropout Layer**
- overfitting을 막기 위한 layer
- NN이 학습중일 때 랜덤하게 값을 발생하여 학습을 방해  
  => 학습용 데이터에 결과가 치우치는 것 방지
- torch.nn.Dropout
  - 훈련 중, 베르누이 분포의 샘플을 사용하여 확률 p로 입력 텐서의 일부 요소를 무작위로 0으로 만듦
  - 코드
  ```Python
  torch.nn.Dropout(p=0.5, inplace=False)
  ```
  - Parameters>
    - p (float): 원소가 0이 될 확률, default: 0.5

# **3. 손실 함수(Loss Function)**
- 지도학습(Supervised Learning) 시 알고리즘이 예측한 값과 실제 정답의 차이를 비교하기 위한 함수
  - 어떤 문제에서 머신러닝 알고리즘이 얼마나 잘못되었는지를 측정하는 기준
  - 작을수록 good
- 비용 함수(Cost Function) vs 손실 함수(Loss Function)  
  - 손실 함수: 샘플 하나에 대한 손실을 정의  
  - 비용 함수: 훈련 세트에 있는 모든 샘플에 대한 손실 함수의 합  
  ※ 사실 둘을 딱히 구분해서 사용하지는 x
- nn.CrossEntropyLoss
  -  입력 logit과 target 사이의 교차 엔트로피 손실을 계산
  -  k개 클래스의 분류 문제를 훈련시킬 때 유용
  -  추가적인 증강 가중치는 1차원 tensor여야 함
  -  불균형한 훈련 세트에 특히 유용함
  - 코드
  ```Python
  torch.nn.CrossEntropyLoss(weight=None, size_average=None, ignore_index=- 100, reduce=None, 
                            reduction='mean', label_smoothing=0.0)
  ```
# **4. 에폭(Epoch)**
- 전체 훈련 데이터가 학습에 한 번 사용되는 주기 
- 각 데이터를 모델에서 몇 번씩 복습할 것인지에 대한 횟수
- 대체적으로 복습 횟수가 너무 적으면 데이터를 제대로 학습할 수 없고(underfit), 복습 횟수가 일정 횟수 이상이면 추가적인 성능 향상 효과가 거의 사라지게 됨

# **5. 배치 사이즈(Batch Size)**
- 여러 이미지에 대한 gradient를 모아서 평균을 낸 뒤, 역전파를 1회만 시켜줄 수 있는데, 이때 1회 역전파에서 gradient를 모을 데이터의 개수
- 예를 들어, 전체 데이터의 개수가 1000개인 경우에 batch size = 100으로 설정한다면, 각 epoch에서는 1 ~ 100, 101 ~ 200, ... , 901 ~ 1000번 사진에 대한 gradient를 각각 모아
총 10회의 역전파로 인한 모델 파라미터 업데이트

# **6. 옵티마이져(Optimizer)**
- 비용 함수의 값을 최소로 하는 W(기울기)와 b(절편)을 찾는 방법(알고리즘)

### **6-1.경사 하강법(Gradient Descent)**
- 가장 기본적인 옵티마이져 알고리즘
- cost가 최소화되는 지점: 접선의 기울기가 0이 되는 지점(= 미분값이 0이 되는 지점)
- 비용 함수를 미분하여 현재 w에서의 접선의 기울기를 구하고, 접선의 기울기가 낮은 방향으로 w의 값을 업데이트하는 작업 반복

### **6-2. SGD(Stochastic Gradient Descent)**
- 배치 크기가 1인 경사 하강법 알고리즘
- 확률적 경사 하강법
- 데이터 셋에서 무작위로 균일하게 선택한 하나의 예에 의존하여 각 단계의 예측 경사 계산
- 코드
```Python
optimizer = torch.optim.SGD(model.parameters(), lr = 0.01, momentum = 0.9)
```

### **6-3. 학습율(learning rate)**
- 기울기 값 변경 시 얼마나 크게 변경할 지 결정

# **7. 모델 구현**
- Sequential 모델로 layer를 쌓는 방식을 주로 활용
- torch.nn.Sequential
  - 순서를 갖는 모듈의 컨테이너
  - 데이터는 정의된 것과 같은 순서로 모든 모듈들을 통해 전달됨
  - 코드
  ```Python
  model = nn.Sequential(
          nn.Conv2d(1,20,5),
          nn.ReLU(),
          nn.Conv2d(20,64,5),
          nn.ReLU()
        )
  ```
 
### **7-1. Early Stopping**
- Epoch를 일단 많이 돌게 한 후 특정 시점에서 멈추도록 하는 기능
- Parameters>
  - monitor: Early Stopping의 기준이 되는 값 ex> 'val_loss'로 설정 시 더 이상 val_loss가 감소하지 않으면 중단
  - patience: 학습이 진행됨에도 더 이상 monitor되는 값의 개선이 없는 경우 몇 번의 Epoch를 잔행할 지 설정
  - mode: monitor되는 값이 최소인지, 최대인지 설정
  - restore_best_weights
    - True로 설정 시 학습 완료 후 모델의 weight를 monitor하고 있던 값이 가장 좋았을 경우의 weight로 복원
    - False로 설정 시 마지막 학습이 끝난 후의 weight로 설정
- 코드
```Python
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor = 'val_loss',patience = 3,mode = 'min',restore_best_weights = False
)
```

### **7-2. ReduceLROnPlateau**
- 모델의 개선이 없을 경우 learning rate를 조절해 모델의 개선 유도
- Parameters>
  - monitor: ReduceLROnPlateau의 기준이 되는 값
  - factor: learning rate를 얼마나 변경시킬 것인지 정하는 값, learning rate * factor
  - patience: training이 진행됨에도 더 이상 monitor되는 값의 개선이 없을 경우 최적의 monitor 값을 기준으로 몇 번의 epoch를 진행하고 learning rate를 조절할 지의 값 설정
- 코드
```Python
reduce_on_plateau = tf.keras.callbacks.ReduceLROnPlateau(
    monitor = 'val_loss',factor = 0.1,patience = 10,mode = 'min',min_lr = 0.0001
)
```

### **7-3. ModelCheckPoint**
- 모델의 경로 설정
- 모델 경로를 '{epoch:02d} - {val_loss:.2f}.hdf5'라고 하면 앞의 명시한 문자열로 파일이 저장됨
  ex> 01-0.12f.h5
- Parameters>
  - save_weights_only
    - True: weight만 저장
    - False: 모델, layer, weight를 모두 저장
  - save_best_only
    - True: 모델의 정확도가 최고값을 갱신했을 때만 저장
    - False: 매번 저장한다.
  - verbose: Epoch이 돌 때마다 warning 등을 출력해주는 역할
- 코드
```Python
filepath = '{epoch:2d}-{val_loss:.2f}.hdf5' # 파일 경로 지정
model_checkpoint = tf.keras.callbacks.ModelCheckPoint(
    filepath,monitor = 'val_loss',save_best_only = True, save_weights_only = False, mode = 'min'
)
```
