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
    - 수정된 선형 단위 함수를 요소별로 적용(선형 -> 비선형)
    - 코드
    ```Python
    torch.nn.ReLU(inplace = False)
    ```
    - 입력: (*), 여기서 *는 임의의 차원 수
    - 출력: (*), 입력과 동일한 차원
   
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

### **2-7. Dropout Layer**
- overfitting을 막기 위한 layer
- NN이 학습중일 때 랜덤하게 값을 발생하여 학습을 방해  
  => 학습용 데이터에 결과가 치우치는 것 방지
 
 # **3. 모델 구현**
 - tensorflow의 경우 Sequential 모델로 layer를 쌓는 방식을 주로 활용
 ```Python
 # 3개의 층을 가진 Sequential Model 정의하기
model = keras.Sequential(
    [
        # Dense Layer: 일반 layer
        layers.Dense(2,activation = "relu",name = "layer1")
        layers.Dense(3,activation = "relu",name = "layer2")
        layers.Dense(4, name = "layer3")
    ]
)

# 테스트용 입력에서 모델 호출
X = tf.ones((3,3))
y = model(X)
 ```
### **3-1. Early Stopping**
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

### **3-2. ReduceLROnPlateau**
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

### **3-3. ModelCheckPoint**
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


 
 
 
 
 
 
 
 
 
 
