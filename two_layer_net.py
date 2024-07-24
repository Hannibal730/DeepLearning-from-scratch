# p.180 오차역전파법을 적용한 신경망 구현

import sys, os
sys.path.append(os.pardir)
import numpy as np
#from common.layers import *
from common.gradient import numerical_gradient
from collections import OrderedDict


# 도트 노드의 순전, 역전
class Affine:
    def __init__(self, W, b):
        self.W = W # 가중치
        self.b = b # 바이어스
        self.x = None # 입력데이터
        self.original_x_shape = None # 입력데이터의 원래 형태
        self.dW = None # 역전파에서 계산된 가중치
        self.db = None # 역전파에서 계산된 바이어스

    def forward(self, x):
        self.original_x_shape = x.shape # 입력데이터의 원래 형태를 저장한다. 역전파 시 원래 형태로 복원하는데에 활용됨
        x = x.reshape(x.shape[0], -1) # 입력데이터를 2차원 배열로 변환해서 행렬 연산이 가능하게 도와준다.
        # 입력데이터는 2차원 이상의 데이터일 수도 있다. 예를 들면 (배치사이즈, 채널, 높이, 두께)일 수 있다.
        # 하지만 어파인 레이어는 2차원배열(행렬)만을   입력받는다.
        # 따라서 (배치사이즈, 채널, 높이, 두께) 이런 꼴의 데이터는 (배치사이즈, 기타) 이렇게 변형시켜야 한다.
        # x.shape[0]는 배치사이즈이고, 리쉐입함수에서 두 번째 매개변수로 -1을 쓰면 나머지 차원을 하나의 차원으로 평탄화시킨다.
        # 예를 들어 x=(10, 3, 32, 32)일 때, x = x.reshape(x.shape[0], -1)를 거치면, (10, 3 * 32 * 32) = (10, 3072)이 된다.
        
        self.x = x # 윗줄에서 변환한 데이터를 저장
        out = np.dot(self.x, self.W) + self.b # 도트실행&바이어스 더하기
        return out

    def backward(self, dout):
        dx = np.dot(dout, self.W.T) # 도트 노드에서의 역전파. p.172에서 증명했던 내용. 깃허브에 업로드해뒀음
        self.dW = np.dot(self.x.T,dout) # 도트 노드에서의 역전파. p.172에서 증명했던 내용. 깃허브에 업로드해뒀음
        self.db = np.sum(dout, axis=0) # p.175의 설명. (도트 노드의 순전파 출력에 바이어스를 더한다는 것은 역전파 때 dout의 각 행을 모두 더한 값을 바이어스노드에 되돌려준다는 것) 
        dx = dx.reshape(*self.original_x_shape)  # 변형시킨 입력 데이터의 형태를 원래 형태로 복원한다.
        return dx

# 소프트맥스
def softmax(a):
    exp_a=np.exp(a)
    sum_exp_a=np.sum(exp_a)
    y=exp_a/sum_exp_a
    return y

# CEE
def cross_entropy_error(y,t):
    delta=1e-7
    return -np.sum(t*np.log(y+delta))   

# 소프트맥스 & CEE 의 순전, 역전
class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None # 손실함수값
        self.y = None    # 각 요소에 대한 정확도 예상치
        self.t = None    # 정답 레이블(원-핫 인코딩 벡터 형태)
        
    def forward(self, x, t):
        self.t = t # 정답레이블
        self.y = softmax(x) # x를 입력하면 소프트맥스함수로 각 요소에 대한 정확도를 예상해줌
        self.loss = cross_entropy_error(self.y, self.t) # 그 두개를 CEE의 매개변수로 삼아서 손실함수값을 계산
        return self.loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0] # 배치 사이즈를 정답레이블의 요소 개수로 설정
        
        if self.t.size == self.y.size: # 정답 레이블이 원 핫 인코딩 벡터 형태인 경우
            dx = (self.y - self.t) / batch_size # p.299에 필기한 내용. 
        
        else: # 정답 레이블이 원 핫 인코딩 벡터 형태가 아닌 경우, 즉 self.t가 원핫인코딩이 아닌 클래스 인덱스 테이블로 주어졌을 때.
                
                # 예를 들어 배치 사이즈가 3이고, 3개의 클래스(A,B,C)가 있는 경우에 A의 인덱스는1, B의 인덱스는2, C의 인덱스는 3이다.
                
                # t가 원 핫 인코딩이라면              
                # B: [0,1,0]
                # C: [0,0,1]
                # A: [1,0,0] 이때 t.size=9이다.
                
                # t가 클래스 인덱스 테이블이라면
                # B:1
                # C:2
                # A:0
                # 이때 t.size=3이다.
                
                # 우리가 사용하는 정답 예측치인 y는
                # [0.3, 0.4, 0.3],
                # [0.2, 0.2, 0.6],
                # [0.1, 0.7, 0.2] 꼴로서 이때 y.size=9이다.

            # 따라서 self.t.size == self.y.size는 정답레이블이 원핫 인코딩 형태임을 의미하는 것이다.
            
            dx = self.y.copy()
            dx[np.arange(batch_size), self.t] -= 1 #np.arange(batch_size)로 0,1,2를 만듦.
            # 결국 dx[0,1], dx[1,2], dx[2,0]에서 각각 1씩 뺌
            # 1을 빼는 이유: y1-t1, y2-t2, y3-t3에서 정답클래스일 때는 t가 1이 되므로, 정답일 때만 y에서 t로써 1을 빼는 거라고 이해하자.
            dx = dx / batch_size
        return dx

# 렐루의 순전, 역전
class Relu:
    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = (x <= 0) #-> 트루행
        out = x.copy()
        out[self.mask] = 0 # ->트루는 0으로
        return out

    def backward(self, dout):
        dout[self.mask] = 0 
        dx = dout
        return dx

# 시그모이드
def sigmoid(self,x):
        return 1 / (1+np.exp(-x))

# 시그모이드의 순전, 역전
class Sigmoid:
    def __init__(self):
        self.out = None

    def forward(self, x):
        out = sigmoid(x)
        self.out = out
        return out

    def backward(self, dout):
        dx = dout * (1.0 - self.out) * self.out
        return dx

# 신경망 구현
class TwoLayerNet:

    def __init__(self, input_size, hidden_size, output_size, weight_init_std = 0.01):
    # 입력, 은닉, 출력 레이어의 뉴런 수, 가중치 초기화의 표준편차
    
    
        self.params = {}
        # params라는 빈 딕셔너리에 파라미터들 넣을 거임
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        # W1랑 b1 저장
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size) 
        self.params['b2'] = np.zeros(output_size)
        # W2랑 b2 저장


        self.layers = OrderedDict()
        # layers라는 빈 딕셔너리에 각 '계층'을 순서대로 저장할 거임. OrderedDict는 입력 순서를 유지하는 딕셔너리
        
        self.layers['Affine1'] = Affine(self.params['W1'], self.params['b1'])
        # 순전이나 역전을 하는 게 아니라, 그냥 어파인 클래스에 매개변수 w1,b1을 넣는 인스턴스를 만들고, 그 인스턴스를 어파인1이라고 이름지음.
        
        self.layers['Relu1'] = Relu()
        # 마찬가지
        
        self.layers['Affine2'] = Affine(self.params['W2'], self.params['b2'])
        # 마찬가지
        
        self.lastLayer = SoftmaxWithLoss()
        # 마찬가지
    
    
    #순전으로 예측값 계산
    def predict(self, x):
        for layer in self.layers.values(): # layers 딕셔너리의 벨류들 가져오고 반복
            x = layer.forward(x) # x데이터를 입력하면 반복되는 딕셔너리 벨류들의 순전을 계속 통과함
        return x # 마지막 레이어까지 통과한 뒤의 출력을 반환
    
    # 손실함수값계산
    def loss(self, x, t): # 입력데이터x, 정답레이블t
        y = self.predict(x) # 예측데이터y
        return self.lastLayer.forward(y, t) #라스트레이어인 SoftmaxWithLoss()에 매개변수로써 예측값과 정답지를 넣고, 손실함수값을 계산
    
    # 모델의 정확도를 측정
    def accuracy(self, x, t):
        y = self.predict(x) # 예측데이터y
        y = np.argmax(y, axis=1) # y에서 가장 큰 값의 인덱스를 선택
        if t.ndim != 1 : # 만약 t의 차원이 1차원이 아닌, 즉 원 핫 인코딩 형태일 경우에
            t = np.argmax(t, axis=1) # t는 t에서 가장 큰값의 인덱스를 선택
        accuracy = np.sum(y == t) / float(x.shape[0]) # 정확도는 예측데이터가 정답지와 일치하는 샘플의 개수 / 전체 샘플의 개수
        return accuracy
        
        
    # sol.1 수치적 기울기를 계산
    def numerical_gradient(self, x, t):
        loss_W = lambda W: self.loss(x, t)
        # numerical_gradient에 x, t를 입력하면 loss(x,t)를 쓰는데,
        # loss(x,t)는 y=predict(x)의 y와 t 간의 SoftmaxWithLoss로 계산됨.
        # 근데 predict가 layers를 순차적으로 쓰는 과정에서 변수 w1,b1,w2,b2를 씀.
        
        grads = {}
        grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
        # 어파인1 레이어에서의 w1에 대한 기울기
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
        # 어파인1 레이어에서의 b1에 대한 기울기
        
        grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
        # 어파인2 레이어에서의 w2에 대한 기울기
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])
        # 어파인2 레이어에서의 b2에 대한 기울기
        
        return grads
        
        
    # sol.2 역전파로 기울기를 계산 
    def gradient(self, x, t):
        
        # 우선 순전파로 왼쪽부터 오른쪽으로 가며 손실함수값 계산
        self.loss(x, t)

        # 본격적으로 역전파
        dout = 1
        dout = self.lastLayer.backward(dout)
        #SoftmaxWithLoss 레이어의 backward에 dout을 입력
        
        layers = list(self.layers.values()) # layers의 모든 벨류들을 리스트로 가져옴 
        layers.reverse() # 순서 뒤집기
        for layer in layers: # 뒤집어진 순서에서 반복문으로 역전파를 진행
            dout = layer.backward(dout) # dout=1에서부터 쭈욱 오른쪽에서 왼쪽으로 진행하며 dout 계산

        # 빈 딕셔너리에 결과 저장
        grads = {}
        
        grads['W1'] = self.layers['Affine1'].dW
        # __init__에서  를 했는데, 이 중 어파인 클래스의 역전파함수 (p.172에서 내가 증명했던 거)
        
        grads['b1'] = self.layers['Affine1'].db
        # 마찬가지
        
        grads['W2'] = self.layers['Affine2'].dW
        # 마찬가지
        
        grads['b2'] = self.layers['Affine2'].db
        # 마찬가지
        
        return grads