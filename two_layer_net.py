# 밑바닥 딥너링1권 p137-138

import sys, os
sys.path.append(os.pardir)  # 부모 디렉터리의 파일을 가져올 수 있도록 설정
from common.functions import *
from common.gradient import numerical_gradient


class TwoLayerNet:
# 2층 신경망 구현

    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
    # 초기값세팅
        
        # 인풋사이즈로 엠니스트 이미지사진을 넣는다면, 사진하나당 28*28=784
        # 아웃풋 데이터는 0부터 9까지 숫자에 대한 원핫인코딩 데이터일테니까 사이즈가 10
        # 사진을 100장 투입한다면, 인풋(100,784), w1(784,100(이때 100은 임의)), b1(100,), w2(100,10(아웃풋크기를 10으로 해야 하니깐)), b2(10,)
        # 히든 사이즈는 은닉층의 뉴런 개수이므로 w1의 784+ w2의 100= 884
        # 참고로 b1이 (100,)인 이유는 w1의 결과가 '100개의 데이터가 각 100개의 요소를 보유'하기 때문에 각 100개의 요소에 바이어스를 더해주기 위해서 (100,)이다.
        
        # weight_init_std는 가중치 초기화 편차이다.
        # 은닉층이 바뀔 때마다 가중치는 바뀐다. 이때 가중치의 편차를 키우면 다양한 패턴학습가능, 줄이면 특정패턴에 집중된 학습가능
        
        self.params = {}
        # 빈 딕셔너리
        
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        # 딕셔너리에 짝 추가.
        # 가중치 초기화 편차에 '요소가 각각 표준정규분포를 따르는 (인풋사이즈,은닉층사이즈)행렬을 곱함.
        # 참고로 randn은 표준정규분포를 따르고, rand는 0~1 사이에서 표준정규분포를 따름.
        
        self.params['b1'] = np.zeros(hidden_size)
        # 초기 바이어스는 0벡터로 설정.
        
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        # 마찬가지
        
        self.params['b2'] = np.zeros(output_size)
        # 마찬가지


    def predict(self, x):
    # x는 입력 데이터
        
        W1, W2 = self.params['W1'], self.params['W2']
        # W1를 def __init__에 있던 params['W1']이라는 로컬변수에 할당
        
        b1, b2 = self.params['b1'], self.params['b2']
        # 마찬가지
    
        a1 = np.dot(x, W1) + b1
        # a1은 입력데이터가 첫번째 은닉층을 지난 후의 데이터
        
        z1 = sigmoid(a1)
        # 활성화함수로 시그모이드 
        
        a2 = np.dot(z1, W2) + b2
        # 마찬가지
        
        y = softmax(a2)
        # 출력해서 보여줘야 하니 활성화함수로 소프트맥스
        
        return y
        
    
    def loss(self, x, t):
    # 손실함수 값
    # x : 입력 데이터, t : 정답 레이블    
        
        y = self.predict(x)
        # x를 입력 받았을 때의 y
        
        return cross_entropy_error(y, t)
        # 계산된 y와 처음에 입력한 t끼리의 교차 엔트로피 오차
        # 참고로 그 값은 -log(실제 정답에서의 예측확률)
    
    
    def accuracy(self, x, t):
    # 정확도 값
    # x : 입력 데이터, t : 정답 레이블  
    
        y = self.predict(x)
        # x를 입력 받았을 때의 y
        
        y = np.argmax(y, axis=1)
        # 그 y의 첫번째 축, 즉 y가 2차원 데이터라면 '행'을 기준으로 봤을 때 최대값의 인덱스를 반환
        
        t = np.argmax(t, axis=1)
        # 정답 레이블에서 마찬가지로 최대값의 인덱스를 반환
        
        accuracy = np.sum(y == t) / float(x.shape[0])
        # 두 인덱스가 같은 경우의 수 / 입력데이터 x의 쉐잎에서 0을 인덱싱, 즉 x가 2차원 데이터라면 '행'의 개수
        
        return accuracy
        
        
    
    def numerical_gradient(self, x, t):
    # x : 입력 데이터, t : 정답 레이블
        
        loss_W = lambda W: self.loss(x, t)
        # 위에서 정의했던 손실함수에 x와 t를 매개변수로 사용함.
        # 근데 그 손실함수는 예측함수를 사용함.
        # 근데 그 예측함수는 매개변수를 4가지나 사용함.
        # 그 매개변수 4가지는 params['W1'], params['W2'], params['b1'],params['b2']이다.
        # 이 매개변수 4가지가 Lambda W에서의 W를 맡고 있는 변수이다.
        # 따라서 참고로 loss_W 함수는 params['W1'], params['W2'], params['b1'],params['b2'] 얘네로 편미분 가능
        
        grads = {}
        # 빈 딕셔너리
        
        grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
        # 딕셔너리에 짝 추가.
        # 위에서 정의했던 params['W1']에 대한 loss_W의 기울기
        
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
        # 마찬가지
        
        grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
        # 마찬가지
        
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])
        # 마찬가지
        
        return grads

        
        
    def gradient(self, x, t):
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']
        grads = {}
        
        batch_num = x.shape[0]
        
        # forward
        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = softmax(a2)
        
        # backward
        dy = (y - t) / batch_num
        grads['W2'] = np.dot(z1.T, dy)
        grads['b2'] = np.sum(dy, axis=0)
        
        da1 = np.dot(dy, W2.T)
        dz1 = sigmoid_grad(a1) * da1
        grads['W1'] = np.dot(x.T, dz1)
        grads['b1'] = np.sum(dz1, axis=0)

        return grads
