# p.201 모멘텀, SGD, AdaGrad, Adam 으로 로스값 계산 구현

import os
import sys
sys.path.append(os.pardir)
import matplotlib.pyplot as plt
from dataset.mnist import load_mnist
from common.util import smooth_curve
from common.multi_layer_net import MultiLayerNet
from common.optimizer import *



(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True)

train_size = x_train.shape[0] # 학습데이터 사이즈 저장
batch_size = 128
max_iterations = 2000

# 옵티마이저라는 빈 딕셔너리에 객체를 담는데, 그 객체들은 클래스의 객체들이다.
optimizers = {}
optimizers['SGD'] = SGD()
optimizers['Momentum'] = Momentum()
optimizers['AdaGrad'] = AdaGrad()
optimizers['Adam'] = Adam()



# 빈 딕셔너리들에 반복문으로 추가
networks = {}
train_loss = {}

for key in optimizers.keys(): # 옵티마이저 딕셔너리에서 반복문으로 키값들을 가져와서 
    networks[key] = MultiLayerNet(input_size=784, hidden_size_list=[100, 100, 100, 100], output_size=10) # 1. 네트웍스 딕셔너리의 키로 쓰고, 벨류는 그 옆의 식으로
    train_loss[key] = [] # 2. 트레인로스 딕셔너리의 키로 쓰고, 벨류는 빈 리스트이다. 예를 들면
                        #{ 'SGD':[] , 'Momentum':[] ....} 꼴이다.



for i in range(max_iterations): # 반복 횟수 정하고, 그 안에서 반복
    batch_mask = np.random.choice(train_size, batch_size) # 전체 학습 사이즈에서 내가 원하는 배치사이즈만큼 랜덤으로 추출함. 
    x_batch = x_train[batch_mask] # 1. 그 숫자들로 인덱싱해서 모은 데이터 리스트
    t_batch = t_train[batch_mask] # 2. 그 숫자들로 인덱싱해서 모든 정답 라벨지 
    
    for key in optimizers.keys(): # SGD, Momentum, AdaGrad, Adam으로 반복문
        grads = networks[key].gradient(x_batch, t_batch) # 각각으로 그래디언트를 계산
        optimizers[key].update(networks[key].params, grads) # 그 계산한 그래디언트랑 내가 가진 파라미터로 업데이트도 해줌
    
        loss = networks[key].loss(x_batch, t_batch) # 마찬가지
        train_loss[key].append(loss) # 마찬가지
    
    if i % 100 == 0: # 100회 단위로 조건문 실행
        print( "===========" + "iteration:" + str(i) + "===========") # 몇 회인지
        for key in optimizers.keys(): # 키의 종류랑
            loss = networks[key].loss(x_batch, t_batch) # 로스값을
            print(key + ":" + str(loss)) # 출력


# 그래프
markers = {"SGD": "o", "Momentum": "x", "AdaGrad": "s", "Adam": "D"}
x = np.arange(max_iterations)
for key in optimizers.keys():
    plt.plot(x, smooth_curve(train_loss[key]), marker=markers[key], markevery=100, label=key)
plt.xlabel("iterations")
plt.ylabel("loss")
plt.ylim(0, 1)
plt.legend()
plt.show()