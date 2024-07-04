# 밑바닥 딥러닝1권 p143-145 직접 주석달며 공부하기

import sys
import os
sys.path.append(os.pardir)
# 이렇게 했지만, 모듈 가져올 때 에러 떠서 결국 경로 박아줌

import numpy as np
import matplotlib.pyplot as plt

sys.path.append('C:\\Users\\Hannibal\\Desktop\\WegraLee-deep-learning-from-scratch-master')
from dataset.mnist import load_mnist
from two_layer_net import TwoLayerNet


(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)
network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)



iters_num = 10000
# 총 반복횟수를 임의로 설정

train_size = x_train.shape[0]
batch_size = 100
# 미니 배치 크기.
# 따라서 만약 트레인사이즈가 60000이라면, 배치 사이즈가 100이기 때문에 1에포크를 위해 600번 반복해야 한다.
# 그러면 총 반복횟수인 10000회를 채우기 위해서 16.67에포크를 돌려야 한다.

learning_rate = 0.1
train_loss_list = []
train_acc_list = []
test_acc_list = []


iter_per_epoch = max(train_size / batch_size, 1)
# 1에포크를 위해 필요한 반복 횟수
# 만약 트레인사이즈가 배치사이즈보다 작은 경우에도, 최소한 1에포크는 보장해줘야 해서 max함수 사용


for i in range(iters_num):
    batch_mask = np.random.choice(train_size, batch_size)
    # 미니배치 획득
    # np.random.choice(a,b): 0부터a미만 범위에서 b개의 서로 다른 인덱스 번호를 반환
    
    x_batch = x_train[batch_mask]
    # 인덱스 번호들로 엑스 트레인을 인덱싱
    
    t_batch = t_train[batch_mask]
    # 마찬가지
    
    grad = network.numerical_gradient(x_batch, t_batch)
    # 포인트는 이전까지 x와 t를 썼던 자리에 그저 x_batch와 t_batch를 썼다는 것뿐이다.
    
    
    
    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learning_rate * grad[key]
    # 매개변수 갱신    
        
    
    
    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)
    # 학습 경과 기록
    
    
    if i % iter_per_epoch == 0:
    # 예를 들면 10000까지의 반복 중에서 600번째, 1200번째, 1800번째... 
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        # 에포크마다 어펜드
        
        print("train acc, test acc | " + str(train_acc) + ", " + str(test_acc))



# 그래프 그리기
markers = {'train': 'o', 'test': 's'}
x = np.arange(len(train_acc_list))
plt.plot(x, train_acc_list, label='train acc')
plt.plot(x, test_acc_list, label='test acc', linestyle='--')
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.ylim(0, 1.0)
plt.legend(loc='lower right')
plt.show()
