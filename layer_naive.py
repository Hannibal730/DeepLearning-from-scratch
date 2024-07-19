#곱셈노드
class MulLayer: 
    def __init__(self):
        self.x = None # 초기화
        self.y = None

    def forward(self, x, y): #순전파에서 x와 y로부터 발생되는 상황
        self.x = x
        self.y = y                
        out = x * y

        return out

    def backward(self, dout): #역전파에서 dout라는 결과값은 x와 y로부터 발생한 상황이라면, 실은 x는 y만큼 dout에, y는 x만큼 dout에 영향을 준 것이다.
        # 따라서 각 원인의 결과에 대한 미분값을 구하면, x의 그 값은 y이고 y의 그 값이다.바꾼다.
        dx = dout * self.y #그래서 dx는 y값이고,
        dy = dout * self.x #dy는 x값이다.

        return dx, dy


#덧셈노드
class AddLayer: 
    def __init__(self):
        pass #초기화 과정이 없다.

    def forward(self, x, y):
        out = x + y #덧셈

        return out

    def backward(self, dout):
        dx = dout * 1 #자기 자신이 그대로 dout에 반영되는, 즉 1의 곱만큼 반영되기 때문에 1만 곱한다.
        dy = dout * 1

        return dx, dy

#곱셈에서는 초기화하고, 덧셈에서는 초기화 안 하는 이유
#곱셈은 역전파를 위해서, 순전파에서 결과뿐만 아니라 인수들까지 저장해둬야 한다.
#따라서 곱셈 클래스의 순전파 함수가 인수들을 저장한다.
#만약 이렇게 인수들을 먼저 저장 안 한다면, 곱셈클래스에서 역전파를 실행할 때 인수에 대한 정보가 없어서 문제가 된다.
#만약 인수가 없다면, 없다는 것을 보여주기 위해 init에서 None을 쓴 것이다.
#그래서 p.162에서도 순전파를 먼저하면서 인수를 저장시킨 뒤에 역전파를 쓴다.

#반면 덧셈은 인수들을 저장할 필요가 없다.
#덧셈의 순전파 때는 곱셈 때처럼 어차피 인수를 입력 받아야 하고, 덧셈의 역전파 때는 그냥 결과값을 각각 인수로 보내주면 된다.