from layer_naive import *

apple = 100
apple_num = 2
orange = 150
orange_num = 3
tax = 1.1


# layer
mul_apple_layer = MulLayer() #곱셈클래스로 사과가격과 개수를 인수로 받고, 역전파를 위헤 저장한 뒤, 곱할 값을 반환
mul_orange_layer = MulLayer() #곱셈클래스로 오렌지가격과 개수를 인수로 받고, 역전파를 위헤 저장한 뒤, 곱할 값을 반환
add_apple_orange_layer = AddLayer() #덧셈클래스로 사과와 오렌지를 인수로 받고 더할 것
mul_tax_layer = MulLayer() #곱셈클래스로 그 값과 텍스를 인수로 받고, 역전파를 위헤 저장한 뒤, 곱할 값을 반환

# forward
apple_price = mul_apple_layer.forward(apple, apple_num)  # (1) 곱셈순전파. 이때 mul_apple_layer.apple=apple, mul_apple_layer.apple_num=apple_num으로 저장됨
orange_price = mul_orange_layer.forward(orange, orange_num)  # (2)곱셈순전파 이때 mul_orange_price.orange=orange, mul_orange_price.orange_num=orange_num으로 저장됨
all_price = add_apple_orange_layer.forward(apple_price, orange_price)  # (3)덧셈순전파
price = mul_tax_layer.forward(all_price, tax)  # (4) 이때 mul_tax_layer.all_price=all_price, mul_tax_layer.tax=tax로 저장됨
# backward
dprice = 1 #역전파의 시작값을 1로 설정.
dall_price, dtax = mul_tax_layer.backward(dprice)  # (4) mul_tax_layer.backward(dprice)=  dout*dtax,  dout*dprice 이렇게 순서 바뀐 채로 반환됨.
dapple_price, dorange_price = add_apple_orange_layer.backward(dall_price)  # (3)
dorange, dorange_num = mul_orange_layer.backward(dorange_price)  # (2)
dapple, dapple_num = mul_apple_layer.backward(dapple_price)  # (1)

print("price:", int(price))
print("dApple:", dapple)
print("dApple_num:", int(dapple_num))
print("dOrange:", dorange)
print("dOrange_num:", int(dorange_num))
print("dTax:", dtax)
