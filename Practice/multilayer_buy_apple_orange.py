class MulLayer:
    def __init__(self):
        self.x = None
        self.y = None

    def forward(self, x, y):
        self.x = x
        self.y = y
        output = x*y
        return output

    def backward(self, dout):
        print('dout=', dout)
        dx = dout * self.y #x와 y를 바꾼다
        dy = dout * self.x
        return dx, dy

apple = 100
apple_num = 2

orange = 150
oragne_num = 3

tax = 1.1

#계층들
mul_fruit_layer = MulLayer()
mul_tax_layer = MulLayer()

#순전파
apple_price = mul_fruit_layer.forward(apple, apple_num)
orange_price = mul_fruit_layer.forward(orange, oragne_num)
apple_orange_price = apple_price + orange_price
fruit_price = mul_tax_layer.forward(apple_orange_price, tax)
print(apple_orange_price)
print(fruit_price)


#역전파
dprice = 1
dfruit_price, dtax = mul_tax_layer.backward(dprice)
