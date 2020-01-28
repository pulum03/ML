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
        #print('dout=', dout)
        dx = dout * self.y #x와 y를 바꾼다
        dy = dout * self.x
        return dx, dy

class AddLayer:
    def __init__(self):
        pass

    def forward(self, x, y):
        out = x + y
        return out

    def backward(self, dout):
        dx = dout * 1
        dy = dout * 1
        return dx, dy

apple = 100
apple_num = 2

orange = 150
orange_num = 3

tax = 1.1

#계층들
mul_apple_layer = MulLayer()
mul_orange_layer = MulLayer()
add_apple_orange_layer = AddLayer()
mul_tax_layer = MulLayer()

#순전파
apple_price = mul_apple_layer.forward(apple, apple_num)
orange_price = mul_orange_layer.forward(orange, orange_num)
all_price = add_apple_orange_layer.forward(apple_price, orange_price)
price = mul_tax_layer.forward(all_price, tax)

#print(price)


#역전파
dprice = 1
dall_price, dtax = mul_tax_layer.backward(dprice)
#print(dall_price, dtax)
dapple_price, dorange_price = add_apple_orange_layer.backward(dall_price)
#print(dapple_price, dorange_price)
dapple, dapple_num = mul_apple_layer.backward(dapple_price)
#print(dapple, dapple_num)
dorange, dorange_num = mul_orange_layer.backward(dorange_price)
#print(dorange, dorange_num)
