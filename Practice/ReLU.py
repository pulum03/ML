class Relu:
    def __init__(self):
        self.mask = None #mask 는 True / False 로 구성된 numpy 배열

    def forward(self, x):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0

        return out

    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout

        return dx
