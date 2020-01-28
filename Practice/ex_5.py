import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.detach(), encoding = 'utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.detach(), encoding = 'utf-8')

import numpy as np

class NN():
    def __init__(self, input_size, output_size, hidden_size):
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size

        self.v = np.random.randn(input_size, hidden_size) #입력층에서 히든층까지의 weight
        self.w = np.random.randn(hidden_size, output_size) #히든에서 출력층까지의 weight

    #활성함수
    def activation(self, x):
        # sigmoid
        return 1 / (1 * np.exp(-x))

    #활성함수의 미분
    def d_activation(self, x):
        # sigmoid deriv
        return x * (1-x)

    def forward(self, x):
        self.h_in = np.dot(x, self.v)           # 입력 x와 가중치 v를 np.dot을 이용하여 행렬곱을 구한다
        self.h_out = self.activation(self.h_in) # 1에서 구한 값이 활성함수(시그모이드)를 거친다
        self.o_in = np.dot(self.h_out, self.w)  # 2에서 구한 값 h_out 과 가중치 w를 np.dot을 이용하여 행렬곱을 구한다
        self.o_out = self.activation(self.o_in) # 3에서 구한 값이 활성함수(시그모이드)를 거친다
        return self.o_out                       # 결과값

    def backward(self, x, y, o):
        o_error = y - o
        o_delta = o_error * self.d_activation(o)

        h_error = np.dot(o_delta, self.w.T)
        h_delta = h_error * self.d_activation(self.h_out)

        self.v += np.dot(x.T, h_delta)
        self.w += np.dot(self.h_out.T, o_delta)

    # x, y, o 데이터로 학습
    def train(self, x, y, o=None):          #입력값 x와 실제 출력값 y, 신경망에 의해 예측된 o를 통해서 backpropagation을 학습. o가 인자로 넘어오지 않을 경우에 o값을 직접 구함
        if o is None: o = self.predict(x)
        self.backward(x,y,o)

    def predict(self, x):                   #신경망을 통해서 x값으로 출력값을 예측한다
        return self.forward(x)


#이진화(threshold 값 이상: 1, 미만: 0 로 변환)
def binarize(x, threshold = 0.5): # threshold 값 기준으로 0과 1로 이진화. 여기서 data는 0과 1만 사용하므로 threshold의 기본값을 0.5로
    return (x >= threshold).astype(float)

def accuracy(y, y_pred):                #y와 y_pred가 얼마나 일치하는지 백분율로 표시
    return 100 * (y == y_pred).sum() / (y.shape[0] * y.shape[1])

def loss(y, y_pred):                    #y와 y_pred사이의 손실함수를 계산
    return np.mean(np.square(y - y_pred))

#각 레이어의 개수
input_size, output_size, hidden_size = 10, 4, 15

#epochs(학습 반복 양)
epochs = 1000

#NN 초기화
NN = NN(input_size, output_size, hidden_size)

#학습데이터 설정
train_file = 'train_data.txt'
train_data = np.loadtxt(train_file, delimiter = ',')
x_train, y_train = train_data[:, :-output_size], train_data[:, -output_size:] #학습데이터와 검증데이터를 불러옴. x는 데이터하나당 0~-output_size미만, y는 -output_size~끝까지

#검증데이터 설정
test_file = 'test_data.txt'
test_data = np.loadtxt(test_file, delimiter = ',')
x_test, y_test = test_data[:, :-output_size], train_data[:, -output_size:]

#학습
for epoch in range(1000):
    y_pred = NN.predict(x_train)
    y_pred_bin = binarize(y_pred)
    print("Loss: {:.10f}({}/{})".format(loss(y_train, y_pred), epoch, epochs))
    NN.train(x_train, y_train, y_pred)

    print(y_pred)
