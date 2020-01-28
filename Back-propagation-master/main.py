import numpy as np

class NN():
    def __init__(self, input_size, output_size, hidden_size):
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size

        self.v = np.random.randn(input_size, hidden_size)
        self.w = np.random.randn(hidden_size, output_size)

    # 활성 함수
    def activation(self, x):
        # 시그모이드 함수를 활성 함수로 사용
        return 1 / (1 + np.exp(-x))

    # 활성 함수의 미분
    def d_activation(self, x):
        # 시그모이드 함수의 미분
        return x * (1 - x)

    # Forward propagation
    def forward(self, x):
        self.h_in = np.dot(x, self.v)
        self.h_out = self.activation(self.h_in)
        self.o_in = np.dot(self.h_out, self.w)
        self.o_out = self.activation(self.o_in)
        return self.o_out

    # Backward propagation
    def backward(self, x, y, o):
        o_error = y - o
        o_delta = o_error * self.d_activation(o)

        h_error = np.dot(o_delta, self.w.T)
        h_delta = h_error * self.d_activation(self.h_out)

        self.v += np.dot(x.T, h_delta)
        self.w += np.dot(self.h_out.T, o_delta)

    # x, y, o 데이터로 학습
    def train(self, x, y, o=None):
        if o is None: o = self.predict(x)
        self.backward(x, y, o)

    # x 데이터로 y 값 예측
    def predict(self, x):
        return self.forward(x)

# 이진화 (threshold 값 이상 : 1, 미만 : 0 로 변환)
def binarize(x, threshold=0.5):
    return (x >= threshold).astype(float)

# y와 y_pred의 정확도(일치도) 계산
def accuracy(y, y_pred):
    return 100 * (y == y_pred).sum() / (y.shape[0] * y.shape[1])

# 손실 함수 계산
def loss(y, y_pred):
    return np.mean(np.square(y - y_pred))


# input, output, hidden 레이어의 개수
input_size, output_size, hidden_size = 10, 4, 15

# epochs(학습 반복 양)
epochs = 1000

# NN 초기화
NN = NN(input_size, output_size, hidden_size)

# 학습 데이터 설정
train_file = 'train_data.txt'
train_data = np.loadtxt(train_file, delimiter=',')
x_train, y_train = train_data[:, :-output_size], train_data[:, -output_size:]

# 검증 데이터 설정
test_file = 'test_data.txt'
test_data = np.loadtxt(test_file, delimiter=',')
x_test, y_test = test_data[:, :-output_size], test_data[:, -output_size:]

# 학습
for epoch in range(epochs):
    y_pred = NN.predict(x_train)
    y_pred_bin = binarize(y_pred)
    print("Loss : {:.10f} ({}/{})".format(loss(y_train, y_pred), epoch, epochs))
    NN.train(x_train, y_train, y_pred)

# 검증
y_test_pred = NN.predict(x_test)
y_test_pred_bin = binarize(y_test_pred)
print("\n- 입력 값\n", x_test)
print("\n- 실제 결과\n", y_test)
print("\n- 예측 결과\n", y_test_pred)
print("\n- 이진화된 예측 결과\n", y_test_pred_bin)
print("\n- 정확도 : ", accuracy(y_test, y_test_pred_bin))
