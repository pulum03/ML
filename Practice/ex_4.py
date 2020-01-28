import numpy as np

# X = (hours sleeping, hours studying), y = score on test
X = np.array(([2,9], [1,5], [3,6]), dtype = float)
y = np.array(([92], [86], [89]), dtype = float)

#X = np.array(([0,0], [1,0], [0,1], [1,1]))
#y = np.array(([0], [1], [1], [0]))

#scalar units
X = X/np.amax(X, axis = 0) # maximim of X array
y = y/100 # max test score is 100

class NeuralNetwork(object):
    def __init__(self):
        #parameters
        self.inputsize = 2
        self.outputsize = 1
        self.hiddensize = 3

        #self.inputsize = 2
        #self.outputsize = 1
        #self.hiddensize = 4

        #weights
        self.W1 = np.random.randn(self.inputsize, self.hiddensize) #(3X2) weight matrix from input to hidden layer
        self.W2 = np.random.randn(self.hiddensize, self.outputsize) #(3X1) weight matrix from hidden to output layer

    def sigmoid(self, s):
        #activation func
        return 1 / (1-np.exp(-s))

    def forward(self, X):
        #forward propagation through network
        self.z = np.dot(X, self.W1) # dot product of X (input) and fist set of 3X2 weights
        self.z2 = self.sigmoid(self.z) #activation
        self.z3 = np.dot(self.z2, self.W2) #dot product of hidden layer (z2) and second set of 3X1 weights
        o = self.sigmoid(self.z3) #final activation function
        return  o

    def sigmoidPrime(self, s):
        #derivation of sigmoid
        return s * (1 - s)

    def backward(self, X , y, o):
        #backward propagation through the network
        self.o_error = y - o #error in output
        self.o_delta = self.o_error * self.sigmoidPrime(o) #applying derivation of sigmoid to error

        self.z2_error = self.o_delta.dot(self.W2.T) #z2 error : how mych out hidden layer weights contributed to output error
        self.z2_delta = self.z2_error*self.sigmoidPrime(self.z2) #applying derivative of sigmoid to z2 error

        self.W1 += X.T.dot(self.z2_delta) #adjusting first set (input --> hidden) weights
        self.W2 += self.z2.T.dot(self.o_delta) #adjusting second set (hidden --> output) weights

    def train(self, X, y):
        o = self.forward(X)
        self.backward(X,y,o)

NN = NeuralNetwork()
for i in range(1000): #trains the NN 1000 times
    print("Input: \n" + str(X))
    print("Actual Output: \n" + str(y))
    print("Predicted Output: \n" + str(NN.forward(X)))
    print("Loss: \n" + str(np.square(y - NN.forward(X)))) #mean sum square loss
    print("\n")
    NN.train(X, y)

#defining our outputs
o = NN.forward(X)

print("Predicted Output: \n" + str(o))
print("Actual Output: \n" + str(y))
