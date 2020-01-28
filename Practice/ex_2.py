import numpy as np

def nonlin(x,deriv=False):
    if(deriv==True):    #deriv
        return x*(1-x)

    return 1/(1+np.exp(-x))    #sigmoid

X = np.array([  [0,0],
                [0,1],
                [1,0],
                [1,1]])

#xor output
y = np.array([    [0],
        [1],
        [1],
        [0]])
np.random.seed(1)

hidden = 32
#syn0 = 2*np.random.random((2,hidden)) - 1
syn0 = np.zeros((2,hidden))+0.8    #good init

syn1 = 2*np.random.random((hidden,1)) - 1
#alpha = 1

l0 = np.zeros((1,2))
l1 = np.zeros((1,hidden))
l2 = np.zeros((1,1))

for j in range(50000):
    for i in range(len(X)):
        # Feed forward through layers 0, 1, and 2
        l0 = X[i]
        l1 = nonlin(np.dot(l0,syn0))
        l2 = nonlin(np.dot(l1,syn1))

        l2_error = y[i] - l2
        l2_delta = l2_error*nonlin(l2,deriv=True)

        if (j% 10000) == 0:
            print ((j),": Error:" + str(np.mean(np.abs(l2_error) ) ) )
            #alpha -= (np.random.random()/8)

        l1_error = l2_delta.dot(syn1.T)
        l1_delta = l1_error * nonlin(l1,deriv=True)

        syn1 += np.reshape(l1,(-1,1))*l2_delta
        syn0 += np.reshape(l0,(-1,1))*l1_delta

print("---- result ----")
#evaluate
for i in range(len(X)):
    l0 = X[i]
    l1 = nonlin(np.dot(l0,syn0))
    l2 = nonlin(np.dot(l1,syn1))

    print(X[i],":", l2)
