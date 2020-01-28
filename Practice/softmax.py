import numpy as np
"""
a = np.array([0.3, 2.9, 4.0])

exp_a = np.exp(a) #지수 함수
print(exp_a)

sum_exp_a = np.sum(exp_a)
print(sum_exp_a)

y = exp_a / sum_exp_a

print(y)
"""
"""
def softmax(a):
    exp_a = np.exp(a) #지수 함수
    sum_exp_a = np.sum(exp_a)
    output = exp_a / sum_exp_a

    return output
"""
"""
def softmax(a):
    max = np.max(a)
    value = a - max
    output = np.exp(value) / np.sum(np.exp(value))
    return output

a = np.array([1010, 1000, 990])
print(softmax(a))
"""
def softmax(a): #final
    c = np.max(a)
    exp_a = np.exp(a-c)
    sum_exp_a = np.sum(exp_a)
    output = exp_a / sum_exp_a
    return output


a = np.array([0.3, 2.9, 4.0])
y = softmax(a)
print(y)
print(np.sum(y))
