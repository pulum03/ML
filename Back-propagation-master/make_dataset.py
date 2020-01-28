from random import *
import numpy as np

# 0~9 사이 랜덤 값을 반환
def ran():
    return randint(0,9)

# 단일 데이터 생성
def makeData():
    result = list()
    one = ran()
    for i in range(10):
        result.append("0")

    result[one] = "1"

    result.extend(makeY(one+1))
    return result

# 2진수를 4자리로 바꾸고, 한글자씩 분리
def makeY(one):
    return list(("0000"+dec2bin(one))[-4:])

# 10진수를 2진수로 변환
def dec2bin(dec):
    return "{0:b}".format(dec)

# 학습 데이터 생성
def makeTrainDatas(count):
    f = open('train_data.txt', 'w')
    for i in range(count):
        data = makeData()
        f.write(",".join(data) + "\n")
    f.close()

# 검증 데이터 생성
def makeTestDatas(count):
    f = open('test_data.txt', 'w')
    for i in range(count):
        data = makeData()
        f.write(",".join(data) + "\n")
    f.close()

# 50개의 학습 데이터, 10만개의 검증 데이터 생성
makeTrainDatas(50)
makeTestDatas(100000)
