import numpy as mp
import matplotlib.pyplot as plt

# 데이터 준비
x = np.arrange(0,6,0.1) #0에서 6까지 0.1간격으로 생성
y = np.sin(x)

#그래프 그리기
plt.plot(x,y)
plt.show()
