"""
def step_function(x):

    if x > 0:
        return 1
    else:
        return 0

print(step_function(2))
"""
"""
import numpy as np
def step_function(x):
    y = x > 0
    return y.astype(np.int)

x = np.array([-1.0, 1.0, 2.0])

print(step_function(x))
"""

import numpy as np
import matplotlib.pylab as plt

def step_function(x):
    return np.array(x > 0, dtype = np.int)

x = np.array(-5.0, 5.0, 0.1)
y = step_function(x)
plt.plot(x,y)
plt.ylim(-0.1, 1.1)
plt.show()
