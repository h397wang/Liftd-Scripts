from __future__ import division
 
import filter

import numpy as np
import re
import matplotlib.pyplot as plt


raw = np.array((filter.read_file('static.txt')))
grad = np.gradient(raw)
m = max(grad)
diff = raw - grad
print(m)

plt.plot(raw, label='RAW')
plt.plot(grad, label='GRAD')
plt.plot(diff, label='DIFF')
plt.ylabel('Signal')
plt.xlabel('Time')
plt.legend(loc='upper right')

"""
raw_power = filter.calc_pwr(raw)
grad_power = filter.calc_pwr(grad)
grad_grad_power = filter.calc_pwr(grad_grad)
print(raw_power)
print(grad_power)
print(grad_grad_power)
"""
plt.show()

