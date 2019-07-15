import filter
import numpy as np
import re
import matplotlib.pyplot as plt

def double_int(a, dt=0.01):
    s = [0] * len(a)
    v = [0] * len(a)
    for i in range(1, len(a)):
        dv = a[i] * dt
        ds = v[i - 1] * dt + 0.5 * a[i] * dt * dt
        s[i] = s[i - 1] + ds
        v[i] = v[i - 1] + dv
    return s, v

acc = filter.read_file('move_15cm_3.txt')

acc_rmv_offset = filter.remove_offset(
    acc,
    noise_power=0.0002,
    window_range = 0.4)

acc_lpf = acc_rmv_offset
acc_lpf = np.convolve(acc_lpf, filter.gen_maf(0.01), 'valid')
#acc_lpf = np.convolve(acc_lpf, filter.gen_lpf(0.01), 'valid')

s, v = double_int(acc_lpf)

plt.subplot(311)
plt.plot(acc, label='acc')
plt.plot(acc_lpf, label='acc_lpf')
plt.ylabel('Signal')
plt.xlabel('Time')
plt.legend(loc='upper right')

plt.subplot(312)
plt.plot(v, label='v')
plt.ylabel('Signal')
plt.xlabel('Time')
plt.legend(loc='upper right')

plt.subplot(313)
plt.plot(s, label='s')
plt.ylabel('Signal')
plt.xlabel('Time')
plt.legend(loc='upper right')

plt.show()

