import filter
import numpy as np
import re
import matplotlib.pyplot as plt

acc_raw = filter.read_file('reps_20.txt')
bpf = filter.gen_bpf(0.005, 0.025, 0.01)
print("bpf len: " + str(len(bpf)))
acc = np.convolve(acc_raw, bpf, 'same')
#acc = acc_raw
acc = filter.remove_offset(
    acc,
    noise_power=0.0002,
    window_range = 0.4)

vel = filter.integrate_trap(acc)
# doesn't seem to be neccesary because there's no significant drift in vel
hpf = filter.gen_hpf(0.001, 0.01)
print("hpf len: " + str(len(hpf)))
vel = np.convolve(vel, filter.gen_hpf(0.001, 0.01), 'same')

# doesn't seem to be neccesary because there's no significant noise
lpf = filter.gen_lpf(0.01, 0.02)
#vel_x = np.convolve(vel, lpf, 'same')
    
vel_x = filter.remove_offset(
    vel,
    noise_power=0.00002,
    window_range = 0.01)

dis = filter.integrate_trap(vel)
#dis = np.convolve(dis, hpf, 'same')

plt.figure(1)
plt.subplot(311)
plt.plot(acc_raw, label='acc_raw')
plt.plot(acc, label='acc_lpf')
plt.ylabel('Signal')
plt.xlabel('Time')
plt.legend(loc='upper right')

plt.subplot(312)
plt.plot(vel, label='vel')
plt.ylabel('Signal')
plt.xlabel('Time')
plt.legend(loc='upper right')

plt.subplot(313)
plt.plot(dis, label='dis')
plt.ylabel('Signal')
plt.xlabel('Time')
plt.legend(loc='upper right')

plt.figure(2)
plt.subplot(311)
plt.plot(bpf, label='bpf')
plt.ylabel('Signal')
plt.xlabel('Time')
plt.legend(loc='upper right')

plt.subplot(312)
plt.plot(lpf, label='lpf')
plt.ylabel('Signal')
plt.xlabel('Time')
plt.legend(loc='upper right')

plt.subplot(313)
plt.plot(hpf, label='hpf')
plt.ylabel('Signal')
plt.xlabel('Time')
plt.legend(loc='upper right')

plt.show()
