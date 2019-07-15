from __future__ import division
 
import numpy as np
import re
import matplotlib.pyplot as plt
import math

def read_file(file_name):
    reg = '(\s*-?\d+\.\d+\s*)(?=,)'
    data = []
    f = open(file_name, 'r')
    for line in f:
        if 'ACCEL' in line.upper():
            m = re.search(reg, line)
            num = float(m.group(0))
            data.append(num)
    f.close()
    return data

# https://tomroelandts.com/articles/how-to-create-a-simple-low-pass-filter
def gen_lpf(fc = 0.1, b = 0.08):
    # Cutoff frequency as a fraction of the sampling rate (in (0, 0.5)).
    # Transition band, as a fraction of the sampling rate (in (0, 0.5)).
    N = int(np.ceil((4 / b)))
    if not N % 2: N += 1  # Make sure that N is odd.
    n = np.arange(N)
     
    # Compute sinc filter.
    h = np.sinc(2 * fc * (n - (N - 1) / 2))
     
    # Compute Blackman window.
    w = 0.42 - 0.5 * np.cos(2 * np.pi * n / (N - 1)) + \
        0.08 * np.cos(4 * np.pi * n / (N - 1))
    # w = np.blackman(N)

    # Multiply sinc filter by window.
    h = h * w
     
    # Normalize to get unity gain.
    h = h / np.sum(h)
    return h

def gen_hpf(fc = 0.1, b = 0.08):
    # Cutoff frequency as a fraction of the sampling rate (in (0, 0.5)).
    # Transition band, as a fraction of the sampling rate (in (0, 0.5)).
    N = int(np.ceil((4 / b)))
    if not N % 2: N += 1  # Make sure that N is odd.
    n = np.arange(N)
     
    # Compute a low-pass filter.
    h = gen_lpf(fc, b)
     
    # Create a high-pass filter from the low-pass filter through spectral inversion.
    h = -h
    h[(N - 1) // 2] += 1
    return h

# https://tomroelandts.com/articles/how-to-create-simple-band-pass-and-band-reject-filters
def gen_bpf(fL = 0.1, fH = 0.4, b = 0.08):
    # Cutoff frequency as a fraction of the sampling rate (in (0, 0.5)).
    # Cutoff frequency as a fraction of the sampling rate (in (0, 0.5)).
    # Transition band, as a fraction of the sampling rate (in (0, 0.5)).
    N = int(np.ceil((4 / b)))
    if not N % 2: N += 1  # Make sure that N is odd.
    n = np.arange(N)
     
    # Compute a low-pass filter with cutoff frequency fH.
    hlpf = gen_lpf(fH, b)
     
    # Compute a high-pass filter with cutoff frequency fL.
    hhpf = gen_hpf(fL, b)
     
    # Convolve both filters.
    h = np.convolve(hlpf, hhpf)
    return h

def gen_brf(fL = 0.1, fH = 0.4, b = 0.08):
    N = int(np.ceil((4 / b)))
    if not N % 2: N += 1  # Make sure that N is odd.
    n = np.arange(N)
     
    # Compute a low-pass filter with cutoff frequency fL.
    hlpf = gen_lpf(fL, b)
     
    # Compute a high-pass filter with cutoff frequency fH.
    hhpf = gen_hpf(fH, b)
     
    # Convolve both filters.
    h = np.convolve(hlpf, hhpf, 'same')
    return h

def gen_maf(fc):
    # Cutoff frequency as a fraction of the sampling rate (in (0, 0.5)).
    N = int(np.ceil(pow(0.2 + fc * fc, 2) / fc))
    if not N % 2: N += 1  # Make sure that N is odd.
    maf = [1.0 / N] * N
    return maf

def calc_pwr(signal):
    pwr = 0
    for s in signal:
        pwr += pow(s, 2)
    return pwr / len(signal)

def integrate(signal, dt=0.01):
    ret = [0]
    integral = 0;
    for i in range(len(signal)):
        integral = ret[-1] + signal[i] * dt
        ret.append(integral)
    return ret

def integrate_trap(signal, dt=0.01):
    ret = [0]
    integral = 0;
    for i in range(1, len(signal)):
        rect = min(signal[i], signal[i - 1]) * dt
        tri = abs(signal[i] - signal[i - 1]) * dt / 2
        integral = ret[-1] + rect + tri
        ret.append(integral)
    return ret

def remove_offset(
    signal,
    dt=0.01,
    window_size=21,
    noise_power=0.0002,
    window_range = 0.04,
    debug=False):
    ret = np.array(signal)
    offset = 0;
    powers = []
    for i in range(0, len(ret) - window_size):
        window = signal[i : i + window_size]
        window_power = calc_pwr(window)
        window_mean = np.mean(window)
        win_range = np.max(window) - np.min(window)
        window_no_offset = window - window_mean
        snr = window_power / noise_power
        power = calc_pwr(window_no_offset)
        powers.append(power)
        if power < noise_power and win_range < window_range: # noise power is about 0.0001
            ret[i] = 0
            #offset = window_mean
        else:
            ret[i] = signal[i]
    if debug:
        return ret, powers
    else:
        return ret

def main():
    #acc = read_file('static.txt')
    acc = read_file('move_15cm.txt')
    
    lpf = gen_lpf(0.01, 0.16)
    hpf = gen_hpf(0.00001, 0.08)
    bpf = gen_bpf(0.001, 0.01, 0.16)
    brf = gen_brf(0.001, 0.1, 0.2)
    maf = gen_maf(0.01)

    mean_acc = np.mean(acc)
    print("mean_acc: %f" % mean_acc)
    pwr_acc = calc_pwr(acc - mean_acc)
    print("pwr_acc: %f" % pwr_acc)
    
    pwr_lpf = calc_pwr(lpf)
    print("pwr_lpf: %.3f" % pwr_lpf)
    print(lpf)

    pwr_hpf = calc_pwr(hpf)
    print("pwr_hpf: %.3f" % pwr_hpf)
    print(hpf)

    pwr_bpf = calc_pwr(bpf)
    print("pwr_bpf: %.3f" % pwr_bpf)
    print(bpf)    

    #print(','.join(map(str, brf)))
    #print(','.join(map(str, lpf)))

    acc_rmv_offset = remove_offset(acc)
    acc_lpf = np.convolve(acc, lpf, 'valid')
    acc_hpf = np.convolve(acc, hpf, 'valid')
    acc_brf = np.convolve(acc, brf, 'valid')
    acc_bpf = 20 * np.convolve(acc, bpf, 'valid')
    acc_maf = np.convolve(acc, maf, 'valid')

    # either seems fine
    acc_cus = np.convolve(acc_rmv_offset, lpf, 'valid')
    #acc_cus = np.convolve(acc_rmv_offset, maf, 'valid')

    plt.subplot(311)
    plt.plot(acc, 'r', label='raw')
    #plt.plot(acc_lpf, 'g', label='LPF')
    #plt.plot(acc_hpf, 'b', label='HPF')
    #plt.plot(acc_brf, 'c', label='BRF')
    #plt.plot(acc_maf, 'y', label='MAF')
    #plt.plot(acc_bpf, 'c', label='BPF')
    #plt.plot(acc_rmv_offset, 'g', label='RMV OFFSET')
    plt.plot(acc_cus, label='CUS')
    plt.ylabel('Signal')
    plt.xlabel('Time')
    plt.legend(loc='upper right')

    plt.subplot(312)
    v = integrate_trap(acc_cus)
    v_rmv_offset = remove_offset(v)
    v_hpf = np.convolve(v, hpf)
    plt.plot(v, label='Velocity')
    plt.plot(v_rmv_offset, label='Velocity RMVOFFS')
    #plt.plot(v_hpf, label='Velocity HPF')
    plt.ylabel('Signal')
    plt.xlabel('Time')
    plt.legend(loc='upper right')
    
    plt.subplot(313)
    d = integrate_trap(v_rmv_offset)
    #d_hpf = np.convolve(d, hpf)
    plt.plot(d, 'b', label='Displacement')
    #plt.plot(d_hpf, 'r', label='Displacement HPF')
    plt.ylabel('Signal')
    plt.xlabel('Time')
    plt.legend(loc='upper right')

    plt.show()

if __name__ == '__main__':
    main()
