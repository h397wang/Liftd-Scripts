import filter
import numpy as np
import re
import matplotlib.pyplot as plt

# if acc is steady.. then it's likely that it's zero but with offset
# we can just zero the velocity as well because we dont have constant vel in our data..

# acceleration curve states
STATIC = 0 # accel is zero
UP = 1
DOWN = 2
NUM_STATES = 3
window_len = 15

lpf = filter.gen_lpf(0.01, 4.0 / window_len)

class FSM():
    def __init__(self, acc_raw):    
        self.state_ = STATIC
        self.dt_ = 0.01
        # debugging purposes, Arduino code would use window
        self.vel_ = [0] 
        self.dis_ =[0]
        self.states_ = [0]
        self.acc_raw_ = acc_raw
        self.acc_filtered_ = [0] * (window_len - 1)
        # other constants

    def trap_area(self, x0, x1):
        rect = min(x1, x0) * self.dt_
        tri = abs(x1 - x0) * self.dt_ / 2
        return rect + tri        

    # takes us from one state to another, and does the action
    # output is a vel and dis value
    # maybe ammend historical data points...
    # it should be sinusoidal...
    def transition(self, acc_filtered_window, next_state):
        next_acc = acc_filtered_window[-1]
        dv = self.trap_area(acc_filtered_window[-2], acc_filtered_window[-1])
        next_vel = self.vel_[-1] + dv
        ds = self.trap_area(self.vel_[-1], next_vel)
        #ds = self.vel_[-1] * self.dt_ + 0.5 * next_acc * self.dt_ * self.dt_
        next_dis = self.dis_[-1] + ds

        if self.state_ == STATIC:
            if next_state == STATIC:
                next_acc = 0
                next_vel = 0
                next_dis = self.dis_[-1]
                pass
            elif next_state == UP:
                pass
            elif next_state == DOWN:
                pass
            else:
                assert(0)
        elif self.state_ == UP:
            if next_state == STATIC:
                pass
            elif next_state == UP:
                pass
            elif next_state == DOWN:
                pass
            else:
                assert(0)
        elif self.state_ == DOWN:
            if next_state == STATIC:
                pass
            elif next_state == UP:
                pass
            elif next_state == DOWN:
                pass
            else:
                assert(0)
        else:
            assert(0)
        if (next_state != self.state_):
            print("Transitioning from " + str(self.state_) + " to " + str(next_state))
        self.state_ = next_state
        self.states_.append(next_state)
        self.acc_filtered_[-1] = next_acc
        self.vel_.append(next_vel)
        self.dis_.append(next_dis)

    # input is a window filtered of high frequency noise
    # takes us to the next state, and modifies values?
    # this function should probably just determine the next state
    def update(self, acc_window):
        acc_window_mean = np.mean(acc_window)
        acc_window_rmvoffset = acc_window - acc_window_mean
        pwr_rmvoffset = filter.calc_pwr(acc_window_rmvoffset)
        acc_window_max = max(acc_window)
        acc_window_min = min(acc_window)
        acc_window_p2p = abs(acc_window_max - acc_window_min)
        acc_window_slope = acc_window[-1] - acc_window[0]

        next_states_count = [0] * NUM_STATES
        if self.state_ == STATIC:
            if acc_window_p2p < 0.04: # likely static
                next_states_count[STATIC] += + 1
            if abs(acc_window_slope) < 0.04: # likely static
                next_states_count[STATIC] += 1
            if pwr_rmvoffset < 0.002: # likely static
                next_states_count[STATIC] += 1

            if acc_window_slope > 0.04:
                next_states_count[UP] += 1
            if acc_window_max == acc_window[-1]:
                next_states_count[UP] += 1
            if acc_window[-1] >= acc_window[-2] and acc_window[-2] >= acc_window[-3]:
                next_states_count[UP] += 1

            if acc_window_slope < -0.04: # likely to be DOWN
                next_states_count[DOWN] += 1
            if acc_window_min == acc_window[-1]: # likely to be DOWN
                next_states_count[DOWN] += 1
            if acc_window[-1] <= acc_window[-2] and acc_window[-2] <= acc_window[-3]:
                next_states_count[DOWN] += 1

        elif self.state_ == UP:
            # how to avoid copy pasta
            if acc_window_slope > 0.04:
                next_states_count[UP] += 1
            if acc_window_max == acc_window[-1]:
                next_states_count[UP] += 1
            if acc_window[-1] >= acc_window[-2] and acc_window[-2] >= acc_window[-3]:
                next_states_count[UP] += 1

            if acc_window_slope < -0.04: # likely to be DOWN
                next_states_count[DOWN] += 1
            if acc_window_min == acc_window[-1]: # likely to be DOWN
                next_states_count[DOWN] += 1
            if acc_window[-1] <= acc_window[-2] and acc_window[-2] <= acc_window[-3]:
                next_states_count[DOWN] += 1

            if acc_window_p2p < 0.04: # likely static
                next_states_count[STATIC] += + 1
            if abs(acc_window_slope) < 0.04: # likely static
                next_states_count[STATIC] += 1
            if pwr_rmvoffset < 0.002: # likely static
                next_states_count[STATIC] += 1

        elif self.state_ == DOWN:
            if acc_window_slope > 0.04:
                next_states_count[UP] += 1
            if acc_window_max == acc_window[-1]:
                next_states_count[UP] += 1
            if acc_window[-1] >= acc_window[-2] and acc_window[-2] >= acc_window[-3]:
                next_states_count[UP] += 1

            if acc_window_slope < -0.04: # likely to be DOWN
                next_states_count[DOWN] += 1
            if acc_window_min == acc_window[-1]: # likely to be DOWN
                next_states_count[DOWN] += 1
            if acc_window[-1] <= acc_window[-2] and acc_window[-2] <= acc_window[-3]:
                next_states_count[DOWN] += 1

            if acc_window_p2p < 0.04: # likely static
                next_states_count[STATIC] += + 1
            if abs(acc_window_slope) < 0.04: # likely static
                next_states_count[STATIC] += 1
            if pwr_rmvoffset < 0.002: # likely static
                next_states_count[STATIC] += 1
        else:
            assert(0)
        # STATIC has highest priority in ties
        max_count = 0
        ns = self.state_
        for state in range(NUM_STATES):
            state_counts = next_states_count[state]
            if state_counts >= max_count:
                ns = state
                max_count = state_counts
        return ns

    def maf(self, acc_raw_window):
        total = 0
        for acc in acc_raw_window:
            total += acc
        return total / len(acc_raw_window)

    def loop(self):
        for i in range(window_len - 1, len(self.acc_raw_)):
            acc_raw_val = self.acc_raw_[i]
            acc_raw_window = self.acc_raw_[i + 1 - window_len : i + 1]
            assert(len(acc_raw_window) == window_len)
            
            acc_val = self.maf(acc_raw_window)
            self.acc_filtered_.append(acc_val)
            # returns a single value
            #acc_lpf_val = np.convolve(acc_raw_window, lpf, 'valid')
            #self.acc_filtered_[i - window_len]

            acc_filtered_window = self.acc_filtered_[i + 1 - window_len : i + 1]

            # extra smoothing??
            # acc_val_2 = self.maf(acc_filtered_window)
            #acc_filtered_window[-1] = acc_val_2

            assert(len(acc_filtered_window) == window_len)
            ns = self.update(acc_filtered_window)
            self.transition(acc_filtered_window, ns)

acc = filter.read_file('reps.txt')
fsm = FSM(acc)
fsm.loop()

plt.subplot(411)
plt.plot(fsm.acc_filtered_, label='acc_filtered_')
plt.plot(acc, label='acc')
plt.ylabel('Signal')
plt.xlabel('Time')
plt.legend(loc='upper right')

plt.subplot(412)
plt.plot(fsm.vel_, label='vel_')
plt.ylabel('Signal')
plt.xlabel('Time')
plt.legend(loc='upper right')

plt.subplot(413)
plt.plot(fsm.dis_, label='dis_')
plt.ylabel('Signal')
plt.xlabel('Time')
plt.legend(loc='upper right')

plt.subplot(414)
plt.plot(fsm.states_, label='states_')
plt.ylabel('State')
plt.xlabel('Time')
plt.legend(loc='upper right')

plt.show()
