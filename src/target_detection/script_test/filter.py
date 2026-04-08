import numpy as np
from collections import deque
from scipy.signal import butter, sosfilt, sosfilt_zi

class Median5:
    """
    固定 5 点滑动中值，online，O(1)
    用法：
        m = Median5()
        y = m.update(x_new)
    """
    def __init__(self):
        self.buf = deque(maxlen=5)   # 永远保持 5 个元素

    def update(self, x):
        self.buf.append(x)
        return np.median(self.buf)   # 5 点排序，耗时 < 1 µs @Python

class LowPassRealTime:
    """
    2 阶 Butterworth 低通，online，带状态
    用法：
        lpf = LowPassRealTime(fs, fc)
        y = lpf.update(x)
    """
    def __init__(self, fs, fc):
        self.sos = butter(2, fc, btype='low', fs=fs, output='sos')
        self.z  = sosfilt_zi(self.sos)  # 初始状态

    def update(self, x):
        y, self.z = sosfilt(self.sos, [x], zi=self.z)
        return y[0]
    

class MedianThenLowPass:
    def __init__(self, fs, fc):
        self.median = Median5()
        self.lpf    = LowPassRealTime(fs, fc)

    def update(self, x_raw):
        x_med = self.median.update(x_raw)
        x_lpf = self.lpf.update(x_med)
        return x_lpf