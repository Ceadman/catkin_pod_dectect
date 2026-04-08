import numpy as np
from collections import deque
from scipy.signal import butter, sosfilt, sosfilt_zi

# ---------------- 3 通道 5 点滑动中值 ----------------
class Median5_3D:
    def __init__(self):
        # 每条通道各自维护一个 deque
        self.buf = [deque(maxlen=5) for _ in range(3)]

    def update(self, x: np.ndarray) -> np.ndarray:
        """x: shape=(3,)  返回同 shape 的滤波结果"""
        out = np.empty(3)
        for i in range(3):
            self.buf[i].append(x[i])
            out[i] = np.median(self.buf[i])
        return out


# ---------------- 3 通道 2 阶 Butterworth 低通 ----------------
class LowPassRealTime3D:
    def __init__(self, fs, fc):
        sos = butter(2, fc, btype='low', fs=fs, output='sos')
        self.sos = sos                                  
        self.sos_zi   = sosfilt_zi(sos)                     
        self.sos_zi   = np.tile(self.sos_zi, (3, 1, 1, 1))

    def update(self, x: np.ndarray) -> np.ndarray:
        """x: shape=(3,)  返回同 shape 的滤波结果"""
        y = np.empty(3)
        for i in range(3):
            y[i], self.sos_zi[i] = sosfilt(self.sos, [[x[i]]], zi=self.sos_zi[i])
        return y


# ---------------- 3 通道 “中值+低通” 级联 ----------------
class MedianThenLowPass3D:
    def __init__(self, fs, fc):
        self.median = Median5_3D()
        self.lpf    = LowPassRealTime3D(fs, fc)

    def update(self, xyz_raw: np.ndarray) -> np.ndarray:
        xyz_med = self.median.update(xyz_raw)
        xyz_lpf = self.lpf.update(xyz_med)
        return xyz_lpf


if __name__ == "__main__":
    fs, fc = 200, 5.0          # 采样频率fc，低通截止频率fc
    filter   = MedianThenLowPass3D(fs, fc)

    # 模拟一条实时数据流
    for k in range(1000):
        xyz_new = np.random.randn(3)
        xyz_flt = filter.update(xyz_new) 
        print(k, xyz_flt)