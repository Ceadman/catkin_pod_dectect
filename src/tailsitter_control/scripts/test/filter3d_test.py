# test_median_lpf.py
import numpy as np
import matplotlib.pyplot as plt
from filter3d import MedianThenLowPass3D  # 把上面类保存成同名文件

# ---------------- 参数 ----------------
fs   = 200          # Hz
fc   = 5            # 低通截止频率
N    = 1000         # 总采样点数
t    = np.arange(N) / fs

# ---------------- 构造脏信号 ----------------
# 3 通道正弦 2 Hz
clean = np.sin(2*np.pi*2*t)[:,None] * np.array([1, 1.2, 0.8])
# 加高频噪声
noise = 0.3 * np.random.randn(N, 3)
# 随机脉冲
impulse = np.zeros((N, 3))
for _ in range(15):
    ch = np.random.randint(3)
    impulse[np.random.randint(N), ch] = np.random.uniform(5, 10)
raw = clean + noise + impulse  # shape=(N,3)

# ---------------- 逐点滤波 ----------------
filt = MedianThenLowPass3D(fs, fc)
med_buf   = np.empty_like(raw)
lpf_buf   = np.empty_like(raw)
for i in range(N):
    med_buf[i] = filt.median.update(raw[i])
    lpf_buf[i] = filt.lpf.update(med_buf[i])

# ---------------- 画图 ----------------
plt.figure(figsize=(10,6))
channels = ['X','Y','Z']
for ch in range(3):
    plt.subplot(3,1,ch+1)
    plt.plot(t, raw[:,ch],   alpha=0.6, label='raw')
    plt.plot(t, med_buf[:,ch], label='median')
    plt.plot(t, lpf_buf[:,ch], lw=2, label='median+lpf')
    plt.ylabel(channels[ch])
    plt.legend()
plt.xlabel('time [s]')
plt.suptitle('Real-time Median5 → Butterworth2-LowPass')
plt.tight_layout()
plt.show()