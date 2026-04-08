import numpy as np
import matplotlib.pyplot as plt

fs   = 10                       # Hz
T    = 20                       # s
N    = int(fs * T)
t    = np.arange(N) / fs

# 1. 真实趋势（低频正弦 + 线性漂移）
trend = 3 * np.sin(2 * np.pi * 0.2 * t) + 0.05 * t

# 2. 高频毛刺（白噪声）
noise = np.random.normal(0, 0.3, N)

# 3. 偶发大脉冲（平均 2 s 一个）
pulse = np.zeros(N)
pulse_idx = np.arange(15, N, 20)          # 每 20 点来一个
pulse[pulse_idx] = np.random.choice([-1, 1], len(pulse_idx)) * \
                   np.random.uniform(8, 12, len(pulse_idx))

# 4. 合成原始数据
raw = trend + noise + pulse

# 5. 保存/直接返回
# np.savetxt('test_data.txt', raw, fmt='%.3f')

from filter import MedianThenLowPass   # 把你写的类引进来
proc = MedianThenLowPass(fs=10, fc=1)

filtered = np.array([proc.update(x) for x in raw])

plt.plot(t, raw,    lw=1, label='raw')
plt.plot(t, filtered,lw=2, label='online median+LPF')
plt.legend(); plt.show()