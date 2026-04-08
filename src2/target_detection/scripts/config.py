
# fs：改成采样频率, 降延迟(但噪声过滤能力降低)：将lowpass_order = 2改为lowpass_order = 1 或是 将fc = 5.0 升到 fc = 20.0
fs, fc = 30, 5.0
lowpass_order = 1

# PID
x_kp_c, x_ki_c, x_kd_c = 3, 0.001, 0.1
y_kp_c, y_ki_c, y_kd_c = 3, 0.001, 0.1
