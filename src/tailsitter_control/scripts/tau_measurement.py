import rospy
import rosbag, numpy as np, matplotlib.pyplot as plt
from scipy.optimize import curve_fit

bag = rosbag.Bag('/home/ceadman111/tau_test.bag')
t,v = [],[]
t0 = None
t_start = rospy.Time(253.893)  
t_end   = rospy.Time(265.393) 

for topic, msg, ts in bag.read_messages(start_time=t_start, end_time=t_end):
    if topic.endswith('/setpoint_raw/local'):
        v_cmd = msg.velocity.z
        if t0 is None and abs(v_cmd)>0.1: t0 = ts.to_sec()  # 阶跃起点
    if topic.endswith('/local_position/velocity_local') and t0 is not None:
        t.append(ts.to_sec()-t0)
        v.append(msg.twist.linear.z)
bag.close()
t = np.array(t); v = np.array(v)

def model(t, tau, vinf, t_shift):
    return vinf * (1 - np.exp(-(t-t_shift)/tau)) * (t>=t_shift)

p0 = [1.0, 2.0, 0.0]          # 初值：tau=1 s, 终值=2 m/s
p,_ = curve_fit(model, t, v, p0)
tau = p[0]
print('实测 tau = %.2f s' % tau)

# 画图验证
plt.plot(t, v, label='measured')
plt.plot(t, model(t,*p), label='fit τ=%.2f s'%tau)
plt.legend(); plt.show()





# import rospy
# import rosbag, numpy as np

# bag = rosbag.Bag('/home/ceadman111/tau_test.bag')

# # 1. 先确认 bag 里到底有哪些话题
# print('bag 里所有话题:')
# for tp in sorted(bag.get_type_and_topic_info()[1].keys()):
#     print('  ', tp)

# setpoint_found = False
# velocity_found = False
# t0 = None
# t = [] 
# v = []
# t_start = rospy.Time(253.893)   # 比 bag 实际 start 晚 2 s
# t_end   = rospy.Time(265.393)   # 比 bag 实际 end  早 2 s

# for topic, msg, ts in bag.read_messages(start_time=t_start, end_time=t_end):
#     # ---------- setpoint ----------
#     if topic.endswith('/tailsitter_0/mavros/setpoint_raw/local'):
#         setpoint_found = True
#         v_cmd = msg.velocity.z
#         print(f'setpoint  y={v_cmd:.3f}  t={ts.to_sec():.3f}')
#         if t0 is None and abs(v_cmd) > 0.1:
#             t0 = ts.to_sec()
#             print('>>> 阶跃起点 t0 =', t0)

#     # ---------- velocity ----------
#     if topic.endswith('/tailsitter_0/mavros/local_position/velocity_local'):
#         velocity_found = True
#         if t0 is not None:
#             t.append(ts.to_sec() - t0)
#             v.append(msg.twist.linear.z)
#         else:
#             print(f'velocity  y={msg.twist.linear.y:.3f}  但 t0 还是 None')

# bag.close()

# print('\n---- 总结 ----')
# print('setpoint 出现过?', setpoint_found)
# print('velocity 出现过?', velocity_found)
# print('t0 赋值成功?', t0 is not None)
# print('最终抓到点数:', len(v))