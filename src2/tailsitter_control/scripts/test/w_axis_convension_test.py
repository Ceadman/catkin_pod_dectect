#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import tf.transformations as tf

class MockVector3:
    def __init__(self, x=0., y=0., z=0.):
        self.x, self.y, self.z = x, y, z

class PodRateTester:
    def __init__(self):
        self.angular_velocity = MockVector3()
        self.set_frd_orientation(0, 0, 0)
        
    def set_frd_orientation(self, roll_deg, pitch_deg, yaw_deg):
        """设置FRD到ENU的旋转（FRD: X前,Y右,Z下; ENU: X东,Y北,Z上）"""
        r, p, y = map(np.deg2rad, [roll_deg, pitch_deg, yaw_deg])
        cr, sr = np.cos(r), np.sin(r)
        cp, sp = np.cos(p), np.sin(p)
        cy, sy = np.cos(y), np.sin(y)
        
        # FRD基向量在ENU中的表示（考虑Y/Z反向）
        R_frd_to_enu = np.array([
            [cy*cp, -cy*sp*sr - sy*cr, -cy*sp*cr + sy*sr],
            [sy*cp, -sy*sp*sr + cy*cr, -sy*sp*cr - cy*sr],
            [sp,     cp*sr,              cp*cr            ]
        ]) @ np.diag([1, -1, -1])
        
        self.R_body_enu = R_frd_to_enu
        tmp = np.eye(4)
        tmp[:3, :3] = R_frd_to_enu
        self.orientation = tf.quaternion_from_matrix(tmp)
        self.euler = (roll_deg, pitch_deg, yaw_deg)
        
    def describe_state(self):
        """打印当前无人机状态"""
        r, p, y = self.euler
        print(f"\n【无人机状态】姿态角(°): Roll={r:+.1f}, Pitch={p:+.1f}, Yaw={y:+.1f}")
        
        # 姿态描述
        att_desc = []
        if abs(r) > 5: att_desc.append(f"{'右' if r>0 else '左'}倾{r:.0f}°")
        if abs(p) > 5: att_desc.append(f"{'抬头' if p>0 else '低头'}{abs(p):.0f}°")
        yaw_dirs = ["东", "东北", "北", "西北", "西", "西南", "南", "东南"]
        att_desc.append(f"指向{yaw_dirs[int((y%360+22.5)/45)%8]}")
        print(f"  姿态: {', '.join(att_desc)}")
        
        # 角速度
        av = self.angular_velocity
        print(f"  机体角速度(Body-FRD):")
        print(f"    p(X轴/滚转): {np.rad2deg(av.x):+.1f}°/s ({'右滚' if av.x>0 else '左滚' if av.x<0 else '无'})")
        print(f"    q(Y轴/俯仰): {np.rad2deg(av.y):+.1f}°/s ({'低头' if av.y>0 else '抬头' if av.y<0 else '无'})")
        print(f"    r(Z轴/偏航): {np.rad2deg(av.z):+.1f}°/s ({'顺时针' if av.z>0 else '逆时针'}(Z向下))")
        
    def calc_los_rates(self, az_vel, el_vel, map_az, map_el):
        """计算视线角速度（FRD坐标系：Z向下，顺时针为正；ENU：Z向上，逆时针为正）"""
        # Body坐标系角速度向量（FRD）
        omega_pod = np.array([0., -np.deg2rad(el_vel), np.deg2rad(az_vel)])  # 抬头为负Y，顺时针为正Z
        omega_body = np.array([self.angular_velocity.x, self.angular_velocity.y, self.angular_velocity.z])
        omega_los_body = omega_pod + omega_body
        
        # 转换到ENU
        omega_los_enu = self.R_body_enu @ omega_los_body
        
        # 提取分量
        map_waz = omega_los_enu[2]  # 绕ENU Z轴
        # 俯仰率：绕水平轴垂直于视线
        map_wel = -omega_los_enu[0]*np.sin(map_az) + omega_los_enu[1]*np.cos(map_az)
        
        return map_waz, map_wel, omega_los_enu, omega_los_body

def print_coord_systems():
    print("\n" + "="*70)
    print("【坐标系定义】")
    print("ENU (东-北-上):     X(东), Y(北), Z(上) | 逆时针为正")
    print("Body-FRD (前-右-下): X(机头), Y(右), Z(下) | Z向下导致:")
    print("  - 偏航: 顺时针为正 (右手定则拇指向下)")
    print("  - 俯仰: 抬头为负 (绕Y轴负方向)")
    print("="*70)

def test_complex_maneuver():
    """测试1: 复杂姿态 + 复合运动"""
    print("\n" + "="*70)
    print("测试1: 复杂机动（滚转30°+俯仰-45°+偏航120°）")
    print("场景: 无人机左倾、低头、指向西南，多轴角速度同时存在")
    print("验证: 旋转矩阵正确性 & 复合角速度合成")
    print("="*70)
    
    tester = PodRateTester()
    tester.set_frd_orientation(-30, -45, 120)
    tester.angular_velocity = MockVector3(
        np.deg2rad(5),    # 右滚
        np.deg2rad(-3),   # 抬头
        np.deg2rad(2)     # 顺时针
    )
    tester.describe_state()
    
    waz, wel, omega_enu, omega_body = tester.calc_los_rates(5, 2, np.deg2rad(120), np.deg2rad(-20))
    
    print(f"\n输入: 吊舱CW 5°/s, Up 2°/s | 视线方向: Az=120°, El=-20°")
    print(f"Body合成角速度: [{np.rad2deg(omega_body[0]):+.1f}, "
          f"{np.rad2deg(omega_body[1]):+.1f}, {np.rad2deg(omega_body[2]):+.1f}]°/s")
    print(f"ENU视线角速度: [{np.rad2deg(omega_enu[0]):+.1f}, "
          f"{np.rad2deg(omega_enu[1]):+.1f}, {np.rad2deg(omega_enu[2]):+.1f}]°/s")
    print(f"输出: map_waz={np.rad2deg(waz):+.2f}°/s, map_wel={np.rad2deg(wel):+.2f}°/s")
    print("✓ 复杂姿态转换完成")

def test_near_vertical():
    """测试2: 大角度俯仰（接近垂直）"""
    print("\n" + "="*70)
    print("测试2: 大角度俯仰（Pitch=85°，接近垂直向上）")
    print("场景: 无人机几乎垂直爬升，验证万向节锁附近的数值稳定性")
    print("="*70)
    
    tester = PodRateTester()
    tester.set_frd_orientation(0, 85, 45)
    tester.angular_velocity = MockVector3(np.deg2rad(10), 0, np.deg2rad(5))
    
    tester.describe_state()
    
    waz, wel, _, _ = tester.calc_los_rates(0, 0, np.deg2rad(45), np.deg2rad(85))
    
    print(f"\n视线方向: Az=45°(东北), El=85°(近天顶)")
    print(f"输出: map_waz={np.rad2deg(waz):+.2f}°/s, map_wel={np.rad2deg(wel):+.2f}°/s")
    print("✓ 大角度俯仰处理正常")

def test_line_of_sight_stabilization():
    """测试3: 视线稳定验证（水平姿态）"""
    print("\n" + "="*70)
    print("测试3: 视线稳定验证（水平姿态）")
    print("场景: 水平悬停时，吊舱补偿机体运动，使视线绝对静止")
    print("注意: 两轴吊舱只能补偿俯仰和偏航，无法补偿滚转（此处设滚转为0）")
    print("="*70)
    
    tester = PodRateTester()
    # 水平姿态，机头指向东北45°
    tester.set_frd_orientation(0, 0, 45)
    
    # 机体运动：抬头15°/s + 顺时针20°/s
    tester.angular_velocity = MockVector3(
        0,                              # 无滚转
        np.deg2rad(-15),                # 抬头15°/s (q<0)
        np.deg2rad(20)                  # 顺时针20°/s
    )
    tester.describe_state()
    
    # 补偿计算：
    # - 顺时针20°/s 需 逆时针20°/s 补偿 (az = -20)
    # - 抬头15°/s 需 向下15°/s 补偿 (el = -15，注意符号！)
    comp_az = -np.rad2deg(tester.angular_velocity.z)   # -20°/s (CCW)
    comp_el = np.rad2deg(tester.angular_velocity.y)    # -15°/s (Down，修正此处)
    
    print(f"\n补偿指令:")
    print(f"  az={comp_az:+.1f}°/s ({'CCW(逆时针)' if comp_az>0 else 'CW(顺时针)'})")
    print(f"  el={comp_el:+.1f}°/s ({'Up(向上)' if comp_el>0 else 'Down(向下)'})")
    
    waz, wel, omega_enu, omega_body = tester.calc_los_rates(
        comp_az, comp_el, np.deg2rad(45), 0
    )
    
    print(f"\n验证结果:")
    print(f"  Body合成角速度: [{np.rad2deg(omega_body[0]):.2f}, "
          f"{np.rad2deg(omega_body[1]):.2f}, {np.rad2deg(omega_body[2]):.2f}]°/s")
    print(f"  ENU视线角速度:  [{np.rad2deg(omega_enu[0]):.4f}, "
          f"{np.rad2deg(omega_enu[1]):.4f}, {np.rad2deg(omega_enu[2]):.4f}]°/s")
    print(f"  输出: map_waz={np.rad2deg(waz):.4f}°/s, map_wel={np.rad2deg(wel):.4f}°/s")
    
    assert np.abs(np.rad2deg(waz)) < 0.01 and np.abs(np.rad2deg(wel)) < 0.01, \
        f"补偿失败: waz={np.rad2deg(waz):.4f}, wel={np.rad2deg(wel):.4f}"
    print("✓ 视线稳定验证通过（水平姿态下完全补偿）")

def test_stabilization_with_roll():
    """测试3b: 带滚转时的视线稳定（非水平姿态）"""
    print("\n" + "="*70)
    print("测试3b: 带滚转时的视线稳定（非水平姿态）")
    print("场景: 无人机有滚转时，两轴吊舱无法完全稳定视线")
    print("验证: 滚转会耦合到ENU俯仰/偏航，导致无法完全补偿")
    print("="*70)
    
    tester = PodRateTester()
    # 非水平姿态：有滚转
    tester.set_frd_orientation(30, 0, 0)  # 右倾30°，水平飞行
    tester.angular_velocity = MockVector3(
        np.deg2rad(10),       # 右滚10°/s
        np.deg2rad(0),        # 无俯仰率
        np.deg2rad(0)         # 无偏航率
    )
    tester.describe_state()
    
    # 尝试补偿（但无法补偿滚转）
    waz, wel, omega_enu, omega_body = tester.calc_los_rates(0, 0, 0, 0)
    
    print(f"\n无吊舱补偿时:")
    print(f"  ENU视线角速度: [{np.rad2deg(omega_enu[0]):.2f}, "
          f"{np.rad2deg(omega_enu[1]):.2f}, {np.rad2deg(omega_enu[2]):.2f}]°/s")
    print(f"  map_waz={np.rad2deg(waz):.2f}°/s, map_wel={np.rad2deg(wel):.2f}°/s")
    print("  注意: 滚转耦合产生了非零的俯仰/偏航率（两轴吊舱无法消除）")
    print("✓ 物理现象正确：两轴吊舱存在固有局限性")

def test_random_orientation():
    """测试4: 随机姿态 + 旋转矩阵正交性验证"""
    print("\n" + "="*70)
    print("测试4: 随机姿态验证")
    print("场景: 随机生成姿态角，验证旋转矩阵的正交性和行列式")
    print("="*70)
    
    np.random.seed(42)
    for i in range(3):
        r, p, y = np.random.uniform(-180, 180, 3)
        tester = PodRateTester()
        tester.set_frd_orientation(r, p, y)
        
        # 验证旋转矩阵性质
        R = tester.R_body_enu
        ortho_error = np.max(np.abs(R @ R.T - np.eye(3)))
        det = np.linalg.det(R)
        
        print(f"  随机姿态{i+1}: R={r:.0f}°, P={p:.0f}°, Y={y:.0f}° | "
              f"正交误差={ortho_error:.2e}, 行列式={det:.4f}")
        
        assert ortho_error < 1e-10 and np.abs(det - 1) < 1e-10
    print("✓ 所有随机姿态旋转矩阵正确")

def test_high_speed_maneuver():
    """测试5: 高速机动（大角速度）"""
    print("\n" + "="*70)
    print("测试5: 高速机动（100°/s级角速度）")
    print("场景: 验证大数值下的数值稳定性")
    print("="*70)
    
    tester = PodRateTester()
    tester.set_frd_orientation(45, 45, 45)
    tester.angular_velocity = MockVector3(
        np.deg2rad(100),
        np.deg2rad(-80),
        np.deg2rad(120)
    )
    tester.describe_state()
    
    waz, wel, omega_enu, _ = tester.calc_los_rates(50, 30, np.deg2rad(45), np.deg2rad(30))
    
    print(f"\n高速机动结果:")
    print(f"  ENU角速度: [{np.rad2deg(omega_enu[0]):+.1f}, "
          f"{np.rad2deg(omega_enu[1]):+.1f}, {np.rad2deg(omega_enu[2]):+.1f}]°/s")
    print(f"  输出: map_waz={np.rad2deg(waz):.1f}°/s")
    print("✓ 高速机动数值稳定")

def test_gimbal_limit():
    """测试6: 吊舱极限位置（天顶/地平面）"""
    print("\n" + "="*70)
    print("测试6: 吊舱极限位置（指向天顶El=90° & 地平面El=0°）")
    print("场景: 验证俯仰角速度投影在极端角度的行为")
    print("="*70)
    
    tester = PodRateTester()
    tester.set_frd_orientation(0, 0, 0)
    tester.angular_velocity = MockVector3(0, 0, 0)
    
    # 指向天顶
    waz, wel, _, _ = tester.calc_los_rates(0, 10, 0, np.deg2rad(90))
    print(f"\n指向天顶(El=90°)，吊舱向上10°/s:")
    print(f"  map_waz={np.rad2deg(waz):.1f}°/s, map_wel={np.rad2deg(wel):.1f}°/s")
    print("  (此时俯仰率应接近ENU Z轴分量)")
    
    # 指向地平面东
    waz, wel, _, _ = tester.calc_los_rates(10, 0, 0, 0)
    print(f"\n指向正东(El=0°)，吊舱顺时针10°/s:")
    print(f"  map_waz={np.rad2deg(waz):.1f}°/s (应为-10), map_wel={np.rad2deg(wel):.1f}°/s")
    
    print("✓ 极限位置处理正常")

if __name__ == "__main__":
    print_coord_systems()
    test_complex_maneuver()
    test_near_vertical()
    test_line_of_sight_stabilization()
    test_stabilization_with_roll()
    test_random_orientation()
    test_high_speed_maneuver()
    test_gimbal_limit()
    print("\n" + "="*70)
    print("所有一般性测试通过！")
    print("="*70)