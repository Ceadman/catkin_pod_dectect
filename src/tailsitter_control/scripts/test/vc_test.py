import numpy as np

class ApproachVelocityTester:
    def __init__(self):
        self.linear_velocity = None  # 模拟的无人机速度 [vx, vy, vz] in ENU
        self.pod_angle = None          # [azimuth, elevation] in radians
    
    def set_velocity(self, vx, vy, vz):
        """设置无人机在大地坐标系下的速度 (m/s)"""
        self.linear_velocity = np.array([vx, vy, vz])
        print(f"设置速度: E={vx:.2f}, N={vy:.2f}, U={vz:.2f} m/s")
        print(f"速度大小: {np.linalg.norm(self.linear_velocity):.2f} m/s")
        return self
    
    def set_pod_angle(self, azimuth_deg, elevation_deg):
        """
        设置光电吊舱指向角度（角度制输入，自动转弧度）
        azimuth: 方位角，从东(X)向北(Y)偏，逆时针为正 [度]
        elevation: 俯仰角，水平面向上为正 [度]
        """
        azimuth = np.radians(azimuth_deg)
        elevation = np.radians(elevation_deg)
        self.pod_angle = np.array([azimuth, elevation])
        
        # 打印方向信息
        dir_text = self._get_direction_text(azimuth_deg)
        print(f"设置角度: Az={azimuth_deg:.1f}° ({dir_text}), El={elevation_deg:.1f}°")
        return self
    
    def _get_direction_text(self, azimuth_deg):
        """将方位角转换为方向描述"""
        az = azimuth_deg % 360
        if 0 <= az < 22.5 or 337.5 <= az < 360:
            return "正东"
        elif 22.5 <= az < 67.5:
            return "东北"
        elif 67.5 <= az < 112.5:
            return "正北"
        elif 112.5 <= az < 157.5:
            return "西北"
        elif 157.5 <= az < 202.5:
            return "正西"
        elif 202.5 <= az < 247.5:
            return "西南"
        elif 247.5 <= az < 292.5:
            return "正南"
        elif 292.5 <= az < 337.5:
            return "东南"
        return "未知"
    
    def calculate_approach_velocity(self):
        """计算接近速度"""
        if self.linear_velocity is None or self.pod_angle is None:
            raise ValueError("请先设置速度和角度！")
        
        azimuth, elevation = self.pod_angle
        
        # 构建ENU坐标系下的单位方向向量
        cos_el = np.cos(elevation)
        direction = np.array([
            cos_el * np.cos(azimuth),   # X-East
            cos_el * np.sin(azimuth),   # Y-North
            np.sin(elevation)            # Z-Up
        ])
        
        # 归一化（防止浮点误差）
        direction = direction / np.linalg.norm(direction)
        
        # 计算接近速度（点积）
        approach_velocity = np.dot(self.linear_velocity, direction)
        
        return approach_velocity, direction
    
    def run_test(self):
        """执行测试并输出详细结果"""
        try:
            approach_vel, direction = self.calculate_approach_velocity()
            
            print("\n" + "="*50)
            print("计算结果")
            print("="*50)
            
            # 方向向量分量
            print(f"目标方向单位向量:")
            print(f"  East (X):  {direction[0]:+.4f}")
            print(f"  North(Y):  {direction[1]:+.4f}")
            print(f"  Up   (Z):  {direction[2]:+.4f}")
            
            # 速度分解
            vel_proj = np.dot(self.linear_velocity, direction) * direction  # 径向分量
            vel_perp = self.linear_velocity - vel_proj                     # 横向分量
            
            print(f"\n速度分解:")
            print(f"  径向速度 (接近速度): {approach_vel:+.4f} m/s")
            print(f"  横向速度大小:      {np.linalg.norm(vel_perp):.4f} m/s")
            
            # 判断接近还是远离
            if approach_vel > 0.01:
                status = "🟢 接近目标"
                time_to_contact = np.linalg.norm(self.linear_velocity) / approach_vel if approach_vel != 0 else float('inf')
            elif approach_vel < -0.01:
                status = "🔴 远离目标"
                time_to_contact = float('inf')
            else:
                status = "🟡 横向飞行（既不接近也不远离）"
                time_to_contact = float('inf')
            
            print(f"\n状态: {status}")
            
            # 额外信息
            angle_between = np.degrees(np.arccos(
                np.dot(self.linear_velocity, direction) / 
                (np.linalg.norm(self.linear_velocity) * np.linalg.norm(direction) + 1e-10)
            ))
            print(f"速度与目标方向夹角: {angle_between:.1f}°")
            
            return approach_vel
            
        except ValueError as e:
            print(f"错误: {e}")
            return None


def interactive_test():
    """交互式测试函数"""
    tester = ApproachVelocityTester()
    
    print("="*60)
    print("无人机接近速度测试工具")
    print("坐标系: ENU (East-North-Up)")
    print("="*60)
    
    while True:
        print("\n" + "-"*40)
        
        # 输入速度
        try:
            vx = float(input("输入速度 East 分量 vx (m/s): "))
            vy = float(input("输入速度 North 分量 vy (m/s): "))
            vz = float(input("输入速度 Up 分量 vz (m/s): "))
        except ValueError:
            print("输入错误，请输入数字！")
            continue
        
        # 输入角度
        try:
            az_deg = float(input("输入方位角 Azimuth (度，东为0，北为90): "))
            el_deg = float(input("输入俯仰角 Elevation (度，向上为正): "))
        except ValueError:
            print("输入错误，请输入数字！")
            continue
        
        # 设置并运行测试
        tester.set_velocity(vx, vy, vz).set_pod_angle(az_deg, el_deg).run_test()
        
        # 是否继续
        cont = input("\n继续测试? (y/n): ").strip().lower()
        if cont != 'y':
            break
    
    print("测试结束")


def batch_test_cases():
    """批量运行典型测试用例"""
    print("\n" + "="*60)
    print("批量测试典型场景")
    print("="*60)
    
    test_cases = [
        # (vx, vy, vz, az_deg, el_deg, 描述)
        (10, 0, 0, 0, 0, "向东飞，目标在东（正对目标）"),
        (10, 0, 0, 180, 0, "向东飞，目标在西（背对目标）"),
        (10, 0, 0, 90, 0, "向东飞，目标在北（横向）"),
        (10, 10, 0, 45, 0, "向东北飞，目标在东北（45度）"),
        (0, 10, 5, 90, 26.57, "向北偏上飞，目标在正北偏上（正对）"),
        (5, 5, 5, 45, 35.26, "空间对角线飞行"),
        (0, 0, 5, 0, 90, "垂直上升，目标在正上方"),
        (-5, 5, 0, 135, 0, "向西北飞，目标在西北"),
    ]
    
    tester = ApproachVelocityTester()
    
    for vx, vy, vz, az, el, desc in test_cases:
        print(f"\n{'='*50}")
        print(f"测试场景: {desc}")
        print(f"{'='*50}")
        tester.set_velocity(vx, vy, vz).set_pod_angle(az, el).run_test()


# 主程序入口
if __name__ == "__main__":
    import sys
    
    print("选择测试模式:")
    print("1. 交互式测试 (手动输入)")
    print("2. 批量测试 (预定义场景)")
    
    choice = input("请输入选项 (1/2): ").strip()
    
    if choice == "1":
        interactive_test()
    elif choice == "2":
        batch_test_cases()
    else:
        print("无效选项，运行批量测试...")
        batch_test_cases()