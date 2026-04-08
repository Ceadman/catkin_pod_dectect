"""
File Name: missile_nav_test.py
Author: Adam Frewin
Date Last Modified: April 5th 2019

This code runs a simulation of a missile strike event, using proportional navigation law for missile guidance.

Option is available to utilize an extended Kalman filter, which will introduce random Gaussian Noise to the state
measurements of the target and missile.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as ani
from math import sin, cos, atan, asin

def target_accel(target_state):
    """
    Computes state derivative of the target, currently no acceleration
    :param target_state: the current state of the target, Px, Py, Pz, V, psi, gamma
    :return: the state derivative
    """

    # Target velocity and heading
    VT = target_state[3]
    psi_t = target_state[4]
    gma_t = target_state[5]

    Rie_t = np.array([[cos(gma_t), 0, -sin(gma_t)],
                      [0, 1, 0],
                      [sin(gma_t), 0, cos(gma_t)]]).dot(
        np.array([[cos(psi_t), sin(psi_t), 0],
                  [-sin(psi_t), cos(psi_t), 0],
                  [0, 0, 1]]))

    VT_inert = Rie_t.T.dot(np.array([VT, 0, 0]).T)

    return np.array([VT_inert[0], VT_inert[1], VT_inert[2], 0, 0, 0]) # for constant pitch and heading


def missile_accel(missile_state, commanded_accel):
    """
    Computes state derivative of the missile given a commanded acceleration
    :param missile_state: the current state of the missile, Px, Py, Pz, V, psi, gamma
    :param commanded_accel: inertial coordinates of commanded acceleration
    :return: actual derivative of the missile state
    """
    Vm = missile_state[3]
    psi_m = missile_state[4]
    gma_m = missile_state[5]

    Rie_m = np.array([[cos(gma_m), 0, -sin(gma_m)],
                      [0, 1, 0],
                      [sin(gma_m), 0, cos(gma_m)]]).dot(
            np.array([[cos(psi_m), sin(psi_m), 0],
                      [-sin(psi_m), cos(psi_m), 0],
                      [0, 0, 1]]))

    Vm_inert = Rie_m.T.dot(np.array([Vm, 0, 0]).T)

    ac_body_fixed = Rie_m.dot(commanded_accel.T)

    ax = ac_body_fixed[0]
    ay = ac_body_fixed[1]
    az = ac_body_fixed[2]

    return np.array([Vm_inert[0], Vm_inert[1], Vm_inert[2],
                     ax, ay/(Vm*cos(gma_m)), -az/Vm])


def get_seeker_state(target_state, missile_state):
    """
    Computes the "seeker state," relative position and velocity in inertial coordinates
    :param target_state:
    :param missile_state:
    :return:
    """
    # define variables for readability
    # T = target
    # M = missile

    # Target position
    RTx = target_state[0]
    RTy = target_state[1]
    RTz = target_state[2]

    # Target velocity and heading
    VT = target_state[3]
    psi_t = target_state[4]
    gma_t = target_state[5]

    # Missile Position
    RMx = missile_state[0]
    RMy = missile_state[1]
    RMz = missile_state[2]

    # Missile velocity and heading
    VM = missile_state[3]
    psi_m = missile_state[4]
    gma_m = missile_state[5]

    # Rotation matrices from inertial to body fixed coordinates for
    # both the target and the missile
    Rie_t = np.array([[cos(gma_t), 0, -sin(gma_t)],
                      [0, 1, 0],
                      [sin(gma_t), 0, cos(gma_t)]]).dot(
        np.array([[cos(psi_t), sin(psi_t), 0],
                  [-sin(psi_t), cos(psi_t), 0],
                  [0, 0, 1]]))

    Rie_m = np.array([[cos(gma_m), 0, -sin(gma_m)],
                      [0, 1, 0],
                      [sin(gma_m), 0, cos(gma_m)]]).dot(
        np.array([[cos(psi_m), sin(psi_m), 0],
                  [-sin(psi_m), cos(psi_m), 0],
                  [0, 0, 1]]))


    # get relative velocity in inertial coordinates
    VT_inert = Rie_t.T.dot(np.array([VT, 0, 0]).T)
    VM_inert = Rie_m.T.dot(np.array([VM, 0, 0]).T)

    return np.array([RTx - RMx, RTy - RMy, RTz - RMz,
                     VT_inert[0] - VM_inert[0], VT_inert[1] - VM_inert[1], VT_inert[2] - VM_inert[2]])


def linearize_seeker(seeker_state):
    '''
    Linearizes the seeker state evolution model at a given state
    :param seeker_state: the current state of the seeker
    :return: the linearized derivative (i.e. the Jacobian)
    '''

    # state variables
    Rx = seeker_state[0]
    Ry = seeker_state[1]
    Rz = seeker_state[2]
    Vx = seeker_state[3]
    Vy = seeker_state[4]
    Vz = seeker_state[5]

    # derived variables
    R = np.sqrt(Rx**2 + Ry**2 + Rz**2) # Magnitude of LOS vector

    # LOS angles
    alpha = atan(Ry / Rx)
    beta = asin(Rz / R)

    # Closing velocity
    Vc = -(Rx * Vx + Ry * Vy + Rz * Vz) / R

    # LOS angle derivatives
    alpha_dot = (Rx * Vy - Ry * Vx) / Rx ** 2
    beta_dot = (R * Vz + Vc * Rz) / R ** 2

    # commanded acceleration magnitudes
    ach = 3 * alpha_dot * Vc
    acv = 3 * beta_dot * Vc

    # Naming convention: dR_dRx -> partial derivative of R with respect to Rx
    # First layer: derivatives of R
    dR_dRx = Rx/R
    dR_dRy = Ry/R
    dR_dRz = Rz/R

    # Second layer: derivatives of alpha, beta, Vc
    dalpha_dRx = (1 / (1 + (Ry / Rx) ** 2)) * (-Ry / Rx ** 2)
    dalpha_dRy = (1 / (1 + (Ry / Rx) ** 2)) * (1 / Rx)

    dbeta_dRx = (1 / np.sqrt(1 - (Rz / R) ** 2)) * (-Rz * dR_dRx / R ** 2)
    dbeta_dRy = (1 / np.sqrt(1 - (Rz / R) ** 2)) * (-Rz * dR_dRy / R ** 2)
    dbeta_dRz = (1 / np.sqrt(1 - (Rz / R) ** 2)) * ((R - Rz * dR_dRz) / R ** 2)

    dVc_dRx = - ((R * Vx - (Rx * Vx + Ry * Vy + Rz * Vz) * dR_dRx) / R ** 2)
    dVc_dRy = - ((R * Vy - (Rx * Vx + Ry * Vy + Rz * Vz) * dR_dRy) / R ** 2)
    dVc_dRz = - ((R * Vz - (Rx * Vx + Ry * Vy + Rz * Vz) * dR_dRz) / R ** 2)
    dVc_dVx = -Rx / R
    dVc_dVy = -Ry / R
    dVc_dVz = -Rz / R

    # Third Layer: derivatives of alphadot, betadot
    dalphadot_dRx = (Rx * Vy - 2 * (Rx * Vy - Ry * Vx)) / Rx ** 3
    dalphadot_dRy = -Vx / Rx ** 2
    dalphadot_dVx = -Ry / Rx ** 2
    dalphadot_dVy = 1 / Rx

    dbetadot_dRx = (R * (Vz * dR_dRx + Rz * dVc_dRx) - 2 * dR_dRx * (R * Vz + Vc * Rz)) / R ** 3
    dbetadot_dRy = (R * (Vz * dR_dRy + Rz * dVc_dRy) - 2 * dR_dRy * (R * Vz + Vc * Rz)) / R ** 3
    dbetadot_dRz = (R * (Vz * dR_dRz + Vc + Rz * dVc_dRz) - 2 * dR_dRz * (R * Vz + Vc * Rz)) / R ** 3
    dbetadot_dVx = Rz / R ** 2 * dVc_dVx
    dbetadot_dVy = Rz / R ** 2 * dVc_dVy
    dbetadot_dVz = 1 / R ** 2 * (R + Rz * dVc_dVz)

    # 4th Layer: derivatives of ach, acv

    dach_dRx = 3 * (Vc * dalphadot_dRx + alpha_dot * dVc_dRx)
    dach_dRy = 3 * (Vc * dalphadot_dRy + alpha_dot * dVc_dRy)
    dach_dRz = 3 * (alpha_dot * dVc_dRz) # dalphadot_dRz = 0
    dach_dVx = 3 * (Vc * dalphadot_dVx + alpha_dot * dVc_dVx)
    dach_dVy = 3 * (Vc * dalphadot_dVy + alpha_dot * dVc_dVy)
    dach_dVz = 3 * (alpha_dot * dVc_dVz) # dalphadot_dVz = 0

    dacv_dRx = 3 * (Vc * dbetadot_dRx + beta_dot * dVc_dRx)
    dacv_dRy = 3 * (Vc * dbetadot_dRy + beta_dot * dVc_dRy)
    dacv_dRz = 3 * (Vc * dbetadot_dRz + beta_dot * dVc_dRz)

    dacv_dVx = 3 * (Vc * dbetadot_dVx + beta_dot * dVc_dVx)
    dacv_dVy = 3 * (Vc * dbetadot_dVy + beta_dot * dVc_dVy)
    dacv_dVz = 3 * (Vc * dbetadot_dVz + beta_dot * dVc_dVz)

    # Final Layer, ax, ay, az
    dax_dRx = sin(alpha) * -dach_dRx - ach * cos(alpha) * dalpha_dRx + \
              sin(beta) * sin(alpha) * dacv_dRx + \
              acv * (sin(beta) * cos(alpha) * dalpha_dRx + sin(alpha) * cos(beta) * dbeta_dRx)

    dax_dRy = sin(alpha) * -dach_dRy - ach * cos(alpha) * dalpha_dRy + \
              sin(beta) * sin(alpha) * dacv_dRy + \
              acv * (sin(beta) * cos(alpha) * dalpha_dRy + sin(alpha) * cos(beta) * dbeta_dRy)

    dax_dRz = sin(alpha) * -dach_dRz + sin(beta) * sin(alpha) * dacv_dRz + \
              acv * (sin(alpha) * cos(beta) * dbeta_dRz)

    dax_dVx = sin(alpha) * -dach_dVx + sin(beta) * sin(alpha) * dacv_dVx
    dax_dVy = sin(alpha) * -dach_dVy + sin(beta) * sin(alpha) * dacv_dVy
    dax_dVz = sin(alpha) * -dach_dVz + sin(beta) * sin(alpha) * dacv_dVz
    #
    day_dRx = cos(alpha) * dach_dRx - ach * sin(alpha) * dalpha_dRx - \
              (sin(beta) * cos(alpha) * dacv_dRx +
              acv * (-sin(beta) * sin(alpha) * dalpha_dRx + cos(alpha) * cos(beta) * dbeta_dRx)) # CHECK

    day_dRy = cos(alpha) * dach_dRy - ach * sin(alpha) * dalpha_dRy - \
              (sin(beta) * cos(alpha) * dacv_dRy +
               acv * (-sin(beta) * sin(alpha) * dalpha_dRy + cos(alpha) * cos(beta) * dbeta_dRy))

    day_dRz = cos(alpha) * dach_dRz - (sin(beta) * cos(alpha) * dacv_dRz +
               acv * (cos(alpha) * cos(beta) * dbeta_dRz))

    day_dVx = cos(alpha) * dach_dVx - (sin(beta) * cos(alpha) * dacv_dVx)
    day_dVy = cos(alpha) * dach_dVy - (sin(beta) * cos(alpha) * dacv_dVy)
    day_dVz = cos(alpha) * dach_dVz - (sin(beta) * cos(alpha) * dacv_dVz)
    #
    daz_dRx = cos(beta) * dacv_dRx - acv * sin(beta) * dbeta_dRx
    daz_dRy = cos(beta) * dacv_dRy - acv * sin(beta) * dbeta_dRy
    daz_dRz = cos(beta) * dacv_dRz - acv * sin(beta) * dbeta_dRz
    daz_dVx = cos(beta) * dacv_dVx
    daz_dVy = cos(beta) * dacv_dVy
    daz_dVz = cos(beta) * dacv_dVz

    # Now set up the linear matrix, A_lin * X ~ F(X) | X_0
    A123 = np.array([[0, 0, 0, 1, 0, 0],[0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 1]])

    A456 = -1 * np.array([[dax_dRx, dax_dRy, dax_dRz, dax_dVx, dax_dVy, dax_dVz],
                          [day_dRx, day_dRy, day_dRz, day_dVx, day_dVy, day_dVz],
                          [daz_dRx, daz_dRy, daz_dRz, daz_dVx, daz_dVy, daz_dVz]])

    return np.vstack((A123, A456))


def seeker_deriv(seeker_state):
    """
    Non-linear state derivative equations
    :param seeker_state: current seeker state
    :return: state derivative
    """

    ac_inert = pn_guidance(seeker_state)

    target_accel = np.array([0,0,0]) # update to function call for different target maneuvers

    return np.array([seeker_state[3], seeker_state[4], seeker_state[5],
                     target_accel[0] - ac_inert[0], target_accel[1]-ac_inert[1], target_accel[2]-ac_inert[2]])


def pn_guidance(seeker_state):
    """
    Provides commanded accelerations using proportional navigation law
    :param seeker_state: current seeker state
    :return: Commanded accelerations in inertial coordinates
    """

    # Rx = seeker_state[0]
    # Ry = seeker_state[1]
    # Rz = seeker_state[2]
    # Vx = seeker_state[3]
    # Vy = seeker_state[4]
    # Vz = seeker_state[5]

    # R = np.linalg.norm([Rx, Ry, Rz])

    # # alpha = angle from x axis to xy projection of R
    # # beta = angle from R down to xy plane
    # alpha = atan(Ry / Rx)
    # beta = asin(Rz / R)

    # # closing velocity, negative of the rate of change of LOS
    # Vc = -(Rx * Vx + Ry * Vy + Rz * Vz) / R

    # # derivatives of angles, neglecting trig terms
    # alpha_dot = (Rx * Vy - Ry * Vx) / (Rx ** 2)
    # beta_dot = (R * Vz + Vc * Rz) / (R ** 2)

    # # Proportional navigation law for horizontal and vertical accelerations
    # ach_mag = 3 * Vc * alpha_dot
    # acv_mag = 3 * Vc * beta_dot

    # # resolve accelerations into xyz components
    # ach = np.array([-ach_mag * sin(alpha), ach_mag * cos(alpha), 0])
    # acv = np.array([acv_mag * sin(beta) * sin(alpha), -acv_mag * sin(beta) * cos(alpha), acv_mag * cos(beta)])

    N=3, 
    r_dead=2.0,
    Vc_max=800.0
    R = np.asarray(seeker_state[:3], dtype=float)
    V = np.asarray(seeker_state[3:], dtype=float)
    r_mag = np.linalg.norm(R)

    # 1) 极近区：纯追击，直接指向目标，加速度=0
    if r_mag < r_dead:
        return np.zeros(3)

    LOS_unit = R / r_mag
    Vc = -np.dot(R, V) / r_mag

    # 2) Vc 符号翻转或过大 → 限幅
    Vc = np.clip(Vc, -Vc_max, Vc_max)

    Omega = np.cross(R, V) / (r_mag * r_mag)

    # 3) 距离加权：r_mag 越小，N 越小 → 增益退避
    N_eff = N * np.tanh(r_mag / (3 * r_dead))  # 平滑降到 0

    a_cmd = N_eff * Vc * np.cross(Omega, LOS_unit)
    return a_cmd


def seeker_EKF(target_meas, missile_meas, last_seeker_est, P, R, dt):
    """
    Runs Extended Kalman Filter on state measurements
    :param target_meas: current measurement of target location, velocity, and heading
    :param missile_meas: current measurement of target location, velocity, and heading
    :param last_seeker_est: last estimate of the seeker state
    :param P: last estimate of covariance matrix
    :param R: Measurement error covariance
    :param dt: time-step
    :return: commanded acceleration, filtered seeker state, and updated covariance estimate
    """

    # > Begin EKF
    seeker_meas = get_seeker_state(target_meas, missile_meas)
    # Input noisy target and missile state to get noisy seeker state

    # Predictions
    A_lin = linearize_seeker(last_seeker_est) # update with function call to linearize at state
    seeker_est = last_seeker_est + seeker_deriv(last_seeker_est) * dt # Estimation on how the state evolved from last time-step
    P_est = A_lin.dot(P).dot(A_lin.T)

    # Innovation
    y_tild = seeker_meas-seeker_est
    S = P_est + R
    
    # Kalman gain
    K = P_est.dot(np.linalg.pinv(S))

    # Correction
    seeker_corrected = seeker_est + K.dot(y_tild) # > this is the "filtered" seeker state
    P_update = (np.eye(6) - K).dot(P_est)

    # now compute commanded accelerations using "filtered" seeker
    command_accel = pn_guidance(seeker_corrected)

    return command_accel, seeker_corrected, P_update

# ====== MEMORY GUIDANCE MOD START ======
def body_to_world_matrix(psi, gma, phi):
    """构建 Z-Y-X 旋转顺序的弹体->世界旋转矩阵"""
    cpsi, cgm, cph = np.cos(psi), np.cos(gma), np.cos(phi)
    spsi, sgm, sph = np.sin(psi), np.sin(gma), np.sin(phi)
    
    R = np.array([
        [cpsi*cgm, cpsi*sgm*sph - spsi*cph, cpsi*sgm*cph + spsi*sph],
        [spsi*cgm, spsi*sgm*sph + cpsi*cph, spsi*sgm*cph - cpsi*sph],
        [   -sgm,               cgm*sph,               cgm*cph]
    ])
    return R

def get_commanded_accel(t_memory, seeker_now, distance, eps):
    """
    统一封装：丢目标判别 + 记忆制导 / 正常 PN 选择
    seeker_now : 当前量测（EKF 分支用 new_seeker_est，其余分支用 seeker_state）
    返回 -> commanded_accel, 是否强制 break
    """
    global memory_active, memory_start_t, last_valid_seeker, last_vm
    # ---- 1. 丢目标判别 ----
    target_lost = False
    if MEMORY_GUIDANCE:
        # 简易判据：视线角速度过大 或 距离突增
        if np.linalg.norm(seeker_now[3:5]) > 50 :
            print(np.linalg.norm(seeker_now[3:5]))
            target_lost = True
        elif (last_valid_seeker is not None and distance > 3 * np.linalg.norm(last_valid_seeker[:3])):
            print("far")
            target_lost = True

    # ---- 2. 首次丢失 → 启动记忆制导 ----
    if MEMORY_GUIDANCE and target_lost and (not memory_active):
        memory_active   = True
        memory_start_t  = t_memory
        last_valid_seeker = seeker_now.copy()
        last_vm = np.linalg.norm(missile_state[3:6]) 
        print(f"[{t_memory:.2f}s] 目标丢失 → 进入记忆制导")

    # ---- 3. 制导律选择 ----
    if MEMORY_GUIDANCE and memory_active and target_lost:
        if t_memory - memory_start_t < MEMORY_TIMEOUT:
            # 取出丢失瞬间的相对视线矢量 (unit LOS)
            los_last = last_valid_seeker[:3]
            los_last = los_last / (np.linalg.norm(los_last) + 1e-8)

            los_rate = last_valid_seeker[3:5] 
            qdot_pitch = np.clip(los_rate[0], -MAX_QDOT_PITCH, MAX_QDOT_PITCH)
            qdot_yaw   = np.clip(los_rate[1], -MAX_QDOT_YAW,   MAX_QDOT_YAW  )

            a_pitch =  qdot_pitch * last_vm                 # 俯仰平面
            a_yaw   =  qdot_yaw   * last_vm                 # 偏航平面

            # 无人机坐标
            accel_body = np.array([0.0, a_yaw, a_pitch])

            psi = missile_state[4]          # 偏航角
            gma = missile_state[5]          # 俯仰角
            phi = 0.0
            R = body_to_world_matrix(psi, gma, phi)
            commanded_accel = R @ accel_body

            print(f"[{t_memory:.2f}s] 记忆制导-最大角速度转向 "
                  f"pitch-rate={qdot_pitch*180/np.pi:5.1f}°/s  "
                  f"yaw-rate={qdot_yaw*180/np.pi:5.1f}°/s")
            return commanded_accel, False
        else:
            # 超时
            memory_active   = False
            print(f"[{t_memory:.2f}s] 记忆制导超时，放弃捕获")
            return np.zeros(3), True

    # ---- 4. 正常 PN ----
    return pn_guidance(seeker_now), False
# ====== MEMORY GUIDANCE MOD END ======

## Script
# ====== MEMORY GUIDANCE MOD START ======
MEMORY_GUIDANCE = True          # 是否启用记忆制导
MEMORY_TIMEOUT  = 10          # 记忆制导最长持续时间（s）

MAX_QDOT_PITCH =  30.0 * np.pi / 180.0   # 俯仰方向最大角速度，rad/s
MAX_QDOT_YAW   =  25.0 * np.pi / 180.0   # 偏航方向最大角速度，rad/s
# ====== MEMORY GUIDANCE MOD END ======

# ====== MEMORY GUIDANCE MOD START ======
global memory_active
global memory_start_t
global last_valid_seeker
memory_active = False           # 是否处于记忆制导状态
memory_start_t = 0.0            # 进入记忆制导的时刻
last_valid_seeker = None        # 丢失瞬间的相对状态（用于记忆）
# ====== MEMORY GUIDANCE MOD END ======

# Target will start in one position
RT_0 = np.array([100, 150, 150])

VT_0 = 50
psi_t_0 = -np.pi*1/2
gma_t_0 = -.1
target_state_0 = np.array([*RT_0, VT_0, psi_t_0, gma_t_0])

# missile starts at the same spot with the same velocity
RM_0 = np.array([0, 0, 200])
VM_0 = 8
psi_m_0 = 0
gma_m_0 = 0
missile_state_0 = np.array([*RM_0, VM_0, psi_m_0, gma_m_0])


# set up loop to stop when contact is made,
# or when t gets too high because we missed the target
t = 0
dt = 0.01
missile_state = missile_state_0
target_state = target_state_0
t_sim = [t]
missile_sim = np.array(missile_state)  # ACTUAL MISSILE STATE
target_sim = np.array(target_state)  # ACTUAL TARGET STATE
distance = np.linalg.norm(missile_state[:3]-target_state[:3])
eps = 5. # impact distance parameter

USE_EKF = False
ADD_NOISE = True

if(ADD_NOISE):
    noise_arr = 1 * np.array([.1, .1, .1, .1, 0.01, 0.01])
    if USE_EKF:
        # Initial conditions for EKF
        R = 2 * noise_arr * np.eye(6) # Measurement covariance matrix
        P = np.ones((6,6)) # Estimation of overal covariance, VARIABLE TO BE UPDATED
        missile_meas = missile_state + np.random.randn(6) * noise_arr# adding noise to actual state
        target_meas = target_state + np.random.randn(6) * noise_arr # adding noise to actual state
        seeker_est = get_seeker_state(target_meas, missile_meas) # Initial noisy seeker est
        # Need to update the simulation by one time-step, consider no accelerations
        missile_state = missile_state + missile_accel(missile_state, np.array([0,0,0])) * dt
        target_state = target_state + target_accel(target_state) * dt
        t = t+dt
        t_sim.append(t)
        missile_sim = np.vstack((missile_sim, missile_state))
        target_sim = np.vstack((target_sim, target_state))

        while (distance > eps and distance < 5000.):
            missile_meas = missile_state + np.random.randn(6) * noise_arr  # adding noise to actual state
            target_meas = target_state + np.random.randn(6) * noise_arr  # adding noise to actual state

            # EKF
            commanded_accel_0, new_seeker_est, P_n = seeker_EKF(target_meas, missile_meas, seeker_est, P, R, dt)
            
            # ====== MEMORY GUIDANCE MOD START ======
            commanded_accel, force_break = get_commanded_accel(t, new_seeker_est, distance, eps)
            # print(commanded_accel)

            # ====== MEMORY GUIDANCE MOD END ======

            # Update states
            missile_state = missile_state + missile_accel(missile_state, commanded_accel) * dt
            target_state = target_state + target_accel(target_state) * dt

            # Update other params
            distance = np.linalg.norm(missile_state[:3] - target_state[:3])
            t = t + dt
            seeker_est = new_seeker_est
            P = P_n

            # Update Simulation
            t_sim.append(t)
            missile_sim = np.vstack((missile_sim, missile_state))
            target_sim = np.vstack((target_sim, target_state))
        if distance > eps and distance < 500.:
            print("miss")
    else:
        while (distance > eps and distance < 300.):
            seeker_state = get_seeker_state(target_state, missile_state) + 2 *np.random.randn(6) * noise_arr

            # commanded_accel = pn_guidance(seeker_state)
            # ====== MEMORY GUIDANCE MOD START ======
            commanded_accel, force_break = get_commanded_accel(t, seeker_state, distance, eps)

            # ====== MEMORY GUIDANCE MOD END ======

            missile_state = missile_state + missile_accel(missile_state, commanded_accel) * dt
            target_state = target_state + target_accel(target_state) * dt
            distance = np.linalg.norm(missile_state[:3] - target_state[:3])
            t = t + dt
            t_sim.append(t)
            missile_sim = np.vstack((missile_sim, missile_state))
            target_sim = np.vstack((target_sim, target_state))
        if distance > eps and distance < 300.:
            print("miss")

    # Now the seeker_est is from the previous time-step, so we can begin the algorithm

else:
    while (distance > eps and distance < 500.):
        seeker_state = get_seeker_state(target_state, missile_state)

        # commanded_accel = pn_guidance(seeker_state)
        # ====== MEMORY GUIDANCE MOD START ======
        commanded_accel, force_break = get_commanded_accel(t, seeker_state, distance, eps)
        if force_break:
            break
        # ====== MEMORY GUIDANCE MOD END ======

        missile_state = missile_state + missile_accel(missile_state, commanded_accel) * dt
        target_state = target_state + target_accel(target_state) * dt
        distance = np.linalg.norm(missile_state[:3] - target_state[:3])
        t = t+dt
        t_sim.append(t)
        missile_sim = np.vstack((missile_sim, missile_state))
        target_sim = np.vstack((target_sim, target_state))
    if distance > eps and distance < 500.:
        print("miss")

# Animation update
def update_lines(num, target_data, missile_data, line_target, line_missile):
    # NOTE: there is no .set_data() for 3 dim data...
    line_target.set_data([target_data[:num, 0], target_data[:num,1]])
    line_target.set_3d_properties(target_data[:num, 2])
    line_missile.set_data([missile_data[:num, 0], missile_data[:num,1]])
    line_missile.set_3d_properties(missile_data[:num, 2])
    return line_target, line_missile

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

line_target, = ax.plot(target_sim[0:1,0],target_sim[0:1,1],target_sim[0:1,2])
line_missile, = ax.plot(missile_sim[0:1,0],missile_sim[0:1,1],missile_sim[0:1,2])


ax.set_xlim3d([-200, 300])

ax.set_ylim3d([-200, 200])

ax.set_zlim3d([100, 200])
'''
ax.set_xlim3d([min(np.min(target_sim[:,0]),np.min(missile_sim[:,0])),
               max(np.max(target_sim[:, 0]), np.max(missile_sim[:, 0]))])

ax.set_ylim3d([min(np.min(target_sim[:,1]),np.min(missile_sim[:,1])),
               max(np.max(target_sim[:, 1]), np.max(missile_sim[:, 1]))])

ax.set_zlim3d([min(np.min(target_sim[:,2]),np.min(missile_sim[:,2])),
               max(np.max(target_sim[:, 2]), np.max(missile_sim[:, 2]))])
'''
N = len(t_sim[::10]) + 1
target_path_plot = np.vstack((target_sim[::10], target_sim[-1]))
missile_path_plot = np.vstack((missile_sim[::10], missile_sim[-1]))
Writer = ani.writers['ffmpeg']
writer = Writer(fps=30, metadata=dict(artist='Me'), bitrate=1800)

anim = ani.FuncAnimation(fig, update_lines, N, fargs=(target_path_plot, missile_path_plot, line_target, line_missile),
                         interval=10, blit=False, repeat_delay=3000)
'''plt.title('Missile Guidance Path Animation')


ax.plot(target_sim[:,0], target_sim[:,1], target_sim[:,2], label='target path', linewidth=3)
ax.plot(missile_sim[:,0], missile_sim[:,1], missile_sim[:,2], label='Missile Path', linewidth=3)

ax.scatter(target_sim[0,0], target_sim[0,1], target_sim[0,2], marker='*', s=20)
ax.scatter(missile_sim[0,0], missile_sim[0,1], missile_sim[0,2], marker='*', s=20)
plt.legend()
#ax.text(values_sim[-1, 0], values_sim[-1, 1], values_sim[-1, 2], "HIT")
'''

anim.save('MissileStrikeSim.mp4', writer=writer)

plt.show()