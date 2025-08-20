#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""轴角姿态矩阵计算工具"""

import numpy as np
from scipy.spatial.transform import Rotation

def axis_angle_multiply(pose1, pose2):
    """
    轴角姿态复合计算
    
    Args:
        pose1: [ax1, ay1, az1, angle1] 第一个轴角姿态
        pose2: [ax2, ay2, az2, angle2] 第二个轴角姿态
    
    Returns:
        result: [ax, ay, az, angle] 复合后的轴角姿态
    """
    # 1. 输入两个轴角表示的姿态
    ax1, ay1, az1, angle1 = pose1
    ax2, ay2, az2, angle2 = pose2
    
    print(f"输入姿态1: [{ax1:.3f}, {ay1:.3f}, {az1:.3f}, {angle1:.3f}]")
    print(f"输入姿态2: [{ax2:.3f}, {ay2:.3f}, {az2:.3f}, {angle2:.3f}]")
    
    # 2. 姿态转换成旋转矩阵
    rotvec1 = np.array([ax1, ay1, az1]) * angle1
    rotvec2 = np.array([ax2, ay2, az2]) * angle2
    
    rot1 = Rotation.from_rotvec(rotvec1)
    rot2 = Rotation.from_rotvec(rotvec2)
    
    matrix1 = rot1.as_matrix()
    matrix2 = rot2.as_matrix()
    
    print(f"\n旋转矩阵1:\n{matrix1}")
    print(f"\n旋转矩阵2:\n{matrix2}")
    
    # 3. 旋转矩阵相乘
    result_matrix = matrix2 @ matrix1  # R = R2 * R1
    print(f"\n相乘结果矩阵:\n{result_matrix}")
    
    # 4. 相乘结果转换成轴角
    result_rot = Rotation.from_matrix(result_matrix)
    result_rotvec = result_rot.as_rotvec()
    
    result_angle = np.linalg.norm(result_rotvec)
    if result_angle < 1e-6:
        result_axis = [0, 0, 1]  # 默认轴
        result_angle = 0
    else:
        result_axis = result_rotvec / result_angle
    
    result = list(result_axis) + [result_angle]
    print(f"\n结果轴角: [{result[0]:.3f}, {result[1]:.3f}, {result[2]:.3f}, {result[3]:.3f}]")
    
    return result

if __name__ == "__main__":
    print("轴角姿态矩阵计算")
    print("="*40)
    
    pose1 = [1, 0, 0, 0.78]  # 绕X轴45度
    pose2 = [0, 0, 1, -0.78]  # 绕Z轴90度
    result1 = axis_angle_multiply(pose1, pose2)
    
    # print("\n" + "-"*40)
    
    # # 示例2: 任意轴角组合
    # print("\n示例2: 任意轴角组合")
    # pose3 = [1, 1, 0, 0.5]      # 绕[1,1,0]轴转0.5弧度
    # pose4 = [0, 1, 1, 0.3]      # 绕[0,1,1]轴转0.3弧度
    # result2 = axis_angle_multiply(pose3, pose4)
    
    print("\n程序结束")
