#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SO-100机械臂控制器测试程序 - 轴角版本"""

import time
import os
import numpy as np
from loguru import logger
from control.arm_controller import Arm

# ==================== 日志打印函数 ====================

def wsy_print(*args, **kwargs):
    """带前缀的打印函数，方便日志筛选"""
    prefix = "wsy: "
    if args:
        # 处理开头的换行符
        first_arg = str(args[0])
        if first_arg.startswith('\n'):
            print()  # 先输出换行
            first_arg = first_arg[1:]  # 去掉开头的换行符
        print(prefix + first_arg, *args[1:], **kwargs)
    else:
        print(prefix, **kwargs)

# import matplotlib
# matplotlib.use('QtAgg')  # 优先使用统一的 Qt 后端
# # 如果你的 Matplotlib 版本较老，则用：
# # matplotlib.use('Qt5Agg')

def test_move_to_joints(arm):
    """测试关节运动"""
    print("=== 测试关节运动 ===")
    joints1 = [-0.78, -0.78, -0.78, -0.78, -0.78]
    joints2 = [0.0, 0.0, 0.0, 0.0, 0.0]

    print("移动到关节位置1...")
    arm.move_to_joints(joints1, timeout=10, tolerance=0.01)
    print(f"当前关节: {arm.get_joints()}")
    
    input("按回车继续...")
    print("移动到关节位置2...")
    arm.move_to_joints(joints2, timeout=10, tolerance=0.01)
    print(f"当前关节: {arm.get_joints()}")

def test_move_to_position(arm):
    """测试位置运动（保持当前姿态）"""
    print("=== 测试位置运动 ===")
    position1 = [0.2, -0.2, 0.2]
    position2 = [-0.1, -0.3, 0.3]
    
    print(f"移动到位置1: {position1}")
    arm.move_to_pose(position1, timeout=10, tolerance=0.01)
    print(f"当前位姿: {arm.get_pose()}")
    
    input("按回车继续...")
    print(f"移动到位置2: {position2}")
    arm.move_to_pose(position2, timeout=10, tolerance=0.01)
    print(f"当前位姿: {arm.get_pose()}")

def test_move_to_pose(arm):
    """测试完整位姿运动 - 轴角格式"""
    print("=== 测试完整位姿运动 (轴角格式) ===")
    # 轴角格式: [x, y, z, ax, ay, az, angle]
    pose10 = [0.2, -0.2, 0.2, 0, 0, 1, 0]  # 无旋转
    pose1rx = [0.2, -0.2, 0.2, 1, 0, 0, 0.78]  # 绕X轴旋转0.2弧度
    pose1ry = [0.2, -0.2, 0.2, 0, 1, 0, 0.78]  # 绕X轴旋转0.2弧度
    pose1rz = [0.2, -0.2, 0.2, 0, 0, 1, 0.78]  # 绕X轴旋转0.2弧度

    pose20 = [-0.1, -0.3, 0.3, 0, 0, 1, 0]  # 无旋转
    pose2rx = [-0.1, -0.3, 0.3, 1, 0, 0, -0.78]  # 绕X轴旋转0.2弧度
    pose2ry = [-0.1, -0.3, 0.3, 0, 1, 0, 0.78]  # 绕X轴旋转0.2弧度
    pose2rz = [-0.1, -0.3, 0.3, 0, 0, 1, 0.78]  # 绕X轴旋转0.2弧度
    
    print(f"移动到位姿1 (轴角): {pose10}")
    arm.move_to_pose(pose10, timeout=10, tolerance=0.01)
    print(f"当前位姿: {arm.get_pose()}")
    input("按回车, 绕X轴旋转0.78弧度...")
    arm.move_to_pose(pose1rx, timeout=10, tolerance=0.01)
    print(f"当前位姿: {arm.get_pose()}")
    input("按回车, 绕Y轴旋转0.78弧度...")
    arm.move_to_pose(pose1ry, timeout=10, tolerance=0.01)
    print(f"当前位姿: {arm.get_pose()}")
    input("按回车, 绕Z轴旋转0.78弧度...")
    arm.move_to_pose(pose1rz, timeout=10, tolerance=0.01)   
    print(f"当前位姿: {arm.get_pose()}")

    input("按回车, 移动到位姿2...")
    print(f"移动到位姿2 (轴角): {pose20}")
    arm.move_to_pose(pose20, timeout=10, tolerance=0.01)
    print(f"当前位姿: {arm.get_pose()}")
    input("按回车, 绕X轴旋转-0.78弧度...")
    arm.move_to_pose(pose2rx, timeout=10, tolerance=0.01)
    print(f"当前位姿: {arm.get_pose()}")
    input("按回车, 绕Y轴旋转0.78弧度...")
    arm.move_to_pose(pose2ry, timeout=10, tolerance=0.01)
    print(f"当前位姿: {arm.get_pose()}")
    input("按回车, 绕Z轴旋转0.78弧度...")
    arm.move_to_pose(pose2rz, timeout=10, tolerance=0.01)
    print(f"当前位姿: {arm.get_pose()}")

def test_move_line(arm):
    """测试直线运动"""
    print("=== 测试直线运动 ===")
    start_pos = [0.2, -0.2, 0.2]
    end_pos = [-0.1, -0.3, 0.3]
    
    print(f"直线运动: {start_pos} -> {end_pos}")
    arm.move_line(start_pos, end_pos, step=0.02, timeout=20, tolerance=0.01)
    print("直线运动完成")

def test_move_home(arm):
    """测试移动到home位置"""
    print("=== 测试移动到home位置 ===")
    print(f"Home关节: {arm.home_joints}")
    arm.move_home(timeout=10)
    print(f"当前关节: {arm.get_joints()}")
    print(f"当前位姿: {arm.get_pose()}")

def test_move_to_up_pose(arm):
    """测试移动到up位置"""
    print("=== 测试移动到up位置 ===")
    print(f"Up关节: {arm.up_joints}")
    arm.move_to_up_pose(timeout=10)
    print(f"当前关节: {arm.get_joints()}")
    print(f"当前位姿: {arm.get_pose()}")

def test_move_to_joint_zero_pose(arm):
    """测试移动到zero位置"""
    print("=== 测试移动到zero位置 ===")
    print(f"Zero关节: {arm.zero_joints}")
    arm.move_to_joint_zero_pose(timeout=10)
    print(f"当前关节: {arm.get_joints()}")
    print(f"当前位姿: {arm.get_pose()}")

def test_joints_precision(arm):
    """测试关节精度"""
    print("=== 测试关节精度 ===")
    target_joints = [0.0, 0.0, 0.0, 0.0, 0.0]
    
    print(f"目标关节: {target_joints}")
    arm.move_to_joints(target_joints, timeout=10, tolerance=0.004)
    current_joints = arm.get_joints()
    
    if current_joints:
        diff = [abs(target_joints[i] - current_joints[i]) for i in range(5)]
        print(f"当前关节: {current_joints}")
        print(f"误差: {diff}")
        print(f"最大误差: {max(diff):.4f}")
    else:
        print("无法获取当前关节位置")

def test_pose_precision(arm):
    """测试位置控制精度"""
    print("=== 测试位置控制精度 ===")
    
    pose1 = [0.2, -0.2, 0.2, 1, 0, 0, 0]
    pose2 = [-0.2, -0.2, 0.2, 1, 0, 0, 0]
    pose3 = [0.2, -0.2, 0.2, 1, 0, 0, 0.78]
    pose4 = [0.2, -0.2, 0.2, 0, 1, 0, 0.78]
    pose5 = [0.2, -0.2, 0.2, 0, 0, 1, 0.78]

    input("按回车移动到pose1....")
    print(f"\n移动到位姿1: {pose1}")
    arm.move_to_pose(pose1, timeout=10, tolerance=0.01)
    current_pose = arm.get_pose()
    pose_diff = np.abs(np.array(current_pose) - np.array(pose1))
    print(f"目标位姿: {pose1}")
    print(f"实际位姿: {arm.get_pose()}")
    print(f"误差: {pose_diff}")
    print(f"最大误差: {np.max(pose_diff):.4f}")

    input("按回车移动到pose2....")
    print(f"\n移动到位姿2: {pose2}")
    arm.move_to_pose(pose2, timeout=10, tolerance=0.01)
    current_pose = arm.get_pose()
    pose_diff = np.abs(np.array(current_pose) - np.array(pose2))
    print(f"目标位姿: {pose2}")
    print(f"实际位姿: {arm.get_pose()}")
    print(f"误差: {pose_diff}")
    print(f"最大误差: {np.max(pose_diff):.4f}")

    input("按回车移动到pose3....")
    print(f"\n移动到位姿3: {pose3}")
    arm.move_to_pose(pose3, timeout=10, tolerance=0.01)
    current_pose = arm.get_pose()
    pose_diff = np.abs(np.array(current_pose) - np.array(pose3))
    print(f"目标位姿: {pose3}")
    print(f"实际位姿: {arm.get_pose()}")
    print(f"误差: {pose_diff}")
    print(f"最大误差: {np.max(pose_diff):.4f}")

    input("按回车移动到pose4....")
    print(f"\n移动到位姿4: {pose4}")
    arm.move_to_pose(pose4, timeout=10, tolerance=0.01)
    current_pose = arm.get_pose()
    pose_diff = np.abs(np.array(current_pose) - np.array(pose4))
    print(f"目标位姿: {pose4}")
    print(f"实际位姿: {arm.get_pose()}")
    print(f"误差: {pose_diff}")
    print(f"最大误差: {np.max(pose_diff):.4f}")

    input("按回车移动到pose5....")
    print(f"\n移动到位姿5: {pose5}")
    arm.move_to_pose(pose5, timeout=10, tolerance=0.01)
    current_pose = arm.get_pose()
    pose_diff = np.abs(np.array(current_pose) - np.array(pose5))
    print(f"目标位姿: {pose5}")
    print(f"实际位姿: {arm.get_pose()}")
    print(f"误差: {pose_diff}")
    print(f"最大误差: {np.max(pose_diff):.4f}")

def test_joints_high_frequency_control(arm):
    """测试高频关节控制"""
    print("=== 测试高频关节控制 ===")
    print("按Ctrl+C停止")
    
    joint_range = 0.3
    center = 0.0
    step = 0.01
    direction = 1
    current_val = center
    
    try:
        while True:
            joints = [0, 0, current_val, 0, 0]
            arm.move_to_joints(joints, timeout=0)  # 无等待
            print(f"关节2位置: {current_val:.3f}")
            
            current_val += step * direction
            if current_val >= center + joint_range or current_val <= center - joint_range:
                direction *= -1
            
            time.sleep(0.05)
    except KeyboardInterrupt:
        print("\n高频控制测试结束")

def test_get_joints(arm):
    """测试获取关节角度"""
    print("=== 测试获取关节角度 ===")
    print("连续获取关节角度10次...")
    
    for i in range(10):
        joints = arm.get_joints()
        print(f"第{i+1}次: {joints}")
        time.sleep(0.1)

def test_get_pose(arm):
    """测试获取位姿 - 轴角格式"""
    print("=== 测试获取位姿 (轴角格式) ===")
    print("连续获取位姿5次...")
    
    for i in range(5):
        tcp_pose = arm.get_pose(tcp=True)
        flange_pose = arm.get_pose(tcp=False)
        print(f"第{i+1}次:")
        print(f"  TCP位姿 [x,y,z,ax,ay,az,angle]: {tcp_pose}")
        print(f"  Flange位姿 [x,y,z,ax,ay,az,angle]: {flange_pose}")
        time.sleep(0.2)

def test_tuning_rotation_reachable(arm):
    """测试旋转可达性调整 - 轴角格式"""
    wsy_print("=== 测试旋转可达性调整 (轴角格式) ===")
    # 轴角格式测试位姿: [x, y, z, ax, ay, az, angle]
    # pose = [-0.2, -0.2, 0.2, 1, 0, 0, 0.78]  # 绕Z轴旋转1.7弧度
    # pose = [-0.2, -0.2, 0.2, 0, 1, 0, 0.78]  # 绕Z轴旋转1.7弧度
    pose = [-0.2, -0.2, 0.2, 0, 0, 1, 0.78]  # 绕Z轴旋转1.7弧度

    # print(f"测试位姿 (轴角): {pose}")
    wsy_print(f"测试位姿 (轴角): {pose}")
    
    if not arm._is_rotation_reachable(pose):
        wsy_print("位姿不可达")
    else:
        wsy_print("位姿可达")

    wsy_print("\n计算逆运动学解...")
    ik_result = arm.cacul_ik_joints_within_limits_from_pose(pose)
    wsy_print(f"逆解结果: {ik_result}")

    wsy_print("\n尝试调整旋转到可达位置...")
    pose2 = arm._tuning_rotation_reachable(pose)
    wsy_print(f"调整后位姿: {pose2}")

    wsy_print("\n检查调整后位姿可达性...")    
    if not arm._is_rotation_reachable(pose2):
        wsy_print("调整后仍不可达")
    else:
        wsy_print("调整后可达")

    ik_result2 = arm.cacul_ik_joints_within_limits_from_pose(pose2)
    wsy_print(f"调整后逆解结果: {ik_result2}")

def test_absolute_direction_movement(arm):
    """测试绝对方向移动"""
    print("=== 测试绝对方向移动 ===")
    input("按回车移动到初始位姿....")
    pose_init = [0.2, -0.2, 0.2, 1, 0, 0, 0.78]
    arm.move_to_pose(pose_init)
    vx = [0.05, 0, 0]
    vy = [0, 0.05, 0]
    vz = [0, 0, 0.05]
    vx1 = [-0.05, 0, 0]
    vy1 = [0, -0.05, 0]
    vz1 = [0, 0, -0.05]

    input("按回车往x方向移动0.05m....")
    arm.move_to_direction_abs(vx)
    input("按回车往x方向移动-0.05m....")
    arm.move_to_direction_abs(vx1)

    input("按回车往y方向移动0.05m....")
    arm.move_to_direction_abs(vy)
    input("按回车往y方向移动-0.05m....")
    arm.move_to_direction_abs(vy1)

    input("按回车往z方向移动0.05m....")
    arm.move_to_direction_abs(vz)
    input("按回车往z方向移动-0.05m....")
    arm.move_to_direction_abs(vz1)

def test_relative_direction_movement(arm):
    """测试绝对方向移动"""
    print("=== 测试绝对方向移动 ===")
    input("按回车移动到初始位姿....")
    pose_init = [0.2, -0.2, 0.2, 1, 0, 0, 0.78]
    arm.move_to_pose(pose_init)
    
    # 相对方向移动
    vx = [0.05, 0, 0]
    vy = [0, 0.05, 0]
    vz = [0, 0, 0.05]
    vx1 = [-0.05, 0, 0]
    vy1 = [0, -0.05, 0]
    vz1 = [0, 0, -0.05]

    input("按回车往x方向移动0.05m....")
    arm.move_to_direction_relative(vx)
    input("按回车往x方向移动-0.05m....")
    arm.move_to_direction_relative(vx1)

    input("按回车往y方向移动0.05m....")
    arm.move_to_direction_relative(vy)
    input("按回车往y方向移动-0.05m....")
    arm.move_to_direction_relative(vy1)

    input("按回车往z方向移动0.05m....")
    arm.move_to_direction_relative(vz)
    input("按回车往z方向移动-0.05m....")
    arm.move_to_direction_relative(vz1)



def main(): 
    """测试主函数"""
    arm = None
    
    try:
        arm = Arm()
        print("机械臂初始化成功")
        current_joints = arm.get_joints()
        current_pose = arm.get_pose()
        print(f"当前关节: {current_joints}")
        print(f"当前TCP位姿 (轴角格式): {current_pose}")
        
        # 测试列表
        tests = {
            '1': ('关节运动', test_move_to_joints),
            '2': ('位置运动', test_move_to_position),
            '3': ('完整位姿运动 (轴角)', test_move_to_pose),
            '4': ('直线运动', test_move_line),
            '5': ('移动到Home', test_move_home),
            '6': ('移动到Up', test_move_to_up_pose),
            '7': ('移动到Zero', test_move_to_joint_zero_pose),
            '8': ('关节精度测试', test_joints_precision),
            '9': ('pose精度测试', test_pose_precision),
            '10': ('高频控制', test_joints_high_frequency_control),
            '11': ('获取关节', test_get_joints),
            '12': ('获取位姿 (轴角)', test_get_pose),
            '13': ('旋转可达性调整 (轴角)', test_tuning_rotation_reachable),
            '14': ('绝对方向移动测试', test_absolute_direction_movement),
            '15': ('相对方向移动测试', test_relative_direction_movement),
        }
        
        print("\n可用测试：")
        for key, (name, _) in tests.items():
            print(f"  {key}: {name}")
        
        choice = input(f"\n请选择测试 (1-{len(tests)}, 回车跳过): ").strip()
        # choice = "12"
        
        if choice in tests:
            test_name, test_func = tests[choice]
            print(f"\n开始测试: {test_name}")
            test_func(arm)
        else:
            print("跳过测试，程序正常结束")

    except KeyboardInterrupt:
        logger.info("用户中断")
    except Exception as e:
        logger.error(f"错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if arm:
            try:
                arm.destroy_node()
            except Exception as e:
                logger.error(f"销毁节点错误: {e}")

if __name__ == '__main__':
    logger.remove()
    logger.add(
        os.sys.stdout,
        format="<level>{level: <4}</level> | <cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
    )
    main()
