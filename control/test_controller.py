#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SO-100机械臂控制器测试程序"""

import rclpy
import time
import os
from loguru import logger
from control.arm_controller import Arm

import matplotlib
matplotlib.use('QtAgg')  # 优先使用统一的 Qt 后端
# 如果你的 Matplotlib 版本较老，则用：
# matplotlib.use('Qt5Agg')

def test_move_to_joints(arm):
    """测试关节运动"""
    print("=== 测试关节运动 ===")
    joints1 = [0.0, 0.0, 0.0, 0.0, 0.0]
    joints2 = [0.5, -0.2, 0.3, 0.1, -0.4]
    
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
    position1 = [0.2, -0.2, 0.3]
    position2 = [-0.1, -0.3, 0.2]
    
    print(f"移动到位置1: {position1}")
    arm.move_to_pose(position1, timeout=10, tolerance=0.01)
    print(f"当前位姿: {arm.get_pose()}")
    
    input("按回车继续...")
    print(f"移动到位置2: {position2}")
    arm.move_to_pose(position2, timeout=10, tolerance=0.01)
    print(f"当前位姿: {arm.get_pose()}")

def test_move_to_pose(arm):
    """测试完整位姿运动"""
    print("=== 测试完整位姿运动 ===")
    pose1 = [0.2, -0.2, 0.3, 0, 0, 0]
    pose2 = [-0.1, -0.3, 0.2, 0.2, 0.1, -0.3]
    
    print(f"移动到位姿1: {pose1}")
    arm.move_to_pose(pose1, timeout=10, tolerance=0.01)
    print(f"当前位姿: {arm.get_pose()}")
    
    input("按回车继续...")
    print(f"移动到位姿2: {pose2}")
    arm.move_to_pose(pose2, timeout=10, tolerance=0.01)
    print(f"当前位姿: {arm.get_pose()}")

def test_move_line(arm):
    """测试直线运动"""
    print("=== 测试直线运动 ===")
    start_pos = [0.2, -0.2, 0.3, 0, 0, 0]
    end_pos = [-0.1, -0.3, 0.2, 0, 0, 0]
    
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
    target_joints = [0.5, -0.2, 0.3, 0.1, -0.4]
    
    print(f"目标关节: {target_joints}")
    arm.move_to_joints(target_joints, timeout=10, tolerance=0.005)
    current_joints = arm.get_joints()
    
    diff = [abs(target_joints[i] - current_joints[i]) for i in range(5)]
    print(f"当前关节: {current_joints}")
    print(f"误差: {diff}")
    print(f"最大误差: {max(diff):.4f}")

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
    """测试获取位姿"""
    print("=== 测试获取位姿 ===")
    print("连续获取位姿5次...")
    
    for i in range(5):
        tcp_pose = arm.get_pose(tcp=True)
        flange_pose = arm.get_pose(tcp=False)
        print(f"第{i+1}次:")
        print(f"  TCP位姿: {tcp_pose}")
        print(f"  Flange位姿: {flange_pose}")
        time.sleep(0.2)

def test_tuning_rotation_reachable(arm):
    """测试旋转可达性调整"""
    print("\nwsy=== 测试旋转可达性调整 ===")
    # pose = [0.06643699856111875, -0.16562473378074943, 0.06919128028718832, 1.7640112216671655, 0.08231736788759281, 0.10629035256903263]
    # pose = [0.2, -0.2, 0.2, 0.5, 0.0, 0.0]
    # pose = [0.2, -0.2, 0.2, 0.0, 0.0, 0.784]
    # pose = [0.2, -0.001, 0.2, 0.0, 0.0, 1.0]
    # pose = [0.2, -0.001, 0.2, 1.7, 0.08, 0.1]
    pose = [0.2, -0.2, 0.2, 1.7, 0.1, 0.1]
    print(f"wsy测试 pose: {pose}")
    if not arm._is_rotation_reachable(pose):
        print(f"wsypose 不可达")

    print(f"\nwsy计算 ik 解")
    ik_result = arm.cacul_ik_joints_within_limits_from_pose(pose)
    print(f"wsyik_result: {ik_result}")

    print(f"\nwsy尝试调整旋转到可达位置")
    pose2 = arm._tuning_rotation_reachable(pose)
    print(f"wsy调整后 pose: {pose2}")

    print(f"\nwsy计算新pose ik 解")    
    if not arm._is_rotation_reachable(pose2):
        print(f"wsy调整后仍不可达")

    ik_result2 = arm.cacul_ik_joints_within_limits_from_pose(pose2)
    print(f"wsyik_result2: {ik_result2}")


def main(): 
    """测试主函数"""
    rclpy.init()
    arm = None
    
    try:
        arm = Arm()
        print("机械臂初始化成功")
        print(f"当前关节: {arm.get_joints()}")
        print(f"当前TCP位姿: {arm.get_pose()}")
        
        # 测试列表
        tests = {
            '1': ('关节运动', test_move_to_joints),
            '2': ('位置运动', test_move_to_position),
            '3': ('完整位姿运动', test_move_to_pose),
            '4': ('直线运动', test_move_line),
            '5': ('移动到Home', test_move_home),
            '6': ('移动到Up', test_move_to_up_pose),
            '7': ('移动到Zero', test_move_to_joint_zero_pose),
            '8': ('关节精度测试', test_joints_precision),
            '9': ('高频控制', test_joints_high_frequency_control),
            '10': ('获取关节', test_get_joints),
            '11': ('获取位姿', test_get_pose),
            '12': ('测试旋转可达性调整', test_tuning_rotation_reachable),
        }
        
        print("\n可用测试：")
        for key, (name, _) in tests.items():
            print(f"  {key}: {name}")
        
        # choice = input(f"\n请选择测试 (1-{len(tests)}, 回车跳过): ").strip()
        choice = '12'
        
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
        
        try:
            rclpy.shutdown()
        except Exception as e:
            logger.error(f"关闭rclpy错误: {e}")

if __name__ == '__main__':
    logger.remove()
    logger.add(
        os.sys.stdout,
        format="<level>{level: <4}</level> | <cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
    )
    main()
