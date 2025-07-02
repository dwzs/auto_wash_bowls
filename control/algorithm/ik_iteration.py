import numpy as np
from scipy.spatial.transform import Rotation

so100_joints_data = [
    {"translation": [0, 0, 0],            "orientation": [0, 0, 0], "rotate_axis": [0, 0, 1], "joint_limit": [-0, 0]}, # 基座虚拟关节
    {"translation": [0, -0.0452, 0.0165], "orientation": [1.5708, 0, 0], "rotate_axis": [0, 1, 0], "joint_limit": [-2, 2]},
    {"translation": [0, 0.1025, 0.0306],  "orientation": [0, 0, 0], "rotate_axis": [1, 0, 0], "joint_limit": [-1.8, 1.8]},
    {"translation": [0, 0.11257, 0.028],  "orientation": [0, 0, 0], "rotate_axis": [1, 0, 0], "joint_limit": [-1.8, 1.57]},
    {"translation": [0, 0.0052, 0.1349],  "orientation": [0, 0, 0], "rotate_axis": [1, 0, 0], "joint_limit": [-3.6, 0.3] },
    {"translation": [0, -0.0601, 0],      "orientation": [0, 0, 0], "rotate_axis": [0, 1, 0], "joint_limit": [-1.57, 1.57]}
    # {"translation": [0, -0.1, 0],         "orientation": [0, 0, 0], "rotate_axis": [0, 0, 1], "joint_limit": [0, 0]} # tcp 虚拟关节
]


def _transformation_matrix(translation, origin_orientation, rotation, angle):
    """Create a transformation matrix for a joint."""
    rot_matrix = Rotation.from_rotvec(np.array(rotation) * angle).as_matrix()
    origin_rot_matrix = Rotation.from_euler('xyz', origin_orientation).as_matrix()
    combined_rot_matrix = origin_rot_matrix @ rot_matrix
    transform = np.eye(4)
    transform[:3, :3] = combined_rot_matrix
    transform[:3, 3] = translation
    return transform

def inverse_kinematics(target_position, target_orientation, initial_joint_angles, max_iterations=1000, tolerance=0.01):
    joint_angles = np.array(initial_joint_angles, dtype=float)
    error_history = []  # 记录误差历史
    
    # 分别设置位置和姿态的容忍度
    position_tolerance = tolerance
    orientation_tolerance = tolerance * 2  # 姿态容忍度放宽一些
    
    # 初始步长和自适应参数
    step_size = 0.1  # 较小的初始步长
    min_step_size = 0.001
    max_step_size = 0.5
    step_increase_factor = 1.1
    step_decrease_factor = 0.8
    
    prev_total_error = float('inf')
    stagnation_count = 0  # 停滞计数器
    
    for iteration in range(max_iterations):
        # 计算当前的末端执行器位置和姿态
        tcp_pose = forward_kinematics(joint_angles)
        current_position = np.array(tcp_pose[:3])
        current_orientation = np.array(tcp_pose[3:])
        
        # 计算位置误差
        position_error = target_position - current_position
        
        # 计算姿态误差（转换为旋转向量）
        target_rot = Rotation.from_quat(target_orientation)
        current_rot = Rotation.from_quat(current_orientation)
        # 计算从当前姿态到目标姿态的旋转差
        error_rot = target_rot * current_rot.inv()
        # 转换为旋转向量（3维）
        orientation_error = error_rot.as_rotvec()
        
        # 检查收敛条件
        pos_norm = np.linalg.norm(position_error)
        ori_norm = np.linalg.norm(orientation_error)
        total_error = pos_norm + ori_norm
        error_history.append((pos_norm, ori_norm, total_error))
        
        # 使用分别的容忍度检查收敛
        if pos_norm < position_tolerance and ori_norm < orientation_tolerance:
            print(f"逆运动学求解成功，迭代 {iteration} 次收敛")
            print(f"最终误差 - 位置: {pos_norm:.6f}, 姿态: {ori_norm:.6f}")
            return joint_angles
        
        # 计算雅可比矩阵
        jacobian = compute_jacobian(joint_angles)
        
        # 检查雅可比矩阵条件数
        jacobian_condition = np.linalg.cond(jacobian)
        
        # 如果条件数太大，说明接近奇异配置
        if jacobian_condition > 1e6:
            print(f"警告: 雅可比矩阵条件数过大 ({jacobian_condition:.2e})，使用阻尼最小二乘")
            # 添加阻尼项来提高数值稳定性
            damping_factor = max(1e-4, 1.0 / jacobian_condition)
            jacobian_damped = jacobian.T @ jacobian + damping_factor * np.eye(jacobian.shape[1])
            jacobian_pseudo_inverse = np.linalg.inv(jacobian_damped) @ jacobian.T
        else:
            # 计算伪逆雅可比矩阵
            jacobian_pseudo_inverse = np.linalg.pinv(jacobian)
        
        # 计算关节角度增量
        error_vector = np.concatenate((position_error, orientation_error))
        delta_angles = jacobian_pseudo_inverse @ error_vector
        
        # 限制增量大小防止大幅跳跃
        max_delta = 0.15  # 减小最大增量
        delta_norm = np.linalg.norm(delta_angles)
        if delta_norm > max_delta:
            delta_angles = delta_angles * (max_delta / delta_norm)
        
        # 检查是否停滞
        if abs(total_error - prev_total_error) < tolerance * 0.001:
            stagnation_count += 1
            if stagnation_count > 10:
                # 如果停滞，尝试增加步长突破局部最优
                step_size = min(step_size * 1.5, max_step_size)
                stagnation_count = 0
        else:
            stagnation_count = 0
        
        # 自适应步长调整
        if total_error < prev_total_error * 0.99:  # 误差明显减小
            step_size = min(step_size * step_increase_factor, max_step_size)
        elif total_error > prev_total_error * 1.01:  # 误差明显增大
            step_size = max(step_size * step_decrease_factor, min_step_size)
        
        # 更新关节角度
        new_joint_angles = joint_angles + step_size * delta_angles
        
        # 应用关节限制约束
        new_joint_angles = apply_joint_limits(new_joint_angles)
        
        # 检查更新是否有效
        if np.allclose(new_joint_angles, joint_angles, atol=1e-8):
            print(f"关节角度变化极小，在迭代 {iteration} 停止")
            break
            
        joint_angles = new_joint_angles
        prev_total_error = total_error
        
        # 如果迭代次数较多，打印中间状态
        if iteration % 100 == 0 and iteration > 0:
            print(f"  迭代 {iteration}: 位置误差={pos_norm:.6f}, 姿态误差={ori_norm:.6f}, 步长={step_size:.4f}")
    
    # 求解失败，打印详细调试信息
    print("\n" + "="*60)
    print("逆运动学求解失败 - 调试信息")
    print("="*60)
    print(f"目标位置: {target_position}")
    print(f"目标姿态(四元数): {target_orientation}")
    print(f"最大迭代次数: {max_iterations}")
    print(f"位置容忍度: {position_tolerance}")
    print(f"姿态容忍度: {orientation_tolerance}")
    print(f"最终位置误差: {error_history[-1][0]:.6f}")
    print(f"最终姿态误差: {error_history[-1][1]:.6f}")
    print(f"最终关节角度: {joint_angles}")
    print(f"最终步长: {step_size:.6f}")
    
    # 检查雅可比矩阵奇异性
    final_jacobian = compute_jacobian(joint_angles)
    final_condition = np.linalg.cond(final_jacobian)
    print(f"最终雅可比矩阵条件数: {final_condition:.2f}")
    if final_condition > 1e6:
        print("警告: 雅可比矩阵接近奇异，可能处于奇异配置")
    
    # 检查关节限制
    joint_limits_violated = check_joint_limits(joint_angles)
    if joint_limits_violated:
        print(f"关节限制违反: {joint_limits_violated}")
    
    # 分析误差趋势
    if len(error_history) > 10:
        recent_errors = [err[2] for err in error_history[-10:]]
        if recent_errors[-1] > recent_errors[0]:
            print("警告: 最近10次迭代误差呈上升趋势，可能发散")
        elif abs(recent_errors[-1] - recent_errors[-5]) < tolerance * 0.01:
            print("警告: 误差变化极小，可能陷入局部最优")
    
    # 检查工作空间
    target_distance = np.linalg.norm(target_position)
    print(f"目标位置距离原点: {target_distance:.4f}m")
    
    # 估算机械臂最大伸展
    max_reach = estimate_max_reach()
    print(f"机械臂估计最大伸展: {max_reach:.4f}m")
    if target_distance > max_reach * 0.9:
        print("警告: 目标位置接近或超出机械臂工作空间边界")
    
    # 如果误差足够小，可能是容忍度设置过严格
    final_pos_error = error_history[-1][0]
    final_ori_error = error_history[-1][1]
    if final_pos_error < position_tolerance * 2 and final_ori_error < orientation_tolerance * 2:
        print("提示: 最终误差相对较小，可能需要调整容忍度或增加迭代次数")
    
    # 打印误差变化趋势
    print("\n误差变化趋势 (最后10次迭代):")
    start_idx = max(0, len(error_history) - 10)
    for i in range(start_idx, len(error_history)):
        pos_err, ori_err, total_err = error_history[i]
        print(f"  迭代 {i}: 位置={pos_err:.6f}, 姿态={ori_err:.6f}, 总计={total_err:.6f}")
    
    print("="*60)
    raise ValueError("逆运动学求解未收敛")

def apply_joint_limits(joint_angles):
    """应用关节限制约束"""
    constrained_angles = joint_angles.copy()
    for i, angle in enumerate(joint_angles):
        if i < len(so100_joints_data) - 1:  # 排除基座虚拟关节
            joint_data = so100_joints_data[i + 1]  # +1 因为跳过基座
            limits = joint_data["joint_limit"]
            # 将角度约束在限制范围内
            constrained_angles[i] = np.clip(angle, limits[0], limits[1])
    return constrained_angles

def estimate_max_reach():
    """估算机械臂最大伸展距离"""
    # 简单估算：所有连杆长度之和
    total_length = 0
    for joint_data in so100_joints_data:
        translation = joint_data["translation"]
        total_length += np.linalg.norm(translation)
    return total_length

def check_joint_limits(joint_angles):
    """检查关节角度是否超出限制"""
    violations = []
    for i, angle in enumerate(joint_angles):
        if i < len(so100_joints_data) - 1:  # 排除基座虚拟关节
            joint_data = so100_joints_data[i + 1]  # +1 因为跳过基座
            limits = joint_data["joint_limit"]
            if angle < limits[0] or angle > limits[1]:
                violations.append(f"关节{i+1}: {angle:.3f} 超出范围 [{limits[0]:.3f}, {limits[1]:.3f}]")
    return violations

def forward_kinematics(joint_angles):
    """Compute the forward kinematics for the SO-100 robot arm.
    
    Parameters:
        joint_angles (list): A list of joint angles [j1, j2, j3, j4, j5, j6, j7].
        
    Returns:
        np.ndarray: The position and orientation (as a quaternion) of the TCP.
    """
    # Initialize the transformation matrix as an identity matrix
    transformi = np.eye(4)
    
    # Define the joints data
    joints_data = so100_joints_data
    
    # Insert a zero angle for the base joint
    joint_angles = [0] + joint_angles
    
    # Compute the transformation matrix for each joint
    for i, angle_i in enumerate(joint_angles):
        joint_i = joints_data[i]
        transformi = transformi @ _transformation_matrix(
            joint_i["translation"], joint_i["orientation"], joint_i["rotate_axis"], angle_i
        )
    
    # Extract the position and orientation
    position = transformi[:3, 3]
    orientation = Rotation.from_matrix(transformi[:3, :3]).as_quat()
    
    return np.concatenate((position, orientation)).tolist()

def compute_jacobian(joint_angles):
    """Compute the Jacobian matrix for the SO-100 robot arm.
    
    Parameters:
        joint_angles (list): A list of joint angles [j1, j2, j3, j4, j5, j6, j7].
        
    Returns:
        np.ndarray: The Jacobian matrix.
    """

    # Initialize the Jacobian matrix
    num_joints = len(joint_angles)
    jacobian = np.zeros((6, num_joints))
    
    # 计算末端执行器位置
    tcp_pose = forward_kinematics(joint_angles)
    end_effector_pos = np.array(tcp_pose[:3])
    
    # Define the joints data
    joints_data = so100_joints_data
    # Initialize the transformation matrix as an identity matrix
    transform_cumulative = np.eye(4)
    
    # Compute the transformation matrix for each joint
    for i in range(num_joints):
        joint_i = joints_data[i]
        angle_i = joint_angles[i]
        
        # Compute the transformation matrix for the current joint
        transform_cumulative = transform_cumulative @ _transformation_matrix(
            joint_i["translation"], joint_i["orientation"], joint_i["rotate_axis"], angle_i
        )
        
        # Extract the rotation matrix and position vector of current joint
        rotation_matrix = transform_cumulative[:3, :3]
        joint_position = transform_cumulative[:3, 3]
        
        # Compute the Jacobian columns
        z_axis = rotation_matrix @ np.array(joint_i["rotate_axis"])
        # 计算从当前关节到末端执行器的位置向量
        position_diff = end_effector_pos - joint_position
        jacobian[:3, i] = np.cross(z_axis, position_diff)
        jacobian[3:, i] = z_axis
    
    return jacobian

# 示例调用和测试
if __name__ == "__main__":
    print("="*50)
    print("SO-100 机械臂逆运动学求解测试")
    print("="*50)
    
    # 测试案例 1: 从零位置开始
    joints0 = [0, 0, 0, 0, 0]
    pose0 = forward_kinematics(joints0)
    print(f"\n测试案例 1:")
    print(f"原始关节角度: {joints0}")
    print(f"正运动学计算的姿态: {pose0}")
    
    # 从稍微不同的初始位置求解逆运动学
    initial_joints = [0.1, 0.1, 0.1, 0.1, 0.1]
    try:
        joints1 = inverse_kinematics(pose0[:3], pose0[3:], initial_joints)
        print(f"逆运动学求解的关节角度: {joints1}")
        
        # 验证求解结果
        verify_pose = forward_kinematics(joints1)
        pos_error = np.linalg.norm(np.array(pose0[:3]) - np.array(verify_pose[:3]))
        ori_error = np.linalg.norm(np.array(pose0[3:]) - np.array(verify_pose[3:]))
        print(f"验证 - 位置误差: {pos_error:.6f}, 姿态误差: {ori_error:.6f}")
        
    except ValueError as e:
        print(f"逆运动学求解失败: {e}")
    
    # 测试案例 2: 不同的关节配置
    print(f"\n测试案例 2:")
    joints2 = [0.5, -0.3, 0.8, -0.2, 0.4]
    pose2 = forward_kinematics(joints2)
    print(f"原始关节角度: {joints2}")
    print(f"正运动学计算的姿态: {pose2}")
    
    try:
        joints2_solved = inverse_kinematics(pose2[:3], pose2[3:], [0, 0, 0, 0, 0])
        print(f"逆运动学求解的关节角度: {joints2_solved}")
        
        # 验证求解结果
        verify_pose2 = forward_kinematics(joints2_solved)
        pos_error2 = np.linalg.norm(np.array(pose2[:3]) - np.array(verify_pose2[:3]))
        ori_error2 = np.linalg.norm(np.array(pose2[3:]) - np.array(verify_pose2[3:]))
        print(f"验证 - 位置误差: {pos_error2:.6f}, 姿态误差: {ori_error2:.6f}")
        
    except ValueError as e:
        print(f"逆运动学求解失败: {e}")
    
    print("\n" + "="*50)


