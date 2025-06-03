from roboticstoolbox import ERobot
import numpy as np
import os
from scipy.spatial.transform import Rotation as R

# 获取脚本所在目录
script_dir = os.path.dirname(os.path.abspath(__file__))

# 使用绝对路径加载 URDF 文件
urdf_file = os.path.join(script_dir, 'resources/SO_5DOF_ARM100_8j_URDF.SLDASM.urdf')
so100 = ERobot.URDF(urdf_file)
# print(so100)

def create_pose_matrix(pose_components):
    """
    Converts a pose (position and quaternion) to a 4x4 homogeneous transformation matrix.

    Args:
        pose_components (list or np.ndarray): A list or array [x, y, z, qx, qy, qz, qw]
                                             representing position and quaternion.

    Returns:
        np.ndarray: A 4x4 homogeneous transformation matrix.
    """
    position = pose_components[:3]
    quaternion = pose_components[3:] # Assumes [qx, qy, qz, qw]

    # Convert quaternion to rotation matrix
    rotation_matrix = R.from_quat(quaternion).as_matrix()

    # Create homogeneous transformation matrix
    pose_matrix = np.eye(4)
    pose_matrix[:3, :3] = rotation_matrix
    pose_matrix[:3, 3] = position
    return pose_matrix

def forward_kinematics(joint_values_list):
    """
    包装工具箱的fk，输出的由旋转矩阵变成list[x, y, z, qx, qy, qz, qw]
    """
    fk_pose_matrix = so100.fkine(np.array(joint_values_list))
    position = fk_pose_matrix.t
    quaternion_xyzw = R.from_matrix(fk_pose_matrix.R).as_quat() 
    pose_list = list(position) + list(quaternion_xyzw)
    return pose_list

def inverse_kinematics(target_pose_list, initial_joint_guess=None):
    """
    包装工具箱的ik，输入由旋转矩阵变成list[x, y, z, qx, qy, qz, qw]
    """
    target_pose_matrix = create_pose_matrix(target_pose_list)
    
    q_guess = np.array(initial_joint_guess) if initial_joint_guess else None
    # mask=[1, 1, 1, 0, 0, 0]
    # mask=[1, 1, 1, 1, 0, 1]
    mask=[1, 1, 1, 0, 1, 1]

    solution = so100.ikine_LM(target_pose_matrix, q0=q_guess, mask=mask)
    print("solution: ", solution)
    if solution.success:
        return list(solution.q)
    else:
        print(f"Inverse kinematics failed to find a solution for pose: {target_pose_list}")
        return None

def test_ikine_LM(joints):
    print(" ")
    print("--------------------------------")
    print("original joints: ", joints)
    fk_pose = forward_kinematics(joints)
    print("forward kinematics pose: ", fk_pose)
    solution = inverse_kinematics(fk_pose)
    print("inverse kinematics solution: ", solution)
    return solution


# so100.plot(so100.qz, backend="swift")
# so100.plot()

joints1 = [0, 0, 0, 0, 0]
joints2 = [0.5, 0, 0, 0, 0]
joints3 = [0, 0.5, 0, 0, 0]
joints4 = [0, 0.5, 0, 0.3, 0]
pose1_components = [0.11, -0.1, 0.2, 0, 0, 0, 1] # x, y, z, qx, qy, qz, qw

# solution1 = test_ikine_LM(joints1)
# solution2 = test_ikine_LM(joints2)
# solution3 = test_ikine_LM(joints3)
# solution4 = test_ikine_LM(joints4)

solution_tmp = inverse_kinematics(pose1_components)
print("solution_tmp: ", solution_tmp)   
