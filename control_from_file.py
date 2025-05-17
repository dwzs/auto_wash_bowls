import time
import numpy as np
from so100_controller import So100Robot

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def read_joint_array_from_file(filename):
    """从文件读取一行5个数字的关节数组"""
    try:
        with open(filename, 'r') as f:
            for line in f:
                # 支持空格或逗号分隔
                parts = line.strip().replace(',', ' ').split()
                if len(parts) != 5:
                    continue
                try:
                    arr = np.array([float(x) for x in parts])
                    yield arr
                except ValueError:
                    continue
    except Exception as e:
        print(f"读取文件出错: {e}")

def quat_to_rotmat(q):
    """四元数转旋转矩阵，q = [w, x, y, z]"""
    w, x, y, z = q
    R = np.array([
        [1-2*(y**2+z**2), 2*(x*y-z*w),   2*(x*z+y*w)],
        [2*(x*y+z*w),     1-2*(x**2+z**2), 2*(y*z-x*w)],
        [2*(x*z-y*w),     2*(y*z+x*w),   1-2*(x**2+y**2)]
    ])
    return R

def plot_quaternions(q_list):
    """可视化多个四元数对应的姿态在一个图中"""
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    origin = np.zeros(3)
    axis_length = 1.0

    colors = ['r', 'g', 'b']
    labels = ['X', 'Y', 'Z']

    for q in q_list:
        R = quat_to_rotmat(q)
        axes = np.eye(3) * axis_length
        for i in range(3):
            vec = R @ axes[:, i]
            ax.quiver(*origin, *vec, color=colors[i], label=f'{labels[i]} (q={q})')

    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    plt.title('Multiple Quaternion Attitude Visualization')
    plt.show()

def main():
    robot = So100Robot()
    filename = 'joints.txt'
    q_list = []
    try:
        for joint_array in read_joint_array_from_file(filename):
            print(f"读取到关节数组: {joint_array}")
            success = robot.arm.set_joints(joint_array, wait=True)
            pose = robot.arm.get_tcp_pose()
            # print(f"机械臂位置: {pose}")
            print(f"机械臂位置: {pose[:3]}")
            if not success:
                print("机械臂运动失败")
            q_list.append(pose[3:])
            # plot_quaternions([pose[3:]])
            time.sleep(1)
        
    except KeyboardInterrupt:
        print("用户中断，程序结束")
    finally:
        # Display all quaternions in q_list simultaneously
        # plot_quaternions(q_list)
        robot.close()

if __name__ == '__main__':
    main()
