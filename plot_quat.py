import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def quat_to_rotmat(q):
    """四元数转旋转矩阵，q = [w, x, y, z]"""
    w, x, y, z = q
    R = np.array([
        [1-2*(y**2+z**2), 2*(x*y-z*w),   2*(x*z+y*w)],
        [2*(x*y+z*w),     1-2*(x**2+z**2), 2*(y*z-x*w)],
        [2*(x*z-y*w),     2*(y*z+x*w),   1-2*(x**2+y**2)]
    ])
    return R

def plot_quaternion(q):
    """可视化四元数对应的姿态"""
    R = quat_to_rotmat(q)
    origin = np.zeros(3)
    axis_length = 1.0

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # 绘制旋转后的坐标轴
    axes = np.eye(3) * axis_length
    colors = ['r', 'g', 'b']
    labels = ['X', 'Y', 'Z']
    for i in range(3):
        vec = R @ axes[:, i]
        ax.quiver(*origin, *vec, color=colors[i], label=labels[i])

    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    plt.title('Quaternion Attitude Visualization')
    plt.show()

if __name__ == "__main__":
    # 示例四元数，单位四元数（无旋转）
    q = [1, 0, 0, 0]
    # q = [0.86, -0.0, -0.0, 0.51]
    plot_quaternion(q)
