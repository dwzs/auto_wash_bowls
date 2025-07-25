

from pgraph.PGraph import np
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS', 'Noto Sans CJK SC']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

def visualize_rotation(original_rotation, new_rotation, title="Rotation Visualization"):
    """
    可视化旋转前后的坐标系变化
    
    Args:
        original_rotation: 原始欧拉角 [roll, pitch, yaw]
        new_rotation: 新的欧拉角 [roll, pitch, yaw]
        title: 图表标题
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # 定义单位坐标轴
    axes = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])  # X, Y, Z轴
    colors = ['red', 'green', 'blue']
    labels = ['X', 'Y', 'Z']
    
    # 绘制原始坐标系
    original_rot = Rotation.from_euler('xyz', original_rotation)
    rotated_axes_orig = original_rot.apply(axes)
    
    for i, (axis, color, label) in enumerate(zip(rotated_axes_orig, colors, labels)):
        ax.quiver(0, 0, 0, axis[0], axis[1], axis[2], 
                 color=color, alpha=0.5, linewidth=2, 
                 label=f'Original {label}')
    
    # 绘制新坐标系
    new_rot = Rotation.from_euler('xyz', new_rotation)
    rotated_axes_new = new_rot.apply(axes)
    
    for i, (axis, color, label) in enumerate(zip(rotated_axes_new, colors, labels)):
        ax.quiver(0, 0, 0, axis[0], axis[1], axis[2], 
                 color=color, alpha=1.0, linewidth=3, 
                 label=f'New {label}', linestyle='--')
    
    # 设置图形
    ax.set_xlim([-1.5, 1.5])
    ax.set_ylim([-1.5, 1.5])
    ax.set_zlim([-1.5, 1.5])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    ax.set_title(title)
    
    plt.show()



def is_position_reachable(position):
    max_radius = 0.1
    position_radius = position[0]**2 + position[1]**2 + position[2]**2
    print(f"position_radius: {position_radius}")
    if position_radius > max_radius**2:
        return False
    return True

def get_vector_from_rotation(rotation, axis):
    rotation = Rotation.from_euler('xyz', rotation)
    vector = rotation.apply(axis)
    # print(f"rotation: {rotation}")
    # print(f"axis: {axis}")
    # print(f"vector: {vector}")
    return vector

def get_theta_from_position(position):
    """
    根据位置获取角度，角度范围 [0, 2π]
    
    Args:
        position: [x, y, z] 坐标
    
    Returns:
        theta: 角度 (弧度)，范围 [0, 2π]
    """
    print(f"position: {position}")
    
    x, y = position[0], position[1]
    
    # 使用 arctan2 获取角度 [-π, π]
    theta = np.arctan2(y, x)
    
    # 转换到 [0, 2π] 范围
    if theta < 0:
        theta += 2 * np.pi
    
    print(f"theta: {theta:.4f} 弧度 ({np.degrees(theta):.2f}°)")
    
    return theta

def get_theta_from_rotation(rotation):
    """
    根据旋转获取角度，角度范围 [0, 2π]
    
    Args:
        rotation: 欧拉角 [roll, pitch, yaw]
    
    Returns:
        theta: 角度 (弧度)，范围 [0, 2π]
    """
    print(f"rotation: {rotation}")
    
    # 获取旋转后的Y轴向量
    vy = get_vector_from_rotation(rotation, [0, 1, 0])
    
    # 在XY平面投影的坐标
    x, y = vy[0], vy[1]
    
    # 使用 arctan2 获取角度 [-π, π]
    theta = np.arctan2(y, x)
    

    # 转换到 [0, 2π] 范围
    if theta < 0:
        theta += 2 * np.pi

    print(f"theta: {theta:.4f} 弧度 ({np.degrees(theta):.2f}°)")

    if theta >= np.pi:  # 针对so100 夹爪，需要把夹爪y轴方向转180度
        theta = theta - np.pi
    elif theta < np.pi:
        theta = np.pi + theta
    else:
        theta = theta
    
    print(f"theta: {theta:.4f} 弧度 ({np.degrees(theta):.2f}°)")
    
    return theta

def is_rotation_reachable(pose):
    position = pose[:3]
    rotation = pose[3:]

    theta_position = get_theta_from_position(position)
    theta_rotation = get_theta_from_rotation(rotation)

    # print(f"theta_position: {theta_position}, theta_rotation: {theta_rotation}")
    if abs(theta_position) - abs(theta_rotation) == 0:
        return True
    
    print(f"pose: {pose} is not reachable")
    print(f"theta_position: {theta_position}, theta_rotation: {theta_rotation}")
    print(f"delta_theta: { abs(theta_position) - abs(theta_rotation)}")
    return False

def roll_rotation(rotation, angle, axis = [0, 0, 1]):
    """控制一个rotation（欧拉角）绕轴旋转angle角度"""
    original_rotation = Rotation.from_euler('xyz', rotation)
    axis = np.array(axis)
    axis = axis / np.linalg.norm(axis)
    axis_rotation = Rotation.from_rotvec(angle * axis)
    combined_rotation = axis_rotation * original_rotation
    new_euler = combined_rotation.as_euler('xyz')
    return new_euler.tolist()


def tuning_rotation_reachable(pose):
    position = pose[:3]
    rotation = pose[3:]
    theta_vecticalPlane = get_theta_from_position(position)
    theta_rotation = get_theta_from_rotation(rotation)
    theta_jaw2vecticalPlane = theta_vecticalPlane - theta_rotation

    print(f"theta_jaw2vecticalPlane: {theta_jaw2vecticalPlane}")

    new_rotation = roll_rotation(rotation, theta_jaw2vecticalPlane)
    new_pose = pose.copy()
    new_pose[3:] = new_rotation
    # if not is_rotation_reachable(new_pose):
    #     print(f"solution: {new_pose} is not reachable, roll to: {-theta_jaw2vecticalPlane}")
    #     new_rotation = roll_rotation(rotation, -theta_jaw2vecticalPlane)

    if not is_rotation_reachable(new_pose):
        print(f"solution: {new_pose} is not reachable")
        return None

    print(f"original_rotation: {rotation}")
    print(f"new_rotation: {new_rotation}")
    # print(f"theta_position: {theta_position}, theta_rotation: {theta_rotation}")
    return new_rotation





def main():
    # # 测试1：绕Z轴旋转90度
    # initial = [0, 0, 0]
    # result = roll_rotation(initial, np.pi/2, [0, 0, 1])
    # print(f"Around Z-axis rotate 90°: {[np.degrees(x) for x in result]}°")
    # visualize_rotation(initial, result, "Rotate 90° around Z-axis")
    
    # # 测试1：绕Z轴旋转90度
    # initial = [np.pi/4, 0, 0]
    # result = roll_rotation(initial, np.pi/4, [0, 0, 1])
    # print(f"Around Z-axis rotate 90°: {[np.degrees(x) for x in result]}°")
    # visualize_rotation(initial, result, "Rotate 90° around Z-axis")
    
    pose = [0.2, -0.0, 0.3, 0, 0, np.pi/4]
    new_rotation = tuning_rotation_reachable(pose)
    # print(f"new_rotation: {new_rotation}")
    

    # position = [1, 0, 0.1]
    # get_theta_from_position(position)

    # rotation = [0, 0, 3 * np.pi/4]
    # get_theta_from_rotation(rotation)


if __name__ == "__main__":
    main()

