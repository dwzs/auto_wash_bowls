#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""交互式位姿显示工具"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import Slider, TextBox
from scipy.spatial.transform import Rotation

# ==================== 轴显示范围设置 ====================
AXIS_RANGE_XY = 0.5    # X, Y轴显示范围：-0.5m 到 +0.5m
AXIS_RANGE_Z = 1       # Z轴显示范围：0m 到 1m
# =======================================================

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class InteractivePoseViewer:
    def __init__(self, pose):
        # 初始位姿 [x, y, z, roll, pitch, yaw]
        self.pose = pose
        self._updating = False  # 防止相互触发的标志
        
        # 创建图形和子图
        self.fig = plt.figure(figsize=(14, 10))
        
        # 3D图形区域
        self.ax = self.fig.add_subplot(121, projection='3d')
        
        # 滑块区域
        slider_positions = [
            [0.6, 0.85, 0.3, 0.03],  # x
            [0.6, 0.80, 0.3, 0.03],  # y  
            [0.6, 0.75, 0.3, 0.03],  # z
            [0.6, 0.65, 0.3, 0.03],  # roll
            [0.6, 0.60, 0.3, 0.03],  # pitch
            [0.6, 0.55, 0.3, 0.03],  # yaw
        ]
        
        # 创建滑块
        self.sliders = []
        slider_params = [
            ('X (m)', -AXIS_RANGE_XY, AXIS_RANGE_XY, self.pose[0]),
            ('Y (m)', -AXIS_RANGE_XY, AXIS_RANGE_XY, self.pose[1]),
            ('Z (m)', 0.0, AXIS_RANGE_Z, self.pose[2]),
            ('Roll (rad)', -np.pi, np.pi, self.pose[3]),
            ('Pitch (rad)', -np.pi, np.pi, self.pose[4]),
            ('Yaw (rad)', -np.pi, np.pi, self.pose[5]),
        ]
        
        for i, (label, min_val, max_val, init_val) in enumerate(slider_params):
            ax = self.fig.add_axes(slider_positions[i])
            slider = Slider(ax, label, min_val, max_val, valinit=init_val)
            slider.on_changed(self.update_pose)
            self.sliders.append(slider)
        
        # 输入框：逗号分隔的 x,y,z,roll,pitch,yaw（弧度）
        self.input_ax = self.fig.add_axes([0.6, 0.28, 0.3, 0.05])
        self.pose_box = TextBox(self.input_ax, 'Pose:', initial=self._pose_str())
        self.pose_box.on_submit(self._on_pose_submit)

        # 添加文本显示区域
        self.text_ax = self.fig.add_axes([0.6, 0.35, 0.3, 0.15])
        self.text_ax.axis('off')
        
        # 初始绘制
        self.update_plot()
        
    def _pose_str(self):
        return f"{self.pose[0]:.1f}, {self.pose[1]:.1f}, {self.pose[2]:.1f}, {self.pose[3]:.1f}, {self.pose[4]:.1f}, {self.pose[5]:.1f}"

    def _on_pose_submit(self, text: str):
        if self._updating:
            return
        try:
            vals = [float(x.strip()) for x in text.split(',')]
            if len(vals) != 6:
                print("格式错误：需要6个数值 x,y,z,roll,pitch,yaw（弧度）")
                return
            x, y, z, r, p, yw = vals
            if abs(x) > AXIS_RANGE_XY or abs(y) > AXIS_RANGE_XY or not (0 <= z <= AXIS_RANGE_Z):
                print(f"位置超出范围：X,Y ∈ [-{AXIS_RANGE_XY},{AXIS_RANGE_XY}], Z ∈ [0,{AXIS_RANGE_Z}]")
                return
            # 驱动滑条（联动图像和pose）
            self.sliders[0].set_val(x)
            self.sliders[1].set_val(y)
            self.sliders[2].set_val(z)
            self.sliders[3].set_val(r)
            self.sliders[4].set_val(p)
            self.sliders[5].set_val(yw)
        except Exception as e:
            print(f"解析失败：{e}")
    
    def update_pose(self, val):
        """更新位姿参数"""
        # 从滑块读取并更新 pose
        self.pose[0] = self.sliders[0].val
        self.pose[1] = self.sliders[1].val
        self.pose[2] = self.sliders[2].val
        self.pose[3] = self.sliders[3].val
        self.pose[4] = self.sliders[4].val
        self.pose[5] = self.sliders[5].val

        # 同步更新输入框文本（防回调循环）
        self._updating = True
        self.pose_box.set_val(self._pose_str())
        self._updating = False

        self.update_plot()
    
    def update_plot(self):
        """更新3D图形"""
        self.ax.clear()
        
        position = np.array(self.pose[:3])
        euler_angles = self.pose[3:]
        
        # 显示原点
        self.ax.scatter(0, 0, 0, color='black', s=100, label='Origin')
        
        # 显示世界坐标系
        world_axis_length = 0.15
        self.ax.quiver(0, 0, 0, world_axis_length, 0, 0, color='red', alpha=0.5, linewidth=2, label='World X')
        self.ax.quiver(0, 0, 0, 0, world_axis_length, 0, color='green', alpha=0.5, linewidth=2, label='World Y')
        self.ax.quiver(0, 0, 0, 0, 0, world_axis_length, color='blue', alpha=0.5, linewidth=2, label='World Z')
        
        # 计算位姿坐标轴
        rotation = Rotation.from_euler('xyz', euler_angles)
        pose_axis_length = 0.1
        
        # 显示位姿点
        self.ax.scatter(*position, color='orange', s=150, label='Pose')
        
        # 显示位姿坐标轴
        for i, (color, name) in enumerate([('red', 'X'), ('green', 'Y'), ('blue', 'Z')]):
            axis = rotation.as_matrix()[:, i] * pose_axis_length
            self.ax.quiver(*position, *axis, color=color, arrow_length_ratio=0.1, linewidth=3, 
                         label=f'Pose {name}')
        
        # 从原点到位姿点的连线
        self.ax.plot([0, position[0]], [0, position[1]], [0, position[2]], 
                    'k--', alpha=0.3, linewidth=1)
        
        # 设置固定的图形范围
        self.ax.set_xlim([-AXIS_RANGE_XY, AXIS_RANGE_XY])
        self.ax.set_ylim([-AXIS_RANGE_XY, AXIS_RANGE_XY])
        self.ax.set_zlim([0, AXIS_RANGE_Z])
        
        self.ax.set_xlabel('X (m)')
        self.ax.set_ylabel('Y (m)')
        self.ax.set_zlabel('Z (m)')
        self.ax.set_title(f'Interactive Pose Display (X,Y: ±{AXIS_RANGE_XY}m, Z: 0~{AXIS_RANGE_Z}m)')
        self.ax.legend()
        self.ax.set_box_aspect([1,1,1])
        
        # 添加网格
        self.ax.grid(True, alpha=0.3)
    
        self.fig.canvas.draw()
    

if __name__ == '__main__':
    try:
        # 输入位姿 [x, y, z, roll, pitch, yaw] (位置单位：米，角度单位：弧度)
        # pose = [0.2, -0.2, 0.2, 1.7, 0.1, 0.1]
        pose = [0.2, -0.2, 0.2, 0, 0, 0]
        
        print(f"轴显示范围设置:")
        print(f"  X, Y轴: ±{AXIS_RANGE_XY}m")
        print(f"  Z轴: 0~{AXIS_RANGE_Z}m")
        print(f"  角度单位: 弧度 (范围: ±π)")
        print(f"如需修改，请编辑文件顶部的 AXIS_RANGE_XY 和 AXIS_RANGE_Z 常量\n")
        
        viewer = InteractivePoseViewer(pose)
        plt.show()
        
    except KeyboardInterrupt:
        print("\n退出")
