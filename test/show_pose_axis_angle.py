#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""交互式位姿显示工具 - 轴角(空间轴)"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import Slider, TextBox, Button
from scipy.spatial.transform import Rotation

# ==================== 轴显示范围设置 ====================
AXIS_RANGE_XY = 0.5    # X, Y轴显示范围：-0.5m 到 +0.5m
AXIS_RANGE_Z = 1       # Z轴显示范围：0m 到 1m
# =======================================================

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class InteractivePoseViewer:
	def __init__(self, pose):
		# pose: [x, y, z, ax, ay, az, angle(rad)]
		self.pose = pose
		self._updating = False

		self.fig = plt.figure(figsize=(18, 12))
		self.ax = self.fig.add_subplot(121, projection='3d')

		# 滑块区域：X,Y,Z + 轴(ax,ay,az) + angle
		slider_positions = [
			[0.60, 0.86, 0.30, 0.03],  # x
			[0.60, 0.81, 0.30, 0.03],  # y
			[0.60, 0.76, 0.30, 0.03],  # z
			[0.60, 0.68, 0.30, 0.03],  # ax
			[0.60, 0.63, 0.30, 0.03],  # ay
			[0.60, 0.58, 0.30, 0.03],  # az
			[0.60, 0.50, 0.30, 0.03],  # angle
		]

		self.sliders = []
		slider_params = [
			('X (m)',     -AXIS_RANGE_XY, AXIS_RANGE_XY, self.pose[0]),
			('Y (m)',     -AXIS_RANGE_XY, AXIS_RANGE_XY, self.pose[1]),
			('Z (m)',      0.0,           AXIS_RANGE_Z,  self.pose[2]),
			('Axis X',     -1.0,          1.0,           self.pose[3]),
			('Axis Y',     -1.0,          1.0,           self.pose[4]),
			('Axis Z',     -1.0,          1.0,           self.pose[5]),
			('Angle (rad)',-np.pi,        np.pi,         self.pose[6]),
		]
		for i, (label, vmin, vmax, vinit) in enumerate(slider_params):
			ax = self.fig.add_axes(slider_positions[i])
			s = Slider(ax, label, vmin, vmax, valinit=vinit)
			s.on_changed(self.update_pose)
			self.sliders.append(s)

		# 输入框：x,y,z,ax,ay,az,angle
		self.input_ax = self.fig.add_axes([0.60, 0.42, 0.30, 0.05])
		self.pose_box = TextBox(self.input_ax, 'Pose [x,y,z,ax,ay,az,angle]:', initial=self._pose_str())
		self.pose_box.on_submit(self._on_pose_submit)

		# 重置按钮
		self.reset_ax = self.fig.add_axes([0.60, 0.36, 0.10, 0.05])
		self.reset_btn = Button(self.reset_ax, 'Reset')
		self.reset_btn.on_clicked(self._on_reset)

		# 文本显示
		self.text_ax = self.fig.add_axes([0.60, 0.15, 0.30, 0.15])
		self.text_ax.axis('off')

		self.update_plot()

	def _pose_str(self):
		return f"{self.pose[0]:.3f}, {self.pose[1]:.3f}, {self.pose[2]:.3f}, {self.pose[3]:.3f}, {self.pose[4]:.3f}, {self.pose[5]:.3f}, {self.pose[6]:.3f}"

	def _on_pose_submit(self, text: str):
		if self._updating:
			return
		try:
			vals = [float(x.strip()) for x in text.split(',')]
			if len(vals) != 7:
				print("格式错误：需要7个数值 x,y,z,ax,ay,az,angle（弧度）"); return
			x, y, z, ax, ay, az, ang = vals
			if abs(x) > AXIS_RANGE_XY or abs(y) > AXIS_RANGE_XY or not (0 <= z <= AXIS_RANGE_Z):
				print(f"位置超出范围：X,Y ∈ [-{AXIS_RANGE_XY},{AXIS_RANGE_XY}], Z ∈ [0,{AXIS_RANGE_Z}]"); return
			# 驱动滑条
			self.sliders[0].set_val(x);  self.sliders[1].set_val(y);  self.sliders[2].set_val(z)
			self.sliders[3].set_val(ax); self.sliders[4].set_val(ay); self.sliders[5].set_val(az)
			self.sliders[6].set_val(ang)
		except Exception as e:
			print("解析失败：", e)

	def update_pose(self, _):
		# 从滑条读取
		self.pose[0] = self.sliders[0].val
		self.pose[1] = self.sliders[1].val
		self.pose[2] = self.sliders[2].val
		self.pose[3] = self.sliders[3].val
		self.pose[4] = self.sliders[4].val
		self.pose[5] = self.sliders[5].val
		self.pose[6] = self.sliders[6].val

		# 同步输入框文本
		self._updating = True
		self.pose_box.set_val(self._pose_str())
		self._updating = False

		self.update_plot()

	def _axis_unit_and_angle(self):
		axis = np.array(self.pose[3:6], dtype=float)
		ang = float(self.pose[6])
		n = np.linalg.norm(axis)
		if n < 1e-8:
			axis_unit = np.array([1.0, 0.0, 0.0])  # 角度为0时轴无关，这里给默认
		else:
			axis_unit = axis / n
		return axis_unit, ang

	def update_plot(self):
		self.ax.clear()

		position = np.array(self.pose[:3])
		axis_unit, ang = self._axis_unit_and_angle()

		# 世界坐标轴
		self.ax.scatter(0, 0, 0, color='black', s=100, label='Origin')
		world_axis_length = 0.15
		self.ax.quiver(0, 0, 0, world_axis_length, 0, 0, color='red',   alpha=0.5, linewidth=2, label='World X')
		self.ax.quiver(0, 0, 0, 0, world_axis_length, 0, color='green', alpha=0.5, linewidth=2, label='World Y')
		self.ax.quiver(0, 0, 0, 0, 0, world_axis_length, color='blue',  alpha=0.5, linewidth=2, label='World Z')

		# 姿态：空间轴角（轴在世界坐标系定义）
		rotation = Rotation.from_rotvec(axis_unit * ang)
		R = rotation.as_matrix()
		pose_axis_length = 0.1

		# 位姿点
		self.ax.scatter(*position, color='orange', s=150, label='Pose')

		# 位姿坐标系（物体自身坐标轴）
		for i, (color, name) in enumerate([('red', 'X'), ('green', 'Y'), ('blue', 'Z')]):
			axis_vec = R[:, i] * pose_axis_length
			self.ax.quiver(*position, *axis_vec, color=color, arrow_length_ratio=0.1, linewidth=3, label=f'Pose {name}')

		# 连线
		self.ax.plot([0, position[0]], [0, position[1]], [0, position[2]], 'k--', alpha=0.3, linewidth=1)

		# 固定范围
		self.ax.set_xlim([-AXIS_RANGE_XY, AXIS_RANGE_XY])
		self.ax.set_ylim([-AXIS_RANGE_XY, AXIS_RANGE_XY])
		self.ax.set_zlim([0, AXIS_RANGE_Z])
		self.ax.set_xlabel('X (m)'); self.ax.set_ylabel('Y (m)'); self.ax.set_zlabel('Z (m)')
		self.ax.set_title(f'Axis-Angle (space-fixed axis)  |  X,Y: ±{AXIS_RANGE_XY}m, Z: 0~{AXIS_RANGE_Z}m')
		self.ax.legend(); self.ax.set_box_aspect([1,1,1]); self.ax.grid(True, alpha=0.3)

		# 文本
		self.text_ax.clear(); self.text_ax.axis('off')
		txt = f"""Pose:
Position (m):  [{position[0]:.3f}, {position[1]:.3f}, {position[2]:.3f}]
Axis (world):  [{axis_unit[0]:.3f}, {axis_unit[1]:.3f}, {axis_unit[2]:.3f}]
Angle (rad):   {ang:.3f}"""
		self.text_ax.text(0, 1, txt, transform=self.text_ax.transAxes, fontfamily='monospace', fontsize=10, va='top')

		self.fig.canvas.draw()

	def _on_reset(self, _):
		# 置零（轴设为0时按默认[1,0,0]处理；angle=0 时轴无关）
		self.pose = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
		self._updating = True
		for i in range(len(self.sliders)):
			self.sliders[i].set_val(0.0)
		self.pose_box.set_val(self._pose_str())
		self._updating = False
		self.update_plot()

if __name__ == '__main__':
	try:
		# 初始位姿 [x, y, z, ax, ay, az, angle]
		pose = [0.2, -0.2, 0.2, 0.0, 0.0, 1.0, 0.5]  # 例：绕世界Z轴0.5rad
		print(f"轴显示范围: X,Y: ±{AXIS_RANGE_XY}m, Z: 0~{AXIS_RANGE_Z}m")
		print("姿态采用空间轴角: axis=(ax,ay,az) in world, angle(rad)")
		InteractivePoseViewer(pose)
		plt.show()
	except KeyboardInterrupt:
		print("\n退出")
