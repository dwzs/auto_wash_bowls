# import numpy as np
# import math
# from scipy.optimize import minimize

# so100_joints_data = [
#     {"translation": [0, 0, 0],            "orientation": [0, 0, 0], "rotate_axis": [0, 0, 1], "joint_limit": [-0, 0]}, # 基座虚拟关节
#     {"translation": [0, -0.0452, 0.0165], "orientation": [1.5708, 0, 0], "rotate_axis": [0, 1, 0], "joint_limit": [-2, 2]}, # 对应 joint_angles[0]
#     {"translation": [0, 0.1025, 0.0306],  "orientation": [0, 0, 0], "rotate_axis": [1, 0, 0], "joint_limit": [-1.8, 1.8]}, # 对应 joint_angles[1]
#     {"translation": [0, 0.11257, 0.028],  "orientation": [0, 0, 0], "rotate_axis": [1, 0, 0], "joint_limit": [-1.8, 1.57]},# 对应 joint_angles[2]
#     {"translation": [0, 0.0052, 0.1349],  "orientation": [0, 0, 0], "rotate_axis": [1, 0, 0], "joint_limit": [-3.6, 0.3] }, # 对应 joint_angles[3]
#     {"translation": [0, -0.0601, 0],      "orientation": [0, 0, 0], "rotate_axis": [0, 1, 0], "joint_limit": [-1.57, 1.57]}, # 对应 joint_angles[4]
#     {"translation": [0, -0.1, 0],         "orientation": [0, 0, 0], "rotate_axis": [0, 0, 1], "joint_limit": [0, 0]} # tcp 虚拟关节
# ]


# position_joints = so100_joints_data[:4]
# orientation_joints = so100_joints_data[4:]

# print(position_joints)
# print(len(position_joints))
# print(orientation_joints)
# print(len(orientation_joints))




import time
import roboticstoolbox as rtb

panda = rtb.models.Panda()
# print("panda: ", panda)
Tep = panda.fkine([0, -0.3, 0, -2.2, 0, 2, 0.7854])
start_time = time.time()
joints = panda.ik_LM(Tep)  #c++ 版本逆解，速度快 80us
end_time = time.time()
print("joints: ", joints)
print("time: ", end_time - start_time)

start_time = time.time()
joints = panda.ikine_LM(Tep) #python 版本逆解，速度慢 100ms
end_time = time.time()
print("joints: ", joints)
print("time: ", end_time - start_time)