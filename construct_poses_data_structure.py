import json
import time
from scipy.spatial.transform import Rotation
import numpy as np

class So100_pose_space:
    def __init__(self, length=60, width=60, height=60):
        self.length = length
        self.width = width
        self.height = height
        self.joints_interval = 90  #单位度，每个关节间隔joints_interval采一个角度
        self.number_to_list_map = {}  # Dictionary to store lists for each number
        self.pose_to_joints_map = {}  # Dictionary to store lists for each pose

        self.joints_data = [
            {"translation": [0, 0, 0],            "orientation": [0, 0, 0], "rotate_axis": [0, 0, 1], "joint_limit": [-0, 0]}, # 基座虚拟关节
            {"translation": [0, -0.0452, 0.0165], "orientation": [1.5708, 0, 0], "rotate_axis": [0, 1, 0], "joint_limit": [-2, 2]},
            {"translation": [0, 0.1025, 0.0306],  "orientation": [0, 0, 0], "rotate_axis": [1, 0, 0], "joint_limit": [-1.8, 1.8]},
            {"translation": [0, 0.11257, 0.028],  "orientation": [0, 0, 0], "rotate_axis": [1, 0, 0], "joint_limit": [-1.8, 1.57]},
            {"translation": [0, 0.0052, 0.1349],  "orientation": [0, 0, 0], "rotate_axis": [1, 0, 0], "joint_limit": [-3.6, 0.3] },
            {"translation": [0, -0.0601, 0],      "orientation": [0, 0, 0], "rotate_axis": [0, 1, 0], "joint_limit": [-1.57, 1.57]},
            {"translation": [0, -0.1, 0],         "orientation": [0, 0, 0], "rotate_axis": [0, 0, 1], "joint_limit": [0, 0]} # tcp 虚拟关节
        ]


    def forward_kinematics(self, joint_angles):
        """Compute the forward kinematics for the SO-100 robot arm.
        
        Parameters:
            joint_angles (list): A list of joint angles [j1, j2, j3, j4, j5].
            
        Returns:
            np.ndarray: The position and orientation (as a quaternion) of the TCP.
        """

        if len(joint_angles) != 5:
            print("joint_angles is: ", joint_angles)
            raise ValueError("The list must contain exactly five numbers.")

        def transformation_matrix(translation, origin_orientation, rotation, angle):
            """Create a transformation matrix for a joint."""
            rot_matrix = Rotation.from_rotvec(np.array(rotation) * angle).as_matrix()
            origin_rot_matrix = Rotation.from_euler('xyz', origin_orientation).as_matrix()
            combined_rot_matrix = origin_rot_matrix @ rot_matrix
            transform = np.eye(4)
            transform[:3, :3] = combined_rot_matrix
            transform[:3, 3] = translation
            return transform
        
        angles = joint_angles.copy()
        angles.insert(0, 0)  # 基座关节转角为0
        transformi = np.eye(4)
        for i, angle_i in enumerate(angles):
            joint_i = self.joints_data[i]
            transformi = transformi @ transformation_matrix(
                joint_i["translation"], joint_i["orientation"], joint_i["rotate_axis"], angle_i
            )
        
        position = transformi[:3, 3]
        orientation = Rotation.from_matrix(transformi[:3, :3]).as_quat()
        
        return np.concatenate((position, orientation)).tolist()



    def xyz_to_number(self, x, y, z):
        """
        将xyz坐标映射为一个数。
        """
        return x + y * self.length + z * self.length * self.width

    def number_to_xyz(self, number):
        """
        将数值映射为xyz坐标。
        """
        z = number // (self.length * self.width)
        number -= z * self.length * self.width
        y = number // self.length
        x = number % self.length
        return x, y, z

    def store_joints_for_number(self, number, values):
        """
        Store a list of five numbers for a given mapped number.
        """
        if len(values) != 5:
            raise ValueError("The list must contain exactly five numbers.")
        self.number_to_list_map[number] = values

    def get_joints_for_number(self, number):
        return self.number_to_list_map.get(number, None)

    def store_joints_for_pose(self, pose, values):
        if len(values) != 5:
            raise ValueError("The list must contain exactly five numbers.")
        self.pose_to_joints_map[pose] = values

    def get_joints_for_pose(self, pose):
        return self.pose_to_joints_map.get(pose, None)


    def iterate_joints(self):
        """
        遍历所有关节角度并记录每个 joint_angles 对应的 position_orientation
        """
        degree_interval = self.joints_interval
        radian_interval = np.deg2rad(degree_interval)
        rotation_joints_data = self.joints_data[1:-1]  # 可活动的关节
        joints_poses_map = {}  # 存储joint_angles 对应的 pose
        joint_angles_min = [joint["joint_limit"][0] for joint in rotation_joints_data]
        joint_angles_max = [joint["joint_limit"][1] for joint in rotation_joints_data]
        joint_angles = joint_angles_min.copy()

        while joint_angles[0] <= joint_angles_max[0]:
            joint_angles[1] = joint_angles_min[1]
            while joint_angles[1] <= joint_angles_max[1]:
                joint_angles[2] = joint_angles_min[2]
                while joint_angles[2] <= joint_angles_max[2]:
                    joint_angles[3] = joint_angles_min[3]
                    while joint_angles[3] <= joint_angles_max[3]:
                        joint_angles[4] = joint_angles_min[4]
                        while joint_angles[4] <= joint_angles_max[4]:
                            pose = self.forward_kinematics(joint_angles)
                            joints_poses_map[tuple(joint_angles)] = pose
                            joint_angles[4] += radian_interval
                        joint_angles[3] += radian_interval
                    joint_angles[2] += radian_interval
                joint_angles[1] += radian_interval
            joint_angles[0] += radian_interval

        return joints_poses_map

    def get_closest_pose_joints(self, pose, joints_poses_map):
        """
        Find the closest pose in joints_poses_map to the given pose.
        
        Parameters:
            pose (list): The target pose [x, y, z, [qx, qy, qz, qw]].
            joints_poses_map (dict): A dictionary mapping joint angles to poses.
        
        Returns:
            tuple: The closest pose, corresponding joints, and the distance.
        """
        min_distance = float('inf')
        closest_pose = None
        closest_joints = None

        target_position = np.array(pose[:3])
        target_orientation = np.array(pose[3])

        for joints, current_pose in joints_poses_map.items():
            current_position = np.array(current_pose[:3])
            current_orientation = np.array(current_pose[3:])

            # Calculate Euclidean distance for position
            position_distance = np.linalg.norm(target_position - current_position)

            # Calculate orientation distance using quaternion difference
            orientation_distance = np.linalg.norm(target_orientation - current_orientation)

            # Total distance as a combination of position and orientation distances
            total_distance = position_distance + orientation_distance

            if total_distance < min_distance:
                min_distance = total_distance
                closest_pose = current_pose
                closest_joints = joints

        return closest_pose, closest_joints, min_distance

    def iterate_positions(self):
        step = 5 
        x_limit = [-30, 30] # 单位cm
        y_limit = [-30, 30] # 单位cm
        z_limit = [-30, 30] # 单位cm
        oretation = [0.71, 0, 0, 0.71] # rx,ry,rz,w

        joints_poses_map = self.iterate_joints()

        for x in range(x_limit[0], x_limit[1], step):
            for y in range(y_limit[0], y_limit[1], step):
                for z in range(z_limit[0], z_limit[1], step):
                    # pose = [x, y, z, oretation]
                    pose = [x, y, z] + oretation
                    closest_pose, joints, distance = self.get_closest_pose_joints(pose, joints_poses_map)
                    self.pose_to_joints_map[tuple(pose)] = joints
        
        return self.pose_to_joints_map


    def save_data(self, file_path):
        # Convert tuple keys to strings
        serializable_map = {str(k): v for k, v in self.pose_to_joints_map.items()}
        with open(file_path, 'w') as f:
            json.dump(serializable_map, f)

# 示例用法
pose_space = So100_pose_space()

# # 将xyz坐标映射为数值
# number = pose_space.xyz_to_number(1, 0, 0)
# print(f"坐标 (1, 0, 0) 映射为数值: {number}")

# # 存储一个5个数的list
# pose_space.store_joints_for_number(number, [10, 20, 30, 40, 50])

# # 获取存储的list
# stored_list = pose_space.get_joints_for_number(number)
# print(f"数值 {number} 存储的list: {stored_list}")

# # 将数值映射为xyz坐标
# x, y, z = pose_space.number_to_xyz(25564)
# print(f"数值 25564 映射为坐标: ({x}, {y}, {z})")

time_start = time.time()
joints_poses_map = pose_space.iterate_joints()
print("joints_poses_map length: ", len(joints_poses_map))
time_end = time.time()
print("time cost: ", time_end - time_start, "s")


time_start = time.time()
pose_to_joints_map = pose_space.iterate_positions()
# print("pose_to_joints_map: ", pose_to_joints_map)
print("pose_to_joints_map length: ", len(pose_to_joints_map))
time_end = time.time()
print("time cost: ", time_end - time_start, "s")



pose_space.save_data("pose_to_joints_map.json")






# #### test: get_closest_pose_joints
# joints_to_pose_map = {}
# joints_to_pose_map[tuple([1, 0, 0, 0, 0])] = [2, 0, 0, 0, 0, 0, 0]
# joints_to_pose_map[tuple([0, 1, 0, 0, 0])] = [0, 2, 0, 0, 0, 0, 0]
# joints_to_pose_map[tuple([0, 0, 1, 0, 0])] = [0, 0, 2, 0, 0, 0, 0]
# joints_to_pose_map[tuple([0, 0, 0, 1, 0])] = [0, 0, 0, 2, 0, 0, 0]
# joints_to_pose_map[tuple([0, 0, 0, 0, 1])] = [0, 0, 0, 0, 2, 0, 0]
# pose = [0.1, 0.1, 0.1, 0, 0, 0, 0]
# pose, joints, distance = pose_space.get_closest_pose_joints(pose, joints_to_pose_map)
# print("closest pose: ", pose)
# print("joints: ", joints)
# print("distance: ", distance)

