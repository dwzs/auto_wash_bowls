import json
import time
from scipy.spatial.transform import Rotation
import numpy as np

# 存储so00所有空间pose以及其对应的joints

class So100_pose_space:
    def __init__(self):
        self.length = 0.6
        self.width = 0.6
        self.height = 0.6

        self.joints_step = 30  #单位度，每个关节间隔joints_step采一个角度
        self.position_step = 0.05 # 单位m，空间中每 position_step 采一个点

        self.threshold_pose = 0.05 #单位m，计算最近距离位姿时位置的阈值
        self.K_positon = 1 #计算最近距离位姿时位置的影响系数
        self.K_orientation = 0 #计算最近距离位姿时姿态的影响系数

        # self.number_to_list_map = {}  # Dictionary to store lists for each number
        self.joints_poses_map = {}  # Dictionary to store lists for each pose
        self.pose_to_joints_map = {}  # Dictionary to store lists for each pose

        self.joints_poses_map_file_path = "joints_poses_map.json"
        self.pose_to_joints_map_file_path = "pose_to_joints_map.json"

        self.joints_data = [
            {"translation": [0, 0, 0],            "orientation": [0, 0, 0], "rotate_axis": [0, 0, 1], "joint_limit": [-0, 0]}, # 基座虚拟关节
            {"translation": [0, -0.0452, 0.0165], "orientation": [1.5708, 0, 0], "rotate_axis": [0, 1, 0], "joint_limit": [-2, 2]}, 
            {"translation": [0, 0.1025, 0.0306],  "orientation": [0, 0, 0], "rotate_axis": [1, 0, 0], "joint_limit": [-1.8, 1.8]},
            {"translation": [0, 0.11257, 0.028],  "orientation": [0, 0, 0], "rotate_axis": [1, 0, 0], "joint_limit": [-1.8, 1.57]},
            {"translation": [0, 0.0052, 0.1349],  "orientation": [0, 0, 0], "rotate_axis": [1, 0, 0], "joint_limit": [-3.6, 0.3] },
            {"translation": [0, -0.0601, 0],      "orientation": [0, 0, 0], "rotate_axis": [0, 1, 0], "joint_limit": [-1.57, 1.57]},
            {"translation": [0, -0.1, 0],         "orientation": [0, 0, 0], "rotate_axis": [0, 0, 1], "joint_limit": [0, 0]} # tcp 虚拟关节
        ]

    def _forward_kinematics(self, joint_angles: list[float]) -> list[float]:
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

    def _iterate_joints(self):
        """
        遍历所有关节角度并记录每个 joint_angles 对应的 position_orientation
        """
        degree_interval = self.joints_step
        radian_interval = np.deg2rad(degree_interval)
        rotation_joints_data = self.joints_data[1:-1]  # 可活动的关节
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
                            pose = self._forward_kinematics(joint_angles)
                            self.joints_poses_map[tuple(joint_angles)] = pose
                            joint_angles[4] += radian_interval
                        joint_angles[3] += radian_interval
                    joint_angles[2] += radian_interval
                joint_angles[1] += radian_interval
            joint_angles[0] += radian_interval

        return self.joints_poses_map

    def _get_closest_pose_joints(self, pose, joints_poses_map):
        """
        每个cube corner pose 不一定有对应的joints，因此迭代pose时，获取pose的最近关节
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
            # print("position_distance: ", position_distance)
            # print("orientation_distance: ", orientation_distance)
            total_distance = position_distance * self.K_positon + orientation_distance * self.K_orientation

            if total_distance < min_distance:
                min_distance = total_distance
                closest_pose = current_pose
                closest_joints = joints

        return closest_pose, closest_joints, min_distance

    def _iterate_positions(self):
        '''
        遍历position空间，oreation默认朝下，获取pose的最近关节
        '''
        step = int(self.position_step * 100) # 单位cm
        x_limit = [-30, 30] # 单位cm
        y_limit = [-30, 30] # 单位cm
        z_limit = [-30, 30] # 单位cm
        oretation = [0.71, 0, 0, 0.71] # rx,ry,rz,w

        joints_poses_map = self._iterate_joints()

        for x in range(x_limit[0], x_limit[1], step):
            for y in range(y_limit[0], y_limit[1], step):
                for z in range(z_limit[0], z_limit[1], step):
                    # pose = [x, y, z, oretation]
                    pose = [x/100, y/100, z/100] + oretation  # 单位m
                    closest_pose, joints, distance = self._get_closest_pose_joints(pose, joints_poses_map)
                    # print("distance: ", distance)
                    if distance < self.threshold_pose:
                        self.pose_to_joints_map[tuple(pose)] = joints
                    else:
                        self.pose_to_joints_map[tuple(pose)] = None
        
        return self.pose_to_joints_map
    
    def construct_pose_space(self):
        # 1. 遍历关节空间，生成关节空间到pose空间的映射
        start_time = time.time()
        self.joints_poses_map = self._iterate_joints()
        end_time = time.time()
        print("joints_poses_map length: ", len(self.joints_poses_map))
        print("time cost: ", end_time - start_time, "s")
        # 2. 遍历pose空间，生成pose空间到关节空间的映射
        start_time = time.time()
        self.pose_to_joints_map = self._iterate_positions()
        end_time = time.time()
        print("pose_to_joints_map length: ", len(self.pose_to_joints_map))
        print("time cost: ", end_time - start_time, "s")

    def save_data(self):
        # 1. 存储关节到pose的映射
        serializable_map = {str(k): v for k, v in self.joints_poses_map.items()} # Convert tuple keys to strings
        with open(self.joints_poses_map_file_path, 'w') as f:
            json.dump(serializable_map, f)
        # 2. 存储pose到关节的映射
        serializable_map = {str(k): v for k, v in self.pose_to_joints_map.items()} # Convert tuple keys to strings
        with open(self.pose_to_joints_map_file_path, 'w') as f:
            json.dump(serializable_map, f)

    def cacul_joints_pose(self, joints): # 计算关节到pose的映射
        pose = self._forward_kinematics(joints)
        return pose

    def get_positon_closest_corner_position(self, pose):
        '''
        获取pose所在的cube中最近角点
        '''
        k = 100 #m to cm
        cube_length = self.position_step * k # cm为单位

        position = np.array(pose[:3]) * k # cm为单位
        # start_corner = np.floor(position).astype(int)
        start_corner = np.floor(position)
        print("pose: ", pose)
        print("position: ", position)
        print("start_corner: ", start_corner)
        print("cube_length: ", cube_length)
        corners = [
            (start_corner[0], start_corner[1], start_corner[2]),
            (start_corner[0] + cube_length, start_corner[1], start_corner[2]),
            (start_corner[0] + cube_length, start_corner[1] + cube_length, start_corner[2]),
            (start_corner[0], start_corner[1] + cube_length, start_corner[2]),
            (start_corner[0], start_corner[1], start_corner[2] + cube_length),
            (start_corner[0] + cube_length, start_corner[1], start_corner[2] + cube_length),
            (start_corner[0] + cube_length, start_corner[1] + cube_length, start_corner[2] + cube_length),
            (start_corner[0], start_corner[1] + cube_length, start_corner[2] + cube_length)
        ]
        print("corners: ", corners)

        # Calculate the distance from the position to each corner
        distances = [np.linalg.norm(position - np.array(corner)) for corner in corners]
        
        # Find the index of the minimum distance
        min_index = np.argmin(distances)
        print("distances: ", distances)
        # print("min_index: ", min_index)
        # print("corners[min_index]: ", corners[min_index])
        # Return the corner with the minimum distance
        # return corners[min_index]
        return list(np.array(corners[min_index]) / k)

    def read_pose_joints(self, pose): #从文件读取pose到关节的映射
        with open(self.pose_to_joints_map_file_path, 'r') as f:
            data = json.load(f)
            return data[str(pose)]
    
    def read_joints_pose(self, joints): #从文件读取关节到pose的映射
        with open(self.joints_poses_map_file_path, 'r') as f:
            data = json.load(f)
            return data[str(joints)]




# pose = [0.018, 0.018, 0.018]
# pose = [0.008, 0.008, 0.008]
pose = [0.00, -0.231, 0.176, 0.71, 0, 0, 0.71]

pose_space = So100_pose_space()
pose_space.construct_pose_space()
pose_space.save_data()

# closest_corner_position = pose_space.get_positon_closest_corner_position(pose)
# print("closest_corner_position: ", closest_corner_position)
# closest_corner_pose = closest_corner_position + [0.71, 0, 0, 0.71]
# print("closest_corner_pose: ", closest_corner_pose)
# joints = pose_space.read_pose_joints(closest_corner_pose)
# print("joints: ", joints)


#     def store_joints_for_number(self, number, values):
#         """
#         Store a list of five numbers for a given mapped number.
#         """
#         if len(values) != 5:
#             raise ValueError("The list must contain exactly five numbers.")
#         self.number_to_list_map[number] = values

#     def get_joints_for_number(self, number):
#         return self.number_to_list_map.get(number, None)

#     def store_joints_for_pose(self, pose, values):
#         if len(values) != 5:
#             raise ValueError("The list must contain exactly five numbers.")
#         self.pose_to_joints_map[pose] = values

#     def get_joints_for_pose(self, pose):
#         return self.pose_to_joints_map.get(pose, None)



# def load_poses_joints_map(file_path):
#     with open(file_path, 'r') as f:
#         # Load the JSON data
#         data = json.load(f)
#         # Convert string keys back to tuples
#         deserialized_map = {eval(k): v for k, v in data.items()}
#     return deserialized_map

# def get_closest_position_in_cube(position):
#     # Convert position to a numpy array for vectorized operations
#     position = np.array(position)
    
#     # Determine the starting corner of the cube that contains the position
#     start_corner = np.floor(position).astype(int)
    
#     # Define the 8 corners of the cube based on the starting corner
#     corners = [
#         (start_corner[0], start_corner[1], start_corner[2]),
#         (start_corner[0] + 1, start_corner[1], start_corner[2]),
#         (start_corner[0] + 1, start_corner[1] + 1, start_corner[2]),
#         (start_corner[0], start_corner[1] + 1, start_corner[2]),
#         (start_corner[0], start_corner[1], start_corner[2] + 1),
#         (start_corner[0] + 1, start_corner[1], start_corner[2] + 1),
#         (start_corner[0] + 1, start_corner[1] + 1, start_corner[2] + 1),
#         (start_corner[0], start_corner[1] + 1, start_corner[2] + 1)
#     ]
#     print("position: ", position)
#     print("corners: ", corners)

#     # Calculate the distance from the position to each corner
#     distances = [np.linalg.norm(position - np.array(corner)) for corner in corners]
    
#     # Find the index of the minimum distance
#     min_index = np.argmin(distances)
#     print("distances: ", distances)
#     print("min_index: ", min_index)
#     print("corners[min_index]: ", corners[min_index])
#     # Return the corner with the minimum distance
#     return corners[min_index]

# # 示例用法
# file_path = "pose_to_joints_map.json"
# pose_to_joints_map = load_poses_joints_map(file_path)
# # joints = pose_to_joints_map[0.0, -0.2, 0.17, 0.71, 0, 0, 0.71]
# # joints = pose_to_joints_map[0.220, -0.15, 0.0, 0.71, 0, 0, 0.71]
# # joints = pose_to_joints_map[0, 0, 0, 0, 0, 0, 0]
# joints = pose_to_joints_map[0.22, -0.15, 0.24, 0.71, -0, 0, 0.71]
# print(joints)

# # # 示例用法
# pose_space = So100_pose_space()

# time_start = time.time()
# joints_poses_map = pose_space._iterate_joints()
# print("joints_poses_map length: ", len(joints_poses_map))
# time_end = time.time()
# print("time cost: ", time_end - time_start, "s")
# pose_space.save_joints_poses_map("joints_poses_map.json")

# time_start = time.time()
# pose_to_joints_map = pose_space._iterate_positions()
# print("pose_to_joints_map length: ", len(pose_to_joints_map))
# time_end = time.time()
# print("time cost: ", time_end - time_start, "s")
# pose_space.save_poses_joints_map("pose_to_joints_map.json")

# get_closest_position_in_cube([1.8, 0.1, 0.2])