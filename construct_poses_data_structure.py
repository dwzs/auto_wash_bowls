class So100_pose_space:
    def __init__(self, length=60, width=60, height=60):
        self.length = length
        self.width = width
        self.height = height
        self.number_to_list_map = {}  # Dictionary to store lists for each number

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

    def store_list_for_number(self, number, values):
        """
        Store a list of five numbers for a given mapped number.
        """
        if len(values) != 5:
            raise ValueError("The list must contain exactly five numbers.")
        self.number_to_list_map[number] = values

    def get_list_for_number(self, number):
        """
        Retrieve the list of five numbers for a given mapped number.
        """
        return self.number_to_list_map.get(number, None)

# 示例用法
pose_space = So100_pose_space()

# 将xyz坐标映射为数值
number = pose_space.xyz_to_number(1, 0, 0)
print(f"坐标 (1, 0, 0) 映射为数值: {number}")

# 存储一个5个数的list
pose_space.store_list_for_number(number, [10, 20, 30, 40, 50])

# 获取存储的list
stored_list = pose_space.get_list_for_number(number)
print(f"数值 {number} 存储的list: {stored_list}")

# 将数值映射为xyz坐标
x, y, z = pose_space.number_to_xyz(25564)
print(f"数值 25564 映射为坐标: ({x}, {y}, {z})")

