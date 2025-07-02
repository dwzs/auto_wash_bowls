## so100机械结构：
臂展：36cm：  11 + 15 + 10 


## git wash bowls：
dwzs_token: ********
用http方式推代码，避免ssh遇到的问题：
	1. 设置http url： git remote set-url origin https://github.com/dwzs/auto_wash_bowls.git
	2. 生成github token
	3. 推送代码：（注意输入密码时输入token）
	4. 永久保存token： git config --global credential.helper store

启动仿真机械臂： pyisim so100_sim_ros_launch.py
启动rviz可视化以及gui关节控制： ros2 launch so100_robot_description display_true_robot.launch.py


## 启动步骤
1. 启动so100驱动层(关节控制，关节限制)
    python so100_driver.py
2. 启动rviz可视化以及gui关节控制
    ros2 launch so100_robot_description display_true_robot.launch.py
3. 启动so100 控制层(正逆解，机械臂控制，夹爪控制)
    python so100_controller.py
4. 启动isaacsim 仿真
    pyisim so100_sim_ros_launch.py
    
## 思路：
1. 目的：
    a. so100, 单目（不标定），抓取立方体
2. 目前能做什么(手里有哪些工具)，规则是什么：
    1.机械臂：
        a. 关节控制 
        b. 获取关节角度
    2.图像：
        a. 图像像素级处理
    3.其它：
        数学基本运算
3. 与目标的差距：
    目标：抓取立方体，自动选择抓取点
    差距：位置差距与姿态差距
4. 减少差距：







EllipseExtracter
    get_original_image()
    get_gray_image()
    get_edge_image()
    get_edge_filtered_image()
    get_rgb_ellipses_image()
    get_edge_ellipses_image()
    get_ellipse()
    process_image()


to do list:
    wash bowls:
        找碗
        放碗
            关节控制
            笛卡尔控制（绝对位置，相对位置，旋转）
            视觉反馈控制

            遥操作
        洗碗





机械臂控制逻辑：
粗控：
    记忆空间

    构建空间图（笛卡尔空间（极坐标、直角坐标，x y z rx rz）-》关节空间）
    


精控：
    关节空间

1 cm  
10度
x y z
rx ry rz
-30 30
61 * 61 * 61


a1 a2 a3 a4 a5

实现下面功能：
矩形三维空间中，对每个xyz坐标映射出一个数。
其中三维空间长宽高分别为60cm，60cm，60cm。xyz分辨率均为1cm。
x方向，从左到右，从0到59。
y方向，从后到前，从0到59。
z方向，从下到上，从0到59。
当xyz坐标为(0,0,0)时，映射出的数为0。
当原点往x方向移动1cm时，映射出的数为1，以此递增。
当原点往y方向移动1cm时，映射出的数为60，以此递增。
当原点往z方向移动1cm时，映射出的数为3600，以此递增。

基于上面需求封装接口。
输入xyz坐标，输出映射出的数。
输入映射出的数，输出xyz坐标。






实现下面功能， 
构建位姿空间，
遍历joints_poses_map 中的每一条数据，根据



修改：
tcp flange 合并成一个函数，默认tcp，参数决定。



id:
1: -1.7 1.7
2: -1.8 1.7
3: -1.7 1.6
4: -3.5 0.2
5: -1.8 1.8
6: -0.18 1.5







1. 代码结构优化。
2. 添加仿真模式。
3. urdf 添加关节limits， ik是否会自动考虑这些limits。

FT_server:
    存放飞特舵机sdk

resources：
    urdf

sense：
    提取椭圆：
        ellipses_extractor.py # 负责图像处理，提取出椭圆位置
        ellipses_publisher.py # 通过topic /sense/ellipses[/vision_msgs/Detection2DArray] 发布多个椭圆以及对应的置信度

    提取夹爪:
        gripper_extractor.py # 负责图像处理，提取出夹爪位置
        gripper_publisher.py # 通过topic /sense/grippers[/vision_msgs/Detection2DArray]发布多个夹爪以及对应的置信度

    config.json

control：
    so100_driver.py # 与舵机内单片机通过串口通信。
    so100_driver_node.py # 对 so100_driver接口包装，通过ros与外部数据通信
        pub:/so100/state/joints[sensor_msgs/JointState]
        sub:/so100/cmd/joints[sensor_msgs/JointState] 

    arm_controller.py # 负责ik,fk，以及机械臂常用控制接口定义，与ros无关。
    arm_controller_interfaces.py # 对arm_controller接口包装，通过ros通信与so100_driver_node交互。
        pub: /so100/cmd/joints[sensor_msgs/JointState] 
        sub: /so100/state/joints[sensor_msgs/JointState]

        flange_move_to_pose(pose) -> bool # pose( list[x, y, z, w, rx, ry, rz, w] )
        flange_get_pose() -> list[x, y, z, w, rx, ry, rz, w]

        tcp_move_to_pose(self, pose, wait=True, timeout=20.0, joint_tolerance=0.01) -> bool
        tcp_get_pose -> list[x, y, z, w, rx, ry, rz, w]

        get_joints() -> list[j1, j2, j3, j4, j5]
        set_joints(joints)-> bool

    
    gripper_controller.py  # 处理gripper 与so00见的关系，如索引对应关系，定义基本接口。
        pub: /driver/cmd/joints[sensor_msgs/JointState] 
        sub: /driver/state/joints[sensor_msgs/JointState]

        open() -> bool
        close() -> bool
        set_joint() -> bool
        get_joint() -> float
        set_opening_percentage(percent) -> bool
        get_opening_percentage() -> float

    so100_joints_control_gui_launch.py

    config.json

simulation：
    so100_rviz_launch.py # 启动rviz，加载so100urdf。


全局配置参数与局部参数
比如log，希望全局配置。

通信层，应用层，算法层，是不是应该分开。层与层通过接口通信。

通信层：数据收发，应用层接口封装
应用层：逻辑，
算法层：算法。


