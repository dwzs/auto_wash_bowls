so100机械结构：
臂展：36cm：  11 + 15 + 10 


git wash bowls：
dwzs_token: ********
用http方式推代码，避免ssh遇到的问题：
	1. 设置http url： git remote set-url origin https://github.com/dwzs/auto_wash_bowls.git
	2. 生成github token
	3. 推送代码：（注意输入密码时输入token）
	4. 永久保存token： git config --global credential.helper store

启动仿真机械臂：
pyisim so100_sim_ros_launch.py


实现下面功能：
1. 构建一个600 * 600 的二值图，背景为黑
2. 在这个二值图中构建一条水平方向的20像素长度单像素宽度的直线。
3. 在这个二值图中构建一条斜着45度的20像素长度单像素宽度的直线。
4. 在这个二值图中构建一个半径为10像素的填充圆形。
5. 在这个二值图中构建一个半径为10像素的空心圆形，边界为单像素。
6. 构建类似上面4和5的矩形。



这个函数改一下，改成画下面图片：
1. 灰度图
2. 边缘图
3. 过滤后的点集图（彩色，用于区分不同点集）
4. 每组点集的所有椭圆，包括候选椭圆画在"3" 上
5. 每组点集的最高置信度椭圆画在“3” 上
6. 每组点集的最高置信度椭圆画在“2” 上
7. 高于阈值置信度的椭圆画在原始图上


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
