import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
import os

class EllipseExtract:
    def __init__(self, 
                 min_points=10,
                 ellipse_params=None,
                 confidence_threshold=0.6
                 ):
        """
        初始化碗口分割器
        
        Parameters:
            min_points: 连续点集最小点数，少于该数量的点集将被过滤 (default: 10)
            blur_kernel_size: 高斯模糊的内核大小 (default: (5, 5))
            canny_threshold1: Canny边缘检测的第一个阈值 (default: 50)
            canny_threshold2: Canny边缘检测的第二个阈值 (default: 150)
            ellipse_params: 椭圆参数字典，包含椭圆拟合的各种参数
            output_dir: 输出结果的目录 (default: "./results/")
            plot_figsize: 绘图的图像大小 (default: (15, 9))
        """
        self.min_points = min_points
        self.blur_kernel_size = (5, 5)
        self.canny_threshold1 = 50
        self.canny_threshold2 = 150
        self.output_dir = "./results/"
        self.plot_figsize = (15, 9)
        self.confidence_threshold = confidence_threshold
        # 设置椭圆参数
        default_ellipse_params = {
            'num_samples': 10,  # 随机采样次数，拟合椭圆时随机选取点集的次数
            'sample_size': 5,   # 每次采样的点数，拟合椭圆至少需要5个点
            'line_width': 2,    # 绘制椭圆时的线宽
            'color': (0, 255, 0),  # 绘制椭圆的颜色 (BGR格式，这里是绿色)
            'tolerance': 3      # 判断点是否在椭圆上的容差值（距离阈值）
        }
        
        self.ellipse_params = default_ellipse_params
        if ellipse_params:
            self.ellipse_params.update(ellipse_params)
        
        # 结果存储
        self.original_image = None
        self.gray_image = None
        self.edge_image = None
        self.edge_filtered_image = None
        self.rgb_ellipses_image = None
        self.edge_ellipses_image = None

        self.point_sets = []
        self.ellipses = []
        self.all_candidates = []
        self.confidences = []
        
        
    def find_continuous_points(self, edge_image, min_points=None):
        """在边缘图像中找到所有连续的点集"""
        if min_points is None:
            min_points = self.min_points

        # 使用OpenCV的连通组件分析
        num_labels, labels = cv2.connectedComponents(edge_image)
        
        # 使用更高效的方法提取连通组件点
        point_sets = []
        
        # 一次性找出所有非背景点坐标
        y_all, x_all = np.nonzero(labels)
        label_values = labels[y_all, x_all]
        
        # 用字典计数每个标签的点数
        label_counts = {}
        for label_val in label_values:
            if label_val > 0:  # 跳过背景
                label_counts[label_val] = label_counts.get(label_val, 0) + 1
        
        # 过滤点数少于min_points的标签
        valid_labels = [label for label, count in label_counts.items() if count >= min_points]
        
        # 只为有效的标签创建点集
        for label in valid_labels:
            indices = np.where(label_values == label)[0]
            x_coords = x_all[indices]
            y_coords = y_all[indices]
            point_set = np.column_stack((x_coords, y_coords))
            point_sets.append(point_set)
        
        self.point_sets = point_sets
        return point_sets

    def count_points_near_ellipse(self, ellipse, points):
        """计算点集中有多少点位于椭圆上或附近，并返回置信度"""
        
        center, axes, angle = ellipse
        center = np.array(center)
        a, b = axes[0] / 2, axes[1] / 2  # 椭圆半长轴和半短轴
        angle_rad = np.deg2rad(angle)  # 角度转弧度
        
        # 旋转矩阵
        cos_angle = np.cos(angle_rad)
        sin_angle = np.sin(angle_rad)
        rotation_matrix = np.array([
            [cos_angle, sin_angle],
            [-sin_angle, cos_angle]
        ])
        
        # 向量化实现 - 一次性计算所有点
        # 将所有点移动到以椭圆中心为原点的坐标系
        points_centered = points - center
        
        # 使用矩阵乘法一次性对所有点应用旋转
        points_rotated = np.dot(points_centered, rotation_matrix.T)
        
        # 为所有点计算标准化距离
        x_values = points_rotated[:, 0]
        y_values = points_rotated[:, 1]
        
        # 防止除以零
        if a == 0 or b == 0:
            return 0, 0
        
        # 一次性计算所有点的椭圆方程值
        values = (x_values**2 / a**2) + (y_values**2 / b**2)
        
        # 计算所有点到椭圆的距离
        distances = np.abs(values - 1) * min(a, b)
        
        # 计算在容差范围内的点数
        count = np.sum(distances <= self.ellipse_params['tolerance'])
        
        # 计算椭圆周长上预期的点数
        h = ((a - b) ** 2) / ((a + b) ** 2)
        perimeter = np.pi * (a + b) * (1 + (3 * h) / (10 + np.sqrt(4 - 3 * h)))
        expected_points = np.ceil(perimeter)
        
        # 计算置信度
        confidence = min(1.0, count / expected_points) if expected_points > 0 else 0
        
        return count, confidence

    def fit_ellipses(self, point_sets=None):
        """拟合椭圆 (使用随机采样提高鲁棒性)"""
        if point_sets is None:
            point_sets = self.point_sets
            
        ellipses = []
        all_candidates = []
        confidences = []
        
        # 合并所有点集用于计算置信度
        all_points = np.vstack(point_sets) if point_sets else np.array([])
        
        for point_set in point_sets:
            # 转换为轮廓格式以兼容OpenCV函数
            contour = point_set.reshape(-1, 1, 2).astype(np.int32)
            
            # 拟合椭圆需要至少5个点
            if len(contour) >= self.ellipse_params['sample_size']:
                # 随机采样
                candidate_ellipses = []
                
                # 预先计算所有采样参数，而不是在循环中重复
                num_samples = self.ellipse_params['num_samples']
                sample_size = self.ellipse_params['sample_size']
                
                # 只尝试有效的随机采样
                valid_samples = 0
                max_attempts = num_samples * 2  # 设置最大尝试次数
                attempts = 0
                
                while valid_samples < num_samples and attempts < max_attempts:
                    try:
                        # 随机抽取样本点
                        indices = np.random.choice(len(contour), sample_size, replace=False)
                        sample_contour = contour[indices]
                        
                        # 拟合椭圆
                        ellipse = cv2.fitEllipse(sample_contour)
                        candidate_ellipses.append(ellipse)
                        valid_samples += 1
                    except:
                        pass
                    attempts += 1
                
                # 如果有候选椭圆，找出最匹配原始点集的一组
                if candidate_ellipses:
                    # 计算每个椭圆上或附近的点数及置信度
                    point_counts = np.zeros(len(candidate_ellipses))
                    point_confidences = np.zeros(len(candidate_ellipses))
                    
                    # 批量评估所有椭圆
                    for i, ellipse in enumerate(candidate_ellipses):
                        count, confidence = self.count_points_near_ellipse(ellipse, all_points)
                        point_counts[i] = count
                        point_confidences[i] = confidence

                    # 找出点数最多的椭圆
                    best_idx = np.argmax(point_counts)
                    best_ellipse = candidate_ellipses[best_idx]
                    best_confidence = point_confidences[best_idx]
                    
                    ellipses.append(best_ellipse)
                    confidences.append(best_confidence)
                    all_candidates.append(candidate_ellipses)
        
        self.ellipses = ellipses
        self.all_candidates = all_candidates
        self.confidences = confidences
        return ellipses, all_candidates, confidences

    def draw_shapes(self, confidence_threshold=None):
        """
        创建并返回多个可视化结果图像:
        1. 灰度图
        2. 边缘图
        3. 过滤后的点集图（彩色）
        4. 所有候选椭圆画在点集图上
        5. 每组点集的最佳椭圆画在点集图上
        6. 每组点集的最佳椭圆画在边缘图上
        7. 高于阈值置信度的椭圆画在原始图上
        """
        # 使用传入的阈值或默认值
        if confidence_threshold is None:
            confidence_threshold = self.confidence_threshold
            
        # 检查必要的图像是否存在
        if self.gray_image is None or self.edge_image is None or not hasattr(self, 'edge_filtered_color'):
            print("图像处理尚未完成，无法生成可视化结果")
            return None
            
        # 1. 灰度图 - 直接使用已有图像
        gray_img = self.gray_image.copy()
        
        # 2. 边缘图 - 直接使用已有图像
        edge_img = self.edge_image.copy()
        
        # 3. 彩色点集图 - 直接使用已有图像
        points_img = self.edge_filtered_color.copy()
        
        # 4. 所有候选椭圆画在点集图上
        candidates_img = self.edge_filtered_color.copy()
        if self.all_candidates:
            for candidates in self.all_candidates:
                for ellipse in candidates:
                    cv2.ellipse(candidates_img, ellipse, (255, 255, 255), 1)
        
        # 5. 每组点集最佳椭圆画在点集图上
        best_on_points_img = self.edge_filtered_color.copy()
        if self.ellipses:
            for ellipse in self.ellipses:
                cv2.ellipse(best_on_points_img, ellipse, (255, 255, 255), 2)
        
        # 6. 每组点集最佳椭圆画在边缘图上
        # 先将边缘图转换为彩色图以便于绘制彩色椭圆
        best_on_edge_img = cv2.cvtColor(edge_img, cv2.COLOR_GRAY2BGR)
        if self.ellipses:
            for ellipse in self.ellipses:
                cv2.ellipse(best_on_edge_img, ellipse, (0, 255, 0), 2)
        
        # 7. 高于阈值置信度的椭圆画在原始图上
        rgb_img = cv2.cvtColor(self.original_image.copy(), cv2.COLOR_BGR2RGB)
        if self.ellipses and self.confidences:
            for i, (ellipse, confidence) in enumerate(zip(self.ellipses, self.confidences)):
                if confidence >= confidence_threshold:
                    cv2.ellipse(rgb_img, ellipse, self.ellipse_params['color'], self.ellipse_params['line_width'])
                    # 显示置信度
                    center_x, center_y = int(ellipse[0][0]), int(ellipse[0][1])
                    cv2.putText(rgb_img, f"{confidence:.2f}", (center_x - 20, center_y), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        # 返回所有图像
        return {
            'gray': gray_img,
            'edge': edge_img,
            'points': points_img,
            'candidates': candidates_img,
            'best_on_points': best_on_points_img,
            'best_on_edge': best_on_edge_img,
            'rgb_with_ellipses': rgb_img
        }

    def filter_ellipses(self, ellipses):
        """筛选椭圆，去除不符合条件的椭圆"""
        if not ellipses:
            return []
            
        # 从原始图像获取尺寸
        if self.original_image is not None:
            image_height, image_width = self.original_image.shape[:2]
        else:
            return list(range(len(ellipses)))
        
        # 定义筛选阈值
        max_axis_length = image_width / 2  # 长轴不能超过图像宽度的一半
        min_axis_length = image_width / 10  # 长轴不能小于图像宽度的十分之一
        max_axis_ratio = 2.0  # 长轴与短轴的比例不能太大（定义圆度）
        
        filtered_indices = []
        
        for i, ellipse in enumerate(ellipses):
            center, axes, angle = ellipse
            major_axis, minor_axis = max(axes), min(axes)
            
            # 计算长短轴比例（圆度）
            axis_ratio = major_axis / minor_axis if minor_axis > 0 else float('inf')
            
            # 检查条件
            if (min_axis_length <= major_axis <= max_axis_length and 
                axis_ratio <= max_axis_ratio):
                filtered_indices.append(i)
        
        return filtered_indices



    def _create_filtered_edge_image(self):
        """创建过滤后的边缘图像，将不同点集用不同颜色表示"""
        # 创建彩色图像而不是灰度图像，以便更好地区分不同点集
        edge_filtered = np.zeros((self.gray_image.shape[0], self.gray_image.shape[1], 3), dtype=np.uint8)
        
        # 使用HSV色彩空间均匀分布颜色
        num_point_sets = max(1, len(self.point_sets))
        for i, point_set in enumerate(self.point_sets):
            # 计算当前点集的HSV颜色 (不同色相，饱和度和亮度固定)
            hue = int(180 * i / num_point_sets)  # Hue值在OpenCV中是0-180
            color_hsv = np.array([hue, 255, 255], dtype=np.uint8).reshape(1, 1, 3)
            color_bgr = cv2.cvtColor(color_hsv, cv2.COLOR_HSV2BGR)[0, 0]
            
            # 绘制当前点集
            for point in point_set:
                x, y = point
                edge_filtered[y, x] = color_bgr
        
        # 转换回灰度图以保持与现有代码兼容
        edge_filtered_gray = cv2.cvtColor(edge_filtered, cv2.COLOR_BGR2GRAY)
        
        # 额外存储彩色版本供显示使用
        self.edge_filtered_color = edge_filtered
        
        return edge_filtered_gray

    def _create_overlay_image(self):
        """创建彩色边缘叠加在原图上的图像"""
        if not hasattr(self, 'edge_filtered_color') or self.original_image is None:
            return
        
        # 复制原图，确保是BGR格式
        overlay_image = self.original_image.copy()
        
        # 创建一个掩码，标记边缘点的位置
        mask = np.any(self.edge_filtered_color > 0, axis=2).astype(np.uint8)
        
        # 在掩码处使用彩色边缘替换原图
        for y in range(overlay_image.shape[0]):
            for x in range(overlay_image.shape[1]):
                if mask[y, x]:
                    overlay_image[y, x] = self.edge_filtered_color[y, x]
        
        # 保存结果
        self.overlay_image = overlay_image

    def _display_results(self, images, min_points):
        """显示处理结果"""
        plt.figure(figsize=self.plot_figsize)
        
        plt.subplot(2, 4, 1)
        plt.title('Original Image')
        plt.imshow(cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB))
        
        plt.subplot(2, 4, 2)
        plt.title('Grayscale Image')
        plt.imshow(images['gray'], cmap='gray')
        
        plt.subplot(2, 4, 3)
        plt.title('Edge Detection')
        plt.imshow(images['edge'], cmap='gray')
        
        plt.subplot(2, 4, 4)
        plt.title(f'Filtered Points (≥{min_points})')
        plt.imshow(cv2.cvtColor(images['points'], cv2.COLOR_BGR2RGB))
        
        plt.subplot(2, 4, 5)
        plt.title('All Candidate Ellipses')
        plt.imshow(cv2.cvtColor(images['candidates'], cv2.COLOR_BGR2RGB))
        total_candidates = sum([len(candidates) for candidates in self.all_candidates])
        plt.text(10, 20, f'Candidates: {total_candidates}', color='white', backgroundcolor='black')
        
        plt.subplot(2, 4, 6)
        plt.title('Best Ellipses on Points')
        plt.imshow(cv2.cvtColor(images['best_on_points'], cv2.COLOR_BGR2RGB))
        plt.text(10, 20, f'Best: {len(self.ellipses)}', color='white', backgroundcolor='black')
        
        plt.subplot(2, 4, 7)
        plt.title('Best Ellipses on Edges')
        plt.imshow(images['best_on_edge'])
        
        plt.subplot(2, 4, 8)
        plt.title(f'Ellipses with Conf.≥{self.confidence_threshold:.2f}')
        plt.imshow(images['rgb_with_ellipses'])
        
        # Display confidence information
        if self.confidences:
            avg_confidence = sum(self.confidences) / len(self.confidences) if self.confidences else 0
            high_conf = sum(1 for c in self.confidences if c >= self.confidence_threshold)
            plt.text(10, 40, f'High Conf: {high_conf}/{len(self.confidences)}', color='blue', backgroundcolor='white')
            plt.text(10, 70, f'Avg Conf: {avg_confidence:.2f}', color='blue', backgroundcolor='white')

        plt.tight_layout()
        plt.show()

    def save_results(self, images):
        """保存处理结果到指定目录"""
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            
        # 保存结果图像
        cv2.imwrite(os.path.join(self.output_dir, '1_gray.png'), images['gray'])
        cv2.imwrite(os.path.join(self.output_dir, '2_edge.png'), images['edge'])
        cv2.imwrite(os.path.join(self.output_dir, '3_points_color.png'), images['points'])
        cv2.imwrite(os.path.join(self.output_dir, '4_all_candidates.png'), images['candidates'])
        cv2.imwrite(os.path.join(self.output_dir, '5_best_on_points.png'), images['best_on_points'])
        cv2.imwrite(os.path.join(self.output_dir, '6_best_on_edge.png'), images['best_on_edge'])
        cv2.imwrite(os.path.join(self.output_dir, '7_rgb_with_ellipses.png'), 
                     cv2.cvtColor(images['rgb_with_ellipses'], cv2.COLOR_RGB2BGR))
        
        # # 保存彩色边缘叠加图
        # if hasattr(self, 'overlay_image'):
        #     cv2.imwrite(os.path.join(self.output_dir, '8_overlay_image.png'), self.overlay_image)

    def get_ellipse(self):
        """
        获取拟合出来的椭圆及其置信度，按置信度由高到低排序
        
        Returns:
            sorted_ellipses: 排序后的椭圆列表
            sorted_confidences: 对应的置信度列表
        """
        if not self.ellipses or not self.confidences:
            return [], []
            
        # 将椭圆和置信度组合在一起
        ellipse_with_conf = list(zip(self.ellipses, self.confidences))
        
        # 按置信度降序排序
        ellipse_with_conf.sort(key=lambda x: x[1], reverse=True)
        
        # 分离排序后的椭圆和置信度
        sorted_ellipses, sorted_confidences = zip(*ellipse_with_conf) if ellipse_with_conf else ([], [])
        
        return sorted_ellipses, sorted_confidences


    def process_image(self, image_path, min_points=None, show_plot=True):
        """处理图像：灰度化、边缘检测、椭圆拟合"""
        if min_points is None:
            min_points = self.min_points
            
        # 创建字典保存每个步骤的执行时间
        step_times = {}
        total_start_time = time.time()
        
        # 1. 读取并预处理图像
        step_start = time.time()
        image = cv2.imread(image_path)
        if image is None:
            print(f"无法读取图像: {image_path}")
            return
        
        # 输出图像基本信息
        height, width = image.shape[:2]
        channels = 1 if len(image.shape) == 2 else image.shape[2]
        file_size = os.path.getsize(image_path) / 1024  # KB
        
        print("\n===== 图像基本信息 =====")
        print(f"图像路径: {image_path}")
        print(f"图像尺寸: {width}×{height} 像素")
        print(f"颜色通道: {channels}")
        print(f"数据类型: {image.dtype}")
        print(f"文件大小: {file_size:.2f} KB")
        print(f"像素总数: {width * height}")
        
        # 存储原始图像
        self.original_image = image
        step_times['1_load_image'] = time.time() - step_start
        
        # 2. 灰度与边缘处理
        step_start = time.time()
        self.gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(self.gray_image, self.blur_kernel_size, 0)
        edges = cv2.Canny(blurred, self.canny_threshold1, self.canny_threshold2)
        self.edge_image = edges
        step_times['2_edge_detection'] = time.time() - step_start

        # 3. 查找连续点集并拟合椭圆
        step_start = time.time()
        self.point_sets = self.find_continuous_points(edges, min_points)
        step_times['3_find_points'] = time.time() - step_start
        
        step_start = time.time()
        self.ellipses, self.all_candidates, self.confidences = self.fit_ellipses(self.point_sets)
        step_times['4_fit_ellipses'] = time.time() - step_start

        # 4. 筛选椭圆
        step_start = time.time()
        filtered_indices = self.filter_ellipses(self.ellipses)
        self.ellipses = [self.ellipses[i] for i in filtered_indices]
        self.confidences = [self.confidences[i] for i in filtered_indices]
        step_times['5_filter_ellipses'] = time.time() - step_start

        # 5. 创建过滤后的边缘图像
        step_start = time.time()
        self._create_filtered_edge_image()
        step_times['6_create_filtered_edge'] = time.time() - step_start

        # 6. 绘制结果
        step_start = time.time()
        result_images = self.draw_shapes(self.confidence_threshold)
        step_times['7_draw_shapes'] = time.time() - step_start
        
        # 7. 显示和保存结果
        step_start = time.time()
        if show_plot and result_images:
            self._display_results(result_images, min_points)
        
        if result_images:
            self.save_results(result_images)
        step_times['8_display_save_results'] = time.time() - step_start
        
        # 计算总执行时间
        total_time = time.time() - total_start_time
        step_times['total'] = total_time
        
        # 打印执行时间统计
        print("\n===== 处理步骤耗时统计 =====")
        print(f"{'步骤':<30}{'耗时(秒)':<15}{'占比':<10}")
        print("-" * 55)
        for step, duration in step_times.items():
            if step != 'total':
                percentage = (duration / total_time) * 100
                print(f"{step[2:].replace('_', ' '):<30}{duration:.6f}{'':>5}{percentage:>6.2f}%")
        print("-" * 55)
        print(f"{'总耗时':<30}{total_time:.6f}")
        
        return result_images if result_images else None


if __name__ == "__main__":
    # image_path = "./resources/bowls1.png"
    image_path = "./resources/bowls2.png"
    
    # 创建碗口分割器实例
    segmenter = EllipseExtract(
        min_points=500,
        ellipse_params={
            'num_samples': 20,
            'sample_size': 5,
            'line_width': 2,
            'color': (0, 255, 0),
            'tolerance': 2
        },
        confidence_threshold=0.2
    )
    
    # 处理图像
    start_time = time.time()
    segmenter.process_image(image_path, show_plot=False)
    # segmenter.process_image(image_path, show_plot=True)
    elapsed_time = time.time() - start_time
    print(f"处理图像耗时: {elapsed_time:.6f}秒")
    
    # 显示结果
    sorted_ellipses, sorted_confidences = segmenter.get_ellipse()
    
    print("\n===== 椭圆检测结果 =====")
    if sorted_ellipses:
        print(f"共检测到 {len(sorted_ellipses)} 个椭圆\n")
        print(f"{'序号':<4}{'中心点 (x, y)':<20}{'轴长 (长轴, 短轴)':<25}{'角度':<10}{'置信度':<10}")
        print("-" * 70)
        
        for i, (ellipse, confidence) in enumerate(zip(sorted_ellipses, sorted_confidences)):
            center, axes, angle = ellipse
            print(f"{i+1:<4}({center[0]:.1f}, {center[1]:.1f}){'':>5}({axes[0]:.1f}, {axes[1]:.1f}){'':>10}{angle:.1f}°{'':>5}{confidence:.3f}")
    else:
        print("未检测到椭圆")
    