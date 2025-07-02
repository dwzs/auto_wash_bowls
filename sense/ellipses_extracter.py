import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional, Any

class EllipseExtracter:
    """
    椭圆提取器 - 从图像中检测和提取椭圆
    """
    
    #---------------------------初始化部分---------------------------
    def __init__(self, 
                 min_points: int = 10,
                 ellipse_params: Optional[Dict[str, Any]] = None,
                 confidence_threshold: float = 0.6
                 ):
        """
        初始化椭圆提取器
        
        参数:
            min_points: 有效点集的最小点数 (默认: 10)
            ellipse_params: 椭圆拟合参数字典
            confidence_threshold: 检测椭圆的接受阈值 (默认: 0.6)
        """
        # 基本参数
        self.min_points = min_points
        self.blur_kernel_size = (5, 5)
        self.canny_threshold1 = 50
        self.canny_threshold2 = 150
        self.confidence_threshold = confidence_threshold
        
        # 设置椭圆参数
        self._setup_ellipse_params(ellipse_params)
        
        # 初始化结果存储
        self._init_storage()
    
    def _setup_ellipse_params(self, ellipse_params: Optional[Dict[str, Any]]) -> None:
        """设置椭圆拟合参数"""
        default_ellipse_params = {
            'num_samples': 10,  # 椭圆拟合的随机样本数
            'sample_size': 5,   # 每个样本的点数（拟合椭圆至少需要5个点）
            'tolerance': 3      # 判断点是否在椭圆上的容差
        }
        
        self.ellipse_params = default_ellipse_params
        if ellipse_params:
            self.ellipse_params.update(ellipse_params)
    
    def _init_storage(self) -> None:
        """初始化结果存储变量"""
        # 图像存储
        self.original_image = None
        self.gray_image = None
        self.edge_image = None

        # 结果存储
        self.point_sets = []
        self.ellipses = []
        self.all_candidates = []
        self.confidences = []
    


    #---------------------------点集提取相关方法---------------------------
    def _find_continuous_points(self, edge_image: np.ndarray, min_points: Optional[int] = None) -> List[np.ndarray]:
        """查找边缘图像中的所有连续点集"""
        if min_points is None:
            min_points = self.min_points

        # 使用OpenCV的连通分量分析
        num_labels, labels = cv2.connectedComponents(edge_image)
        
        # 提取连通分量的更高效方法
        point_sets = []
        
        # 一次性查找所有非背景点坐标
        y_all, x_all = np.nonzero(labels)
        label_values = labels[y_all, x_all]
        
        # 统计每个标签的点数
        label_counts = self._count_points_per_label(label_values)
        
        # 过滤点数少于min_points的标签
        valid_labels = [label for label, count in label_counts.items() if count >= min_points]
        
        # 仅为有效标签创建点集
        for label in valid_labels:
            point_set = self._create_point_set_for_label(label, label_values, x_all, y_all)
            point_sets.append(point_set)
        
        return point_sets
    
    def _count_points_per_label(self, label_values: np.ndarray) -> Dict[int, int]:
        """计算每个标签的点数"""
        label_counts = {}
        for label_val in label_values:
            if label_val > 0:  # 跳过背景
                label_counts[label_val] = label_counts.get(label_val, 0) + 1
        return label_counts
    
    def _create_point_set_for_label(self, label: int, label_values: np.ndarray, 
                                    x_all: np.ndarray, y_all: np.ndarray) -> np.ndarray:
        """为指定标签创建点集"""
        indices = np.where(label_values == label)[0]
        x_coords = x_all[indices]
        y_coords = y_all[indices]
        return np.column_stack((x_coords, y_coords))
    
    #---------------------------椭圆拟合相关方法---------------------------
    def _fit_ellipses(self, point_sets: Optional[List[np.ndarray]] = None) -> Tuple[List, List, List]:
        """拟合椭圆到点集(使用随机采样提高鲁棒性)"""
        if point_sets is None:
            point_sets = self.point_sets
            
        ellipses = []
        all_candidates = []
        confidences = []
        
        # 合并所有点集进行置信度计算
        all_points = np.vstack(point_sets) if point_sets else np.array([])
        
        for point_set in point_sets:
            # 为OpenCV转换为轮廓格式
            contour = point_set.reshape(-1, 1, 2).astype(np.int32)
            
            # 拟合椭圆需要至少5个点
            if len(contour) >= self.ellipse_params['sample_size']:
                # 获取候选椭圆
                candidate_ellipses = self._get_candidate_ellipses(contour)
                
                # 如果有候选椭圆，找出最佳的一个
                if candidate_ellipses:
                    best_ellipse, best_confidence = self._find_best_ellipse(candidate_ellipses, all_points)
                    
                    ellipses.append(best_ellipse)
                    confidences.append(best_confidence)
                    all_candidates.append(candidate_ellipses)
        
        return ellipses, all_candidates, confidences
    
    def _get_candidate_ellipses(self, contour: np.ndarray) -> List:
        """获取候选椭圆"""
        candidate_ellipses = []
        
        # 预先计算所有采样参数
        num_samples = self.ellipse_params['num_samples']
        sample_size = self.ellipse_params['sample_size']
        
        # 只尝试有效的随机样本
        valid_samples = 0
        max_attempts = num_samples * 2  # 设置最大尝试次数
        attempts = 0
        
        while valid_samples < num_samples and attempts < max_attempts:
            try:
                # 随机样本点
                indices = np.random.choice(len(contour), sample_size, replace=False)
                sample_contour = contour[indices]
                
                # 拟合椭圆
                ellipse = cv2.fitEllipse(sample_contour)
                candidate_ellipses.append(ellipse)
                valid_samples += 1
            except:
                pass
            attempts += 1
        
        return candidate_ellipses
    
    def _find_best_ellipse(self, candidate_ellipses: List, all_points: np.ndarray) -> Tuple:
        """从候选椭圆中找出最佳椭圆"""
        # 计算每个椭圆上或附近的点和置信度
        point_counts = np.zeros(len(candidate_ellipses))
        point_confidences = np.zeros(len(candidate_ellipses))
        
        # 批量评估所有椭圆
        for i, ellipse in enumerate(candidate_ellipses):
            count, confidence = self._count_points_near_ellipse(ellipse, all_points)
            point_counts[i] = count
            point_confidences[i] = confidence

        # 找到点最多的椭圆
        best_idx = np.argmax(point_counts)
        best_ellipse = candidate_ellipses[best_idx]
        best_confidence = point_confidences[best_idx]
        
        return best_ellipse, best_confidence
    
    #---------------------------椭圆评估相关方法---------------------------
    def _count_points_near_ellipse(self, ellipse: Tuple, points: np.ndarray) -> Tuple[int, float]:
        """计算有多少点在椭圆上或附近并返回置信度"""
        
        center, axes, angle = ellipse
        center = np.array(center)
        a, b = axes[0] / 2, axes[1] / 2  # 半长轴和半短轴
        angle_rad = np.deg2rad(angle)  # 将角度转换为弧度
        
        # 旋转矩阵
        rotation_matrix = self._get_rotation_matrix(angle_rad)
        
        # 向量化实现 - 一次性计算所有点
        # 以椭圆中心为中心的点
        points_centered = points - center
        
        # 一次性对所有点应用旋转
        points_rotated = np.dot(points_centered, rotation_matrix.T)
        
        # 为所有点计算标准化距离
        x_values = points_rotated[:, 0]
        y_values = points_rotated[:, 1]
        
        # 防止除以零
        if a == 0 or b == 0:
            return 0, 0
        
        # 一次性计算所有点的椭圆方程值
        values = (x_values**2 / a**2) + (y_values**2 / b**2)
        
        # 计算到椭圆的距离
        distances = np.abs(values - 1) * min(a, b)
        
        # 计算容差内的点数
        count = np.sum(distances <= self.ellipse_params['tolerance'])
        
        # 计算置信度
        confidence = self._calculate_confidence(count, a, b)
        
        return count, confidence
    
    def _get_rotation_matrix(self, angle_rad: float) -> np.ndarray:
        """获取旋转矩阵"""
        cos_angle = np.cos(angle_rad)
        sin_angle = np.sin(angle_rad)
        return np.array([
            [cos_angle, sin_angle],
            [-sin_angle, cos_angle]
        ])
    
    def _calculate_confidence(self, count: int, a: float, b: float) -> float:
        """计算椭圆的置信度分数"""
        # 计算椭圆周长上的预期点数
        h = ((a - b) ** 2) / ((a + b) ** 2)
        perimeter = np.pi * (a + b) * (1 + (3 * h) / (10 + np.sqrt(4 - 3 * h)))
        expected_points = np.ceil(perimeter)
        
        # 计算置信度
        confidence = min(1.0, count / expected_points) if expected_points > 0 else 0
        return confidence
    
    #---------------------------椭圆过滤相关方法---------------------------
    def _filter_ellipses(self, ellipses: List) -> List[int]:
        """过滤椭圆，删除不符合条件的椭圆"""
        if not ellipses:
            return []
            
        # 从原始图像获取尺寸
        if self.original_image is not None:
            image_height, image_width = self.original_image.shape[:2]
        else:
            return list(range(len(ellipses)))
        
        # 定义过滤阈值
        max_axis_length = image_width / 2  # 长轴不能超过图像宽度的一半
        min_axis_length = image_width / 10  # 长轴不能小于图像宽度的十分之一
        max_axis_ratio = 2.0  # 长轴与短轴的最大比率(圆度)
        
        filtered_indices = []
        
        for i, ellipse in enumerate(ellipses):
            if self._is_valid_ellipse(ellipse, min_axis_length, max_axis_length, max_axis_ratio):
                filtered_indices.append(i)
        
        return filtered_indices
    
    def _is_valid_ellipse(self, ellipse: Tuple, min_axis_length: float, 
                          max_axis_length: float, max_axis_ratio: float) -> bool:
        """检查椭圆是否有效"""
        center, axes, angle = ellipse
        major_axis, minor_axis = max(axes), min(axes)
        
        # 计算长轴与短轴的比率(圆度)
        axis_ratio = major_axis / minor_axis if minor_axis > 0 else float('inf')
        
        # 检查条件
        if (min_axis_length <= major_axis <= max_axis_length and 
            axis_ratio <= max_axis_ratio):
            return True
        return False



    #---------------------------公共API部分---------------------------
    def get_ellipse(self) -> Tuple[List, List]:
        """
        获取按置信度排序的拟合椭圆及其置信度分数
        
        返回:
            sorted_ellipses: 按置信度排序的椭圆列表
            sorted_confidences: 对应的置信度值列表
        """
        if not self.ellipses or not self.confidences:
            return [], []
            
        # 将椭圆和置信度组合
        ellipse_with_conf = list(zip(self.ellipses, self.confidences))
        
        # 按置信度降序排序
        ellipse_with_conf.sort(key=lambda x: x[1], reverse=True)
        
        # 分离排序后的椭圆和置信度
        sorted_ellipses, sorted_confidences = zip(*ellipse_with_conf) if ellipse_with_conf else ([], [])
        
        return sorted_ellipses, sorted_confidences
    
    def get_edge_ellipses_image(self) -> np.ndarray:
        """返回边缘检测后的椭圆图像"""
        if self.edge_image is None:
            return np.zeros((100, 100, 3), dtype=np.uint8)
            
        # 创建彩色图像 (将边缘图像转换为3通道)
        edge_ellipses_image = cv2.cvtColor(self.edge_image, cv2.COLOR_GRAY2BGR)
        
        # 绘制所有检测到的椭圆
        for i, (ellipse, confidence) in enumerate(zip(self.ellipses, self.confidences)):
            # 仅绘制置信度超过阈值的椭圆
            if confidence >= self.confidence_threshold:
                # 使用纯绿色
                color = (0, 255, 0)  # 纯绿色 (BGR格式)
                
                # 绘制椭圆
                cv2.ellipse(edge_ellipses_image, ellipse, color, 2)
                
                # 添加置信度标签到椭圆中心
                center = tuple(map(int, ellipse[0]))
                label = f"{confidence:.2f}"
                cv2.putText(edge_ellipses_image, label, 
                           center,  # 放在椭圆中心
                           cv2.FONT_HERSHEY_SIMPLEX, 2, color, 2)
        
        return edge_ellipses_image

    # 获取图像的公开方法
    def get_original_image(self) -> np.ndarray:
        """返回原始输入图像"""
        return self.original_image
    
    def get_original_ellipses_image(self) -> np.ndarray:
        """返回原始图像上标记椭圆的版本"""
        if self.original_image is None:
            return np.zeros((100, 100, 3), dtype=np.uint8)
            
        # 创建原始图像的副本
        original_with_ellipses = self.original_image.copy()
        
        # 绘制所有检测到的椭圆
        for i, (ellipse, confidence) in enumerate(zip(self.ellipses, self.confidences)):
            # 仅绘制置信度超过阈值的椭圆
            if confidence >= self.confidence_threshold:
                # 使用纯绿色
                color = (0, 255, 0)  # 纯绿色 (BGR格式)
                
                # 绘制椭圆
                cv2.ellipse(original_with_ellipses, ellipse, color, 2)
                
                # 添加置信度标签到椭圆中心
                center = tuple(map(int, ellipse[0]))
                label = f"{confidence:.2f}"
                cv2.putText(original_with_ellipses, label, 
                           center,  # 放在椭圆中心
                           cv2.FONT_HERSHEY_SIMPLEX, 2, color, 2)
        
        # # 转换为RGB颜色空间以便正确显示
        # rgb_image = cv2.cvtColor(original_with_ellipses, cv2.COLOR_BGR2RGB)
        
        return original_with_ellipses
    
    def get_gray_image(self) -> np.ndarray:
        """返回图像的灰度版本"""
        return self.gray_image
    
    def get_edge_image(self) -> np.ndarray:
        """返回边缘检测后的图像"""
        return self.edge_image

    def get_point_sets(self) -> List[np.ndarray]:
        """返回所有点集"""
        return self.point_sets

    def get_point_sets_image(self) -> np.ndarray:
        """返回所有点集的图像，每个点集用不同颜色区分"""
        if self.edge_image is None:
            return np.zeros((100, 100, 3), dtype=np.uint8)
            
        # 创建彩色图像 (3通道)
        point_sets_image = np.zeros((self.edge_image.shape[0], self.edge_image.shape[1], 3), dtype=np.uint8)
        
        # 为每个点集生成不同的颜色
        num_point_sets = len(self.point_sets)
        if num_point_sets > 0:
            # 使用HSV色彩空间均匀分布色调，保持饱和度和亮度固定
            colors = []
            for i in range(num_point_sets):
                # 色调在0-179范围内均匀分布 (OpenCV中的H范围)
                hue = int(179 * i / num_point_sets)
                # 转换HSV到BGR (使用完全饱和和较高亮度)
                rgb_color = cv2.cvtColor(np.uint8([[[hue, 255, 255]]]), cv2.COLOR_HSV2BGR)[0, 0]
                colors.append((int(rgb_color[0]), int(rgb_color[1]), int(rgb_color[2])))
            
            # 绘制每个点集，使用不同的颜色
            for i, point_set in enumerate(self.point_sets):
                color = colors[i % len(colors)]
                for point in point_set:
                    x, y = point
                    if 0 <= y < point_sets_image.shape[0] and 0 <= x < point_sets_image.shape[1]:
                        point_sets_image[y, x] = color
        
        return point_sets_image

    def get_all_candidates(self) -> List[List]:
        """返回所有候选椭圆"""
        return self.all_candidates

    def process_frame(self, frame: np.ndarray, min_points: Optional[int] = None) -> Tuple[List, List]:
        """
        处理视频帧以检测椭圆
        
        参数:
            frame: 输入图像帧
            min_points: 有效点集的最小点数
        
        返回:
            ellipses: 检测到的椭圆
            confidences: 每个椭圆的置信度值
        """
        if min_points is None:
            min_points = self.min_points
            
        # 存储原始图像
        self.original_image = frame
        
        # 转换为灰度并进行边缘检测
        self.gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(self.gray_image, self.blur_kernel_size, 0)
        self.edge_image = cv2.Canny(blurred, self.canny_threshold1, self.canny_threshold2)
        
        # 查找点集并拟合椭圆
        self.point_sets = self._find_continuous_points(self.edge_image, min_points)
        self.ellipses, self.all_candidates, self.confidences = self._fit_ellipses(self.point_sets)
        
        # 过滤椭圆
        filtered_indices = self._filter_ellipses(self.ellipses)
        self.ellipses = [self.ellipses[i] for i in filtered_indices]
        self.confidences = [self.confidences[i] for i in filtered_indices]
        
        return self.ellipses, self.confidences
    
    def process_image(self, image_path: str, min_points: Optional[int] = None) -> Tuple[List, List]:
        """
        处理图像以检测椭圆
        
        参数:
            image_path: 输入图像的路径
            min_points: 有效点集的最小点数
        
        返回:
            ellipses: 检测到的椭圆
            confidences: 每个椭圆的置信度值
        """
        # 加载图像
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"无法从{image_path}加载图像")
            
        # 调用process_frame处理加载的图像
        return self.process_frame(image, min_points)
    

if __name__ == "__main__":
    image_path = "./resources/bowls2.png"
    
    # 创建椭圆提取器实例
    extractor = EllipseExtracter(
        min_points=500,
        ellipse_params={
            'num_samples': 20,
            'sample_size': 5,
            'tolerance': 2
        },
        confidence_threshold=0.2
    )
    
    # 处理图像
    ellipses, confidences = extractor.process_image(image_path)
    
    # 获取按置信度排序的结果
    sorted_ellipses, sorted_confidences = extractor.get_ellipse()
    
    print("\n===== 椭圆检测结果 =====")
    if sorted_ellipses:
        print(f"检测到 {len(sorted_ellipses)} 个椭圆\n")
        print(f"{'编号':<4}{'中心 (x, y)':<20}{'轴长 (长轴, 短轴)':<25}{'角度':<10}{'置信度':<10}")
        print("-" * 70)
        
        for i, (ellipse, confidence) in enumerate(zip(sorted_ellipses, sorted_confidences)):
            center, axes, angle = ellipse
            print(f"{i+1:<4}({center[0]:.1f}, {center[1]:.1f}){'':>5}({axes[0]:.1f}, {axes[1]:.1f}){'':>10}{angle:.1f}°{'':>5}{confidence:.3f}")
    else:
        print("未检测到椭圆")
    

