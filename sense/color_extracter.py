import cv2
import numpy as np
import json
import os
from typing import List, Tuple, Optional

# red_ranges = [
#     [0, 50, 50], [10, 255, 255],      # 红色段1: 0-10
#     [170, 50, 50], [179, 255, 255]    # 红色段2: 170-179
# ]

# orange = [[11, 50, 50], [25, 255, 255]]    # 橙色: 11-25
# yellow = [[26, 50, 50], [35, 255, 255]]    # 黄色: 26-35
# green = [[36, 50, 50], [85, 255, 255]]     # 绿色: 36-85
# cyan = [[86, 50, 50], [100, 255, 255]]     # 青色: 86-100
# blue = [[101, 50, 50], [130, 255, 255]]    # 蓝色: 101-130
# purple = [[131, 50, 50], [160, 255, 255]]  # 紫色: 131-160
# magenta = [[161, 50, 50], [169, 255, 255]] # 品红: 161-169

# # 特殊颜色
# white = [[0, 0, 200], [179, 30, 255]]      # 白色: 低饱和度，高明度
# black = [[0, 0, 0], [179, 255, 50]]        # 黑色: 低明度
# gray = [[0, 0, 50], [179, 30, 200]]        # 灰色: 低饱和度，中明度


class ColorExtracter:
    """
    提供通用的边缘与颜色区域提取工具。
    """

    def __init__(self, min_area: int = 30, config_path: str = None):
        self._image: Optional[np.ndarray] = None
        
        # 读取配置文件
        if config_path is None:
            config_path = os.path.join(os.path.dirname(__file__), "config.json")
        
        self.config = {}
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                self.config = json.load(f)
        
        # 基本参数
        self.min_area = min_area          
        self._canny = (50, 150)           
        
        # 摄像头配置
        self.camera_id = self.config.get("camera_id", 0)
        camera_res = self.config.get("camera_resolution", {"width": 640, "height": 480})
        self.camera_width = camera_res.get("width", 640)
        self.camera_height = camera_res.get("height", 480)
        
        # 稳定性参数
        self.blur_kernel = 5              # 增加模糊程度
        self.morph_kernel = 5             # 增加形态学核大小
        self.morph_iterations = 2         # 形态学操作次数

    # ----------------------------------------------------------------------
    # 原图管理
    # ----------------------------------------------------------------------
    def set_original_image(self, image: np.ndarray) -> None:
        """保存原图到内部成员变量"""
        self._image = image

    def get_original_image(self) -> Optional[np.ndarray]:
        """获取已保存的原图"""
        return self._image
    
    def create_camera_capture(self):
        """根据配置创建摄像头对象"""
        cap = cv2.VideoCapture(self.camera_id)
        if cap.isOpened():
            # 设置分辨率
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.camera_width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.camera_height)
            
            # 设置稳定性参数
            cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
            cap.set(cv2.CAP_PROP_EXPOSURE, -6)
        return cap
    
    def print_config(self):
        """打印当前配置信息"""
        print("=== ColorExtracter 配置 ===")
        print(f"摄像头ID: {self.camera_id}")
        print(f"分辨率: {self.camera_width}x{self.camera_height}")
        print(f"模糊核大小: {self.blur_kernel}")
        print(f"形态学核大小: {self.morph_kernel}")
        print("========================")

    # ----------------------------------------------------------------------
    # 边缘相关
    # ----------------------------------------------------------------------
    def get_edage(self, image: np.ndarray = None) -> np.ndarray:
        """
        提取图像的边缘，并返回边缘点坐标
        返回边缘点，2xn 矩阵，每行表示一个点的 x,y 坐标
        """
        img = self._ensure_image(image)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        edges = cv2.Canny(gray, *self._canny)
        ys, xs = np.where(edges > 0)
        return np.vstack((xs, ys)).T          # (n,2)

    def get_edage_image(self, image: np.ndarray = None) -> np.ndarray:
        """增加边缘检测的稳定性"""
        img = self._ensure_image(image)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # 增强模糊以减少噪声
        gray = cv2.GaussianBlur(gray, (7, 7), 0)
        return cv2.Canny(gray, *self._canny)

    def get_edage_original_image(self, image: np.ndarray = None) -> np.ndarray:
        """
        提取图像的边缘，并返回边缘图像，背景为原始图像，物体轮廓为绿色
        """
        img = self._ensure_image(image).copy()
        edges = self.get_edage_image(img)
        img[edges > 0] = (0, 255, 0)
        return img

    # ----------------------------------------------------------------------
    # 颜色相关
    # ----------------------------------------------------------------------
    def get_color_objects(
        self,
        low_color: List[int],
        high_color: List[int],
        image: np.ndarray = None
    ) -> np.ndarray:
        """
        提取图像的颜色对象，并返回颜色对象点坐标
        返回颜色对象，2xn 矩阵，每行表示一个颜色点的 x,y 坐标
        """
        mask = self._color_mask(low_color, high_color, image)
        ys, xs = np.where(mask > 0)
        return np.vstack((xs, ys)).T

    def get_color_objects_edage(
        self,
        low_color: List[int],
        high_color: List[int],
        image: np.ndarray = None
    ) -> np.ndarray:
        """
        提取图像指定颜色轮廓，并返回轮廓点列
        返回边缘点，2xn 矩阵，每行表示一个轮廓点的 x,y 坐标
        """
        mask = self._color_mask(low_color, high_color, image)
        edges = cv2.Canny(mask, 50, 150)
        ys, xs = np.where(edges > 0)
        return np.vstack((xs, ys)).T

    def get_color_objects_image(
        self,
        low_color: List[int],
        high_color: List[int],
        image: np.ndarray = None
    ) -> np.ndarray:
        """
        提取图像的颜色对象，并返回颜色对象图像，背景为黑色，物体为原色
        """
        img = self._ensure_image(image)
        mask = self._color_mask(low_color, high_color, img)
        result = cv2.bitwise_and(img, img, mask=mask)
        return result

    def get_color_objects_edage_image(self,
        low_color: List[int],
        high_color: List[int],
        image: np.ndarray = None
    ) -> np.ndarray:
        """改进颜色边缘检测"""
        mask = self._color_mask(low_color, high_color, image)
        # 对掩码进行额外模糊以减少噪声
        mask = cv2.GaussianBlur(mask, (3, 3), 0)
        edges = cv2.Canny(mask, 30, 100)  # 降低阈值，减少闪烁
        return edges

    def get_color_objects_edage_original_image(
        self,
        low_color: List[int],
        high_color: List[int],
        image: np.ndarray = None
    ) -> np.ndarray:
        """
        提取图像指定颜色轮廓，并返回图像，背景为原图，轮廓为绿色
        """
        img = self._ensure_image(image).copy()
        edges = self.get_color_objects_edage_image(low_color, high_color, img)
        img[edges > 0] = (0, 255, 0)
        return img

    # ----------------------------------------------------------------------
    # 辅助工具
    # ----------------------------------------------------------------------
    def _ensure_image(self, image: np.ndarray = None) -> np.ndarray:
        if image is not None:
            return image
        if self._image is None:
            raise ValueError("未提供 image，且尚未设置原图。")
        return self._image

    def _color_mask(
        self,
        low: List[int],
        high: List[int],
        image: np.ndarray = None
    ) -> np.ndarray:
        img = self._ensure_image(image)
        img = cv2.GaussianBlur(img, (self.blur_kernel, self.blur_kernel), 0)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # 规范化到 OpenCV HSV 范围：H∈[0,179]（支持负值取模），S,V∈[0,255]
        def norm_hsv(b):
            h = int(round(b[0])) % 180
            s = int(np.clip(b[1], 0, 255))
            v = int(np.clip(b[2], 0, 255))
            return h, s, v

        h1, s1, v1 = norm_hsv(low)
        h2, s2, v2 = norm_hsv(high)

        # S/V 统一上下界
        s_lo, s_hi = min(s1, s2), max(s1, s2)
        v_lo, v_hi = min(v1, v2), max(v1, v2)

        if h1 <= h2:
            # 不跨零，单段
            mask = cv2.inRange(
                hsv,
                np.array([h1, s_lo, v_lo], np.uint8),
                np.array([h2, s_hi, v_hi], np.uint8)
            )
        else:
            # 跨零闭环，两段相加：[h1..179] ∪ [0..h2]
            mask1 = cv2.inRange(
                hsv,
                np.array([h1, s_lo, v_lo], np.uint8),
                np.array([179, s_hi, v_hi], np.uint8)
            )
            mask2 = cv2.inRange(
                hsv,
                np.array([0, s_lo, v_lo], np.uint8),
                np.array([h2, s_hi, v_hi], np.uint8)
            )
            mask = cv2.bitwise_or(mask1, mask2)

        # 形态学清理（保持简洁）
        kernel = np.ones((self.morph_kernel, self.morph_kernel), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=self.morph_iterations)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=self.morph_iterations)
        return mask


# ------------------------------ DEMO ------------------------------
if __name__ == "__main__":
    import os
    import sys

    # 使用配置文件创建提取器
    extractor = ColorExtracter(min_area=100)  # 增加最小面积阈值
    extractor.print_config()  # 显示配置信息
    
    use_img = len(sys.argv) > 1 and sys.argv[1].lower() == "img"
    
    # # 扩大蓝色范围，提高稳定性
    # blue_low, blue_high = [100, 150, 100], [130, 255, 255]  # 更宽松的范围
    # color_low, color_high = blue_low, blue_high

    # 扩大红色范围到包含两段
    red_low, red_high = [170, 20, 20], [189, 255, 255]  # 红色h（170，179）（0，10）
    color_low, color_high = red_low, red_high


    if use_img:
        img_path = os.path.join(os.path.dirname(__file__),
                                "../resources/images/others/gray.png")
        demo_img = cv2.imread(img_path)
        if demo_img is None:
            print(f"无法读取 {img_path} ，改用摄像头实时演示")
            use_cam = True
        else:
            extractor.set_original_image(demo_img)

        edge_overlay = extractor.get_edage_original_image()
        blue_only = extractor.get_color_objects_image(color_low, color_high)
        blue_edge = extractor.get_color_objects_edage_original_image(color_low, color_high)

        cv2.imshow("Edge Overlay", edge_overlay)
        cv2.imshow("Blue Objects", blue_only)
        cv2.imshow("Blue Edge", blue_edge)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    else:
        cap = extractor.create_camera_capture()
        if not cap.isOpened():
            raise RuntimeError("无法打开摄像头")
        
        # 打印实际配置
        actual_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        actual_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        print(f"摄像头ID: {extractor.camera_id}")
        print(f"摄像头分辨率: {int(actual_width)}x{int(actual_height)}")
        # print(f"左右翻转: {'启用' if extractor.flip_left_right else '禁用'}")
        
        print("按  q  退出")
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            extractor.set_original_image(frame)

            color_objects = extractor.get_color_objects(color_low, color_high)
            color_objects_edage = extractor.get_color_objects_edage(color_low, color_high)
            color_objects_image = extractor.get_color_objects_image(color_low, color_high)
            color_objects_edage_image = extractor.get_color_objects_edage_image(color_low, color_high)
            color_objects_edage_original_image = extractor.get_color_objects_edage_original_image(color_low, color_high)

            # 减少打印频率以提高性能
            frame_count = getattr(extractor, '_frame_count', 0) + 1
            extractor._frame_count = frame_count
            if frame_count % 30 == 0:  # 每30帧打印一次
                print(f"color_objects: {color_objects.shape}")
                print(f"color_objects_edage: {color_objects_edage.shape}")

            cv2.imshow("Color Objects image", color_objects_image)
            cv2.imshow("Color Objects Edge image", color_objects_edage_image)
            cv2.imshow("Color Objects Edge Original", color_objects_edage_original_image)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()


    

