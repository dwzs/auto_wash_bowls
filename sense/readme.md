class ColorExtracter:

set_original_image(image: np.array) -> None

get_original_image() -> np.array
get_edage(image = None) -> [np.array]
    作用： 提取图像的边缘，并返回边缘点坐标
    参数：
        image: 图片，缺省时使用成员变量
    输出： 边缘点，2xn 矩阵，每行表示一个点的 x,y 坐标
get_edage_image(image = None) -> np.array
    作用： 提取图像的边缘，并返回边缘图像，底色为黑，物体为白
    参数：
        image: 图片，缺省时使用成员变量
    输出： 边缘图像，底色为黑，物体为白
get_edage_original_image(image = None) -> np.array
    作用： 提取图像的边缘，并返回边缘图像，背景为原图，物体轮廓为绿色
    参数：
        image: 图片，缺省时使用成员变量
    输出： 边缘图像，背景为原图，物体轮廓为绿色


get_color_objects(low_color: list[int, int, int], high_color: list[int, int, int], image = None) -> [np.array]
    作用： 提取图像的颜色对象，并返回颜色对象点坐标
    参数：
        image: 图片，缺省时使用成员变量
        low_color: 颜色下限，三元list，表示 hsv 颜色
        high_color: 颜色上限，三元list，表示 hsv 颜色
    输出： 颜色对象，2xn 矩阵，每行表示一个颜色点的 x,y 坐标

get_color_objects_edage(low_color: list[int, int, int], high_color: list[int, int, int], image = None) -> np.array
    作用： 提取图像指定颜色轮廓，并返回轮廓点列
    参数：
        image: 图片，缺省时使用成员变量
        low_color: 颜色下限，三元list，表示 hsv 颜色
        high_color: 颜色上限，三元list，表示 hsv 颜色
    输出： 边缘点，2xn 矩阵，每行表示一个轮廓点的 x,y 坐标

get_color_objects_image(low_color: list[int, int, int], high_color: list[int, int, int], image = None) -> np.array
    作用： 提取图像的颜色对象，并返回颜色对象图像，背景为黑色，物体为原色
    参数：
        image: 图片，缺省时使用成员变量
        low_color: 颜色下限，三元list，表示 hsv 颜色
        high_color: 颜色上限，三元list，表示 hsv 颜色
    输出： 颜色对象图像，背景为黑色，物体为原色

get_color_objects_edage_image(low_color: list[int, int, int], high_color: list[int, int, int], image = None) -> np.array
    作用： 提取图像指定颜色轮廓，并返回图像，背景为黑色，轮廓为白色
    参数：
        image: 图片，缺省时使用成员变量
    输出： 提取图像指定颜色轮廓，并返回图像，背景为黑色，轮廓为白色

get_color_objects_edage_original_image(low_color: list[int, int, int], high_color: list[int, int, int], image = None) -> np.array
    作用： 提取图像指定颜色轮廓，并返回图像，背景为原图，轮廓为绿色
    参数：
        image: 图片，缺省时使用成员变量
    输出： 提取图像指定颜色轮廓，并返回图像，背景为原图，轮廓为绿色
