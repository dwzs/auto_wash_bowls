import tkinter as tk
from PIL import Image, ImageDraw, ImageTk

def update_image():
    # 获取滑块的值
    r = red_slider.get()
    g = green_slider.get()
    b = blue_slider.get()

    # 创建一个新的图像
    image = Image.new('RGB', (300, 100), 'white')
    draw = ImageDraw.Draw(image)

    # 绘制矩形
    draw.rectangle([10, 25, 60, 75], fill=(r, g, b))

    # 更新显示的图像
    tk_image = ImageTk.PhotoImage(image)
    canvas.itemconfig(image_on_canvas, image=tk_image)
    canvas.image = tk_image

# 创建主窗口
root = tk.Tk()
root.title("RGB Color Adjuster")

# 创建一个画布来显示图像
canvas = tk.Canvas(root, width=300, height=100)
canvas.pack()

# 初始化图像
initial_image = Image.new('RGB', (300, 100), 'white')
draw = ImageDraw.Draw(initial_image)
draw.rectangle([10, 25, 60, 75], fill=(0, 0, 0))
tk_image = ImageTk.PhotoImage(initial_image)
image_on_canvas = canvas.create_image(0, 0, anchor=tk.NW, image=tk_image)

# 创建滑块
red_slider = tk.Scale(root, from_=0, to=255, orient=tk.HORIZONTAL, label='Red', command=lambda x: update_image())
red_slider.pack()
green_slider = tk.Scale(root, from_=0, to=255, orient=tk.HORIZONTAL, label='Green', command=lambda x: update_image())
green_slider.pack()
blue_slider = tk.Scale(root, from_=0, to=255, orient=tk.HORIZONTAL, label='Blue', command=lambda x: update_image())
blue_slider.pack()

# 启动主循环
root.mainloop()
