#survey.py
# /Users/andy/Downloads/WechatIMG2617.jpeg"
# /Users/andy/Downloads/高雄旅行展一般客.jpg
import cv2
import numpy as np
from tensorflow.keras.models import load_model

def detect_checkmark_in_large_image(large_image_path, model_path, window_size):
    # 加载模型
    model = load_model(model_path)

    # 读取大图
    large_image = cv2.imread(large_image_path, cv2.IMREAD_GRAYSCALE)

    # 获取大图的尺寸
    image_height, image_width = large_image.shape[:2]

    # 定义窗口的步幅
    stride = window_size // 2

    # 存储找到的对勾位置
    checkmark_positions = []

    # 遍历滑动窗口
    for y in range(0, image_height - window_size + 1, stride):
        for x in range(0, image_width - window_size + 1, stride):
            # 提取窗口区域
            window = large_image[y:y+window_size, x:x+window_size]

            # 将窗口转换为与模型期望的输入形状相匹配的数组
            window = np.expand_dims(window, axis=-1)
            window = np.expand_dims(window, axis=0)

            # 预测
            prediction = model.predict(window)
            predicted_class = np.argmax(prediction[0])

            if predicted_class == 1:
                checkmark_positions.append((x, y))

    # 输出找到的对勾位置
    for position in checkmark_positions:
        print(f"检测到手写对勾，位于坐标 {position}")

large_image_path = '/Users/andy/Downloads/WechatIMG2617.jpeg'
model_path = '/Users/andy/Downloads/TrainedCheckbox'
window_size = 32  # 窗口的大小

detect_checkmark_in_large_image(large_image_path, model_path, window_size)


