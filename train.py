import os
import cv2
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical

# 数据集路径
dataset_path = '/Users/andy/Downloads/checkboxdata'

# 调整图像大小
def resize_image(image, target_size):
    resized_image = cv2.resize(image, target_size)
    return resized_image

# 加载数据集
def load_dataset(target_size):
    images = []
    labels = []
    for filename in os.listdir(dataset_path):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            # 读取图像
            image = cv2.imread(os.path.join(dataset_path, filename), cv2.IMREAD_GRAYSCALE)
            # 调整图像大小
            resized_image = resize_image(image, target_size)
            # 将图像添加到列表中
            images.append(resized_image)
            # 根据文件名判断是否为有对勾的图像
            label = 1 if 'checkmark' in filename else 0
            labels.append(label)
    return np.array(images), np.array(labels)

# 图像目标尺寸
target_size = (32, 32)

# 加载数据集
images, labels = load_dataset(target_size)

# 数据预处理
images = images.reshape(images.shape[0], target_size[0], target_size[1], 1)
labels = to_categorical(labels)

# 构建模型
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(target_size[0], target_size[1], 1)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='sigmoid'))  # 修改输出层为1个神经元，并使用sigmoid激活函数

# 编译模型
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])  # 使用binary_crossentropy损失函数

# 训练模型
model.fit(images, labels, epochs=10, batch_size=32)


# 保存模型
model.save('/Users/andy/Downloads/TrainedCheckbox')
