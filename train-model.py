import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense
import numpy as np
import json
import os

# 1. 載入資料
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
x_train = x_train / 255.0
x_test = x_test / 255.0

# 2. 建立簡單的 MLP 模型
model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 3. 訓練模型
model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))

# 4. 建立 model 資料夾
os.makedirs("model", exist_ok=True)


# 5. 儲存權重
weights = {}
for i, layer in enumerate(model.layers):
    if isinstance(layer, Dense):
        weights[f'dense_{i}_w'] = layer.get_weights()[0]
        weights[f'dense_{i}_b'] = layer.get_weights()[1]

np.savez('model/fashion_mnist.npz', **weights)

# 6. 儲存模型結構
arch = []
for i, layer in enumerate(model.layers):
    if isinstance(layer, Flatten):
        arch.append({
            'name': f'flatten_{i}',
            'type': 'Flatten',
            'config': {},
            'weights': []
        })
    elif isinstance(layer, Dense):
        activation = layer.activation.__name__
        arch.append({
            'name': f'dense_{i}',
            'type': 'Dense',
            'config': {'activation': activation},
            'weights': [f'dense_{i}_w', f'dense_{i}_b']
        })

with open('model/fashion_mnist.json', 'w') as f:
    json.dump(arch, f, indent=2)
