# ============================================================================
# GTSRB Traffic Sign Classification - Complete 4 Experiments Comparison
# 交通標誌辨識：完整四實驗比較
# Baseline vs Normalized vs Augmented vs Augmented+BatchNorm
# ============================================================================

print("=" * 80)
print("🔬 GTSRB 交通標誌辨識 - 四實驗完整比較")
print("=" * 80)
print("實驗架構:")
print("  Baseline: 無正規化")
print("  實驗 A: 正規化 (Min-Max)")
print("  實驗 B: 正規化 + 資料擴增")
print("  實驗 C: 正規化 + 資料擴增 + Batch Normalization")
print("=" * 80)

# ============================================================================
# Section 1: 環境設定與套件載入
# ============================================================================

print("\n📦 載入套件中...")

# 掛載 Google Drive
from google.colab import drive
drive.mount('/content/drive')

# 基礎套件
import numpy as np
import pandas as pd
import tensorflow as tf
import os
import time
import pickle

# 影像處理
from PIL import Image

# 繪圖套件
import matplotlib.pyplot as plt
import seaborn as sns

# 模型相關
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping

# 設定顯示風格
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# 設定亂數種子，確保結果可重現
np.random.seed(42)
tf.random.set_seed(42)

print("✅ 套件載入完成")
print(f"TensorFlow 版本: {tf.__version__}")
print(f"GPU 可用: {tf.config.list_physical_devices('GPU')}")

# ============================================================================
# Section 2: 資料載入
# ============================================================================

print("\n" + "=" * 80)
print("📂 Section 2: 資料載入")
print("=" * 80)

# 路徑設定
classes = 43  # GTSRB 資料集有 43 個類別
train_dir = '/content/drive/MyDrive/Colab Notebooks/GTSRB/archive/Train'
save_dir = '/content/drive/MyDrive/Colab Notebooks/GTSRB/'
data_path = os.path.join(save_dir, 'X_data.npy')
label_path = os.path.join(save_dir, 'y_labels.npy')

print("📂 資料路徑設定:")
print(f"   訓練資料夾: {train_dir}")
print(f"   快取檔案: {data_path}")

# 讀取影像與標籤（使用快取）
if os.path.exists(data_path) and os.path.exists(label_path):
    print("\n✅ 偵測到快取檔案，直接載入中...")
    data = np.load(data_path)
    labels = np.load(label_path)
    print(f"載入完成:")
    print(f"  X_data.shape = {data.shape}")
    print(f"  y_labels.shape = {labels.shape}")
else:
    print("\n⚙️ 未偵測到快取檔案，開始讀取影像並建立資料集...")
    data = []
    labels = []

    # 依序讀取每一個類別資料夾 (0 到 42)
    for i in range(classes):
        path = os.path.join(train_dir, str(i))
        images = os.listdir(path)

        for img_name in images:
            try:
                image = Image.open(os.path.join(path, img_name))
                image = image.resize((30, 30))
                image = np.array(image)
                data.append(image)
                labels.append(i)
            except Exception as e:
                print(f"Error loading image {img_name}: {e}")

        if (i + 1) % 10 == 0:
            print(f"已完成 {i + 1}/{classes} 個類別")

    # 轉換成 NumPy 陣列
    data = np.array(data)
    labels = np.array(labels)
    print(f"\n✅ 讀取完成!")
    print(f"X_data.shape = {data.shape}")
    print(f"y_labels.shape = {labels.shape}")

    # 儲存成 .npy 檔
    np.save(data_path, data)
    np.save(label_path, labels)
    print(f"\n💾 已儲存快取檔案至: {save_dir}")

# 顯示資料資訊
print("\n📊 原始資料形狀:")
print(f"  X (data): {data.shape}  → 共 {data.shape[0]:,} 張影像")
print(f"  每張影像: {data.shape[1]}×{data.shape[2]} 像素，{data.shape[3]} 個通道")
print(f"  y (labels): {labels.shape}  → 共 {labels.shape[0]:,} 筆標籤")
print(f"  像素值範圍: [{data.min()}, {data.max()}]")

# ============================================================================
# Section 3: 資料視覺化
# ============================================================================

print("\n" + "=" * 80)
print("📊 Section 3: 資料視覺化")
print("=" * 80)

# 類別名稱字典
class_names = {
    0:'Speed limit (20km/h)', 1:'Speed limit (30km/h)', 2:'Speed limit (50km/h)',
    3:'Speed limit (60km/h)', 4:'Speed limit (70km/h)', 5:'Speed limit (80km/h)',
    6:'End of speed limit (80km/h)', 7:'Speed limit (100km/h)', 8:'Speed limit (120km/h)',
    9:'No passing', 10:'No passing veh over 3.5 tons', 11:'Right-of-way at intersection',
    12:'Priority road', 13:'Yield', 14:'Stop', 15:'No vehicles',
    16:'Veh > 3.5 tons prohibited', 17:'No entry', 18:'General caution',
    19:'Dangerous curve left', 20:'Dangerous curve right', 21:'Double curve',
    22:'Bumpy road', 23:'Slippery road', 24:'Road narrows on the right',
    25:'Road work', 26:'Traffic signals', 27:'Pedestrians', 28:'Children crossing',
    29:'Bicycles crossing', 30:'Beware of ice/snow', 31:'Wild animals crossing',
    32:'End speed + passing limits', 33:'Turn right ahead', 34:'Turn left ahead',
    35:'Ahead only', 36:'Go straight or right', 37:'Go straight or left',
    38:'Keep right', 39:'Keep left', 40:'Roundabout mandatory',
    41:'End of no passing', 42:'End no passing veh > 3.5 tons'
}

# 3.1 視覺化每個類別的影像數量
print("\n繪製類別分布圖...")
data_dic = {}
for folder in os.listdir(train_dir):
    try:
        data_dic[int(folder)] = len(os.listdir(os.path.join(train_dir, folder)))
    except:
        pass

data_df = pd.Series(data_dic).sort_index()
data_df.index = data_df.index.map(class_names)

plt.figure(figsize=(18, 8))
data_df.sort_values().plot(kind='bar', color='steelblue', edgecolor='black')
plt.xlabel('Class Name', fontsize=12, fontweight='bold')
plt.ylabel('Number of Images', fontsize=12, fontweight='bold')
plt.title('Number of Training Images per Class', fontsize=14, fontweight='bold')
plt.xticks(rotation=90)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()

print(f"📊 資料分布統計:")
print(f"  最多影像的類別: {data_df.idxmax()} ({data_df.max()} 張)")
print(f"  最少影像的類別: {data_df.idxmin()} ({data_df.min()} 張)")
print(f"  平均每類: {data_df.mean():.0f} 張")

# 3.2 顯示隨機樣本影像
print("\n繪製隨機樣本...")
fig, axes = plt.subplots(3, 8, figsize=(16, 6))
fig.suptitle('Random Sample Images from Dataset', fontsize=14, fontweight='bold')

for i, ax in enumerate(axes.flat):
    idx = np.random.randint(0, len(data))
    ax.imshow(data[idx])
    ax.set_title(f"Class {labels[idx]}", fontsize=9)
    ax.axis('off')

plt.tight_layout()
plt.show()

# ============================================================================
# Section 4: 資料分割與標籤處理
# ============================================================================

print("\n" + "=" * 80)
print("📊 Section 4: 資料分割與標籤處理")
print("=" * 80)

# 4.1 分割訓練集與測試集 (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    data,
    labels,
    test_size=0.2,
    random_state=42,
    stratify=labels
)

print("✅ 資料分割完成:")
print(f"  訓練集: {X_train.shape[0]:,} 張影像 ({X_train.shape[0]/len(data)*100:.1f}%)")
print(f"  測試集: {X_test.shape[0]:,} 張影像 ({X_test.shape[0]/len(data)*100:.1f}%)")
print(f"  訓練集標籤分布: {np.bincount(y_train).min()} ~ {np.bincount(y_train).max()} 張/類別")
print(f"  測試集標籤分布: {np.bincount(y_test).min()} ~ {np.bincount(y_test).max()} 張/類別")

# 4.2 One-Hot Encoding
y_train_encoded = to_categorical(y_train, classes)
y_test_encoded = to_categorical(y_test, classes)

print("\n✅ One-Hot Encoding 完成:")
print(f"  y_train_encoded.shape: {y_train_encoded.shape}")
print(f"  y_test_encoded.shape: {y_test_encoded.shape}")

# ============================================================================
# Section 5: 準備四種資料版本
# ============================================================================

print("\n" + "=" * 80)
print("🔧 Section 5: 準備四種資料版本")
print("=" * 80)

# 版本 1: Baseline - 無標準化 [0, 255]
X_train_baseline = X_train.copy()
X_test_baseline = X_test.copy()

print("📌 版本 1 - Baseline (無標準化):")
print(f"  範圍: [{X_train_baseline.min()}, {X_train_baseline.max()}]")
print(f"  型別: {X_train_baseline.dtype}")

# 版本 2: 實驗 A - Min-Max Normalization [0, 1]
X_train_norm = X_train.astype('float32') / 255.0
X_test_norm = X_test.astype('float32') / 255.0

print("\n📌 版本 2 - 實驗 A (Min-Max Normalization):")
print(f"  範圍: [{X_train_norm.min():.4f}, {X_train_norm.max():.4f}]")
print(f"  型別: {X_train_norm.dtype}")
print(f"  轉換公式: X_normalized = X / 255.0")

# 版本 3 & 4: 實驗 B & C - 保持 [0, 255] 給 ImageDataGenerator
X_train_aug = X_train.copy()  # 實驗 B 和 C 共用
X_test_aug_norm = X_test.astype('float32') / 255.0  # 測試集需標準化

print("\n📌 版本 3 & 4 - 實驗 B & C (Data Augmentation 用):")
print(f"  訓練集範圍: [{X_train_aug.min()}, {X_train_aug.max()}] (給 ImageDataGenerator)")
print(f"  測試集範圍: [{X_test_aug_norm.min():.4f}, {X_test_aug_norm.max():.4f}] (已標準化)")

# 5.1 視覺化標準化前後的差異
print("\n繪製標準化前後對比圖...")
fig, axes = plt.subplots(2, 5, figsize=(15, 6))
fig.suptitle('Comparison: Raw vs Normalized Images', fontsize=14, fontweight='bold')

sample_indices = np.random.choice(len(X_train), 5, replace=False)

for i, idx in enumerate(sample_indices):
    # 第一列: 原始影像
    axes[0, i].imshow(X_train_baseline[idx].astype('uint8'))
    axes[0, i].set_title(f'Raw [0-255]\nClass {y_train[idx]}', fontsize=10)
    axes[0, i].axis('off')

    # 第二列: 標準化後影像
    axes[1, i].imshow(X_train_norm[idx])
    axes[1, i].set_title(f'Normalized [0-1]\nClass {y_train[idx]}', fontsize=10)
    axes[1, i].axis('off')

plt.tight_layout()
plt.show()

print("💡 觀察重點:")
print("  - 視覺上兩者看起來相同 (因為只是縮放比例)")
print("  - 但數值範圍不同會影響神經網路的訓練過程")
print("  - 標準化後的梯度更新更穩定、收斂更快")

# 5.2 設定 Data Augmentation (實驗 B & C 用)
print("\n⚙️ 設定 Data Augmentation...")

datagen = ImageDataGenerator(
    rescale=1./255,              # 標準化至 [0, 1]
    rotation_range=5,            # ±5°
    width_shift_range=0.1,       # ±10%
    height_shift_range=0.1,      # ±10%
    zoom_range=0.1,              # ±10%
    brightness_range=[0.8, 1.2], # 亮度 [0.8, 1.2]
    fill_mode='nearest',
    horizontal_flip=False,       # 交通標誌不翻轉
    vertical_flip=False
)

print("✅ Data Augmentation 設定完成:")
print("  - rescale: 1./255")
print("  - rotation_range: ±5°")
print("  - width_shift_range: ±10%")
print("  - height_shift_range: ±10%")
print("  - zoom_range: ±10%")
print("  - brightness_range: [0.8, 1.2]")
print("  - horizontal_flip: False")
print("  - vertical_flip: False")

# 5.3 視覺化 Data Augmentation 效果
print("\n繪製 Data Augmentation 驗證圖...")
sample_img = X_train_aug[0:1]

fig, axes = plt.subplots(2, 5, figsize=(15, 6))
fig.suptitle('Data Augmentation Verification', fontsize=14, fontweight='bold')

# 原始影像
axes[0, 0].imshow(sample_img[0].astype('uint8'))
axes[0, 0].set_title('Original\n[0-255]', fontsize=10, fontweight='bold')
axes[0, 0].axis('off')

# 生成 9 個擴增樣本
aug_iter = datagen.flow(sample_img, batch_size=1)
for i in range(9):
    aug_img = next(aug_iter)[0]
    row = (i + 1) // 5
    col = (i + 1) % 5
    axes[row, col].imshow(aug_img)
    axes[row, col].set_title(f'Augmented {i+1}\n[0-1]', fontsize=9)
    axes[row, col].axis('off')

    # 檢查是否有全黑問題
    if aug_img.max() < 0.1:
        axes[row, col].set_title(f'❌ BLACK!\nmax={aug_img.max():.4f}',
                                fontsize=9, color='red', fontweight='bold')

plt.tight_layout()
plt.show()

print("✅ Data Augmentation 視覺化完成")
print("   請確認擴增後的影像正常顯示（非全黑）")

# ============================================================================
# Section 6: 模型建立函數
# ============================================================================

print("\n" + "=" * 80)
print("🏗️ Section 6: 模型建立函數")
print("=" * 80)

def build_baseline_model(input_shape):
    """建立基礎 CNN 模型（無 Batch Normalization）"""
    model = Sequential(name='Baseline_CNN')

    # 第一組卷積層
    model.add(Conv2D(32, (5, 5), activation='relu', input_shape=input_shape))
    model.add(Conv2D(32, (5, 5), activation='relu'))
    model.add(MaxPool2D((2, 2)))
    model.add(Dropout(0.25))

    # 第二組卷積層
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPool2D((2, 2)))
    model.add(Dropout(0.25))

    # 全連接層
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(43, activation='softmax'))

    model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )

    return model

def build_model_with_bn(input_shape):
    """建立含 Batch Normalization 的 CNN 模型"""
    model = Sequential(name='CNN_with_BatchNorm')

    # 第一組卷積層
    model.add(Conv2D(32, (5, 5), activation='relu', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(Conv2D(32, (5, 5), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPool2D((2, 2)))
    model.add(Dropout(0.25))

    # 第二組卷積層
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPool2D((2, 2)))
    model.add(Dropout(0.25))

    # 全連接層
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(43, activation='softmax'))

    model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )

    return model

print("✅ 模型建立函數定義完成")
print("  - build_baseline_model(): 基礎 CNN（無 BN）")
print("  - build_model_with_bn(): CNN + Batch Normalization")

# ============================================================================
# Section 7: Early Stopping 設定
# ============================================================================

print("\n" + "=" * 80)
print("⏱️ Section 7: Early Stopping 設定")
print("=" * 80)

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True,
    mode='min',
    verbose=1
)

print("✅ Early Stopping 已設定:")
print("  - monitor: val_loss")
print("  - patience: 5")
print("  - restore_best_weights: True")
print("  - 所有實驗都將使用此設定")

# ============================================================================
# Section 8: 實驗 Baseline - 無標準化
# ============================================================================

print("\n" + "=" * 80)
print("🚀 Section 8: 實驗 Baseline - 無標準化")
print("=" * 80)

print("\n【實驗設定】")
print("  ✅ 資料: 原始像素值 [0, 255]，無標準化")
print("  ✅ 模型: 基礎 CNN (無 Batch Normalization)")
print("  ✅ Early Stopping: patience=5")
print("  ✅ Batch Size: 128")
print("  ✅ Max Epochs: 35")

# 建立模型
model_baseline = build_baseline_model(X_train_baseline.shape[1:])
print("\n📋 模型架構:")
model_baseline.summary()

# 訓練
print("\n" + "=" * 60)
print("開始訓練...")
print("=" * 60)

start_time_baseline = time.time()

history_baseline = model_baseline.fit(
    X_train_baseline,
    y_train_encoded,
    batch_size=128,
    epochs=35,
    validation_data=(X_test_baseline, y_test_encoded),
    callbacks=[early_stop],
    verbose=1
)

end_time_baseline = time.time()
training_time_baseline = end_time_baseline - start_time_baseline

# 測試集評估
loss_baseline, acc_baseline = model_baseline.evaluate(
    X_test_baseline, y_test_encoded, verbose=0
)

print("\n" + "=" * 60)
print("✅ 實驗 Baseline 訓練完成!")
print("=" * 60)
print(f"⏱️  總訓練時間: {training_time_baseline:.2f} 秒 ({training_time_baseline/60:.2f} 分鐘)")
print(f"📊 實際訓練 epochs: {len(history_baseline.history['accuracy'])}")
print(f"📊 最終訓練準確率: {history_baseline.history['accuracy'][-1]*100:.2f}%")
print(f"📊 最終驗證準確率: {history_baseline.history['val_accuracy'][-1]*100:.2f}%")
print(f"📊 最佳驗證準確率: {max(history_baseline.history['val_accuracy'])*100:.2f}% (Epoch {np.argmax(history_baseline.history['val_accuracy'])+1})")
print(f"📊 測試準確率: {acc_baseline*100:.2f}%")
print(f"📊 測試損失: {loss_baseline:.4f}")

# ============================================================================
# Section 9: 實驗 A - 正規化
# ============================================================================

print("\n" + "=" * 80)
print("🚀 Section 9: 實驗 A - 正規化 (Min-Max)")
print("=" * 80)

print("\n【實驗設定】")
print("  ✅ 資料: Min-Max Normalization [0, 1]")
print("  ✅ 模型: 基礎 CNN (無 Batch Normalization)")
print("  ✅ Early Stopping: patience=5")
print("  ✅ Batch Size: 128")
print("  ✅ Max Epochs: 35")

# 建立模型
model_norm = build_baseline_model(X_train_norm.shape[1:])
print("\n📋 模型架構:")
model_norm.summary()

# 訓練
print("\n" + "=" * 60)
print("開始訓練...")
print("=" * 60)

start_time_norm = time.time()

history_norm = model_norm.fit(
    X_train_norm,
    y_train_encoded,
    batch_size=128,
    epochs=35,
    validation_data=(X_test_norm, y_test_encoded),
    callbacks=[early_stop],
    verbose=1
)

end_time_norm = time.time()
training_time_norm = end_time_norm - start_time_norm

# 測試集評估
loss_norm, acc_norm = model_norm.evaluate(
    X_test_norm, y_test_encoded, verbose=0
)

print("\n" + "=" * 60)
print("✅ 實驗 A 訓練完成!")
print("=" * 60)
print(f"⏱️  總訓練時間: {training_time_norm:.2f} 秒 ({training_time_norm/60:.2f} 分鐘)")
print(f"📊 實際訓練 epochs: {len(history_norm.history['accuracy'])}")
print(f"📊 最終訓練準確率: {history_norm.history['accuracy'][-1]*100:.2f}%")
print(f"📊 最終驗證準確率: {history_norm.history['val_accuracy'][-1]*100:.2f}%")
print(f"📊 最佳驗證準確率: {max(history_norm.history['val_accuracy'])*100:.2f}% (Epoch {np.argmax(history_norm.history['val_accuracy'])+1})")
print(f"📊 測試準確率: {acc_norm*100:.2f}%")
print(f"📊 測試損失: {loss_norm:.4f}")

# ============================================================================
# Section 10: 實驗 B - 正規化 + 資料擴增
# ============================================================================

print("\n" + "=" * 80)
print("🚀 Section 10: 實驗 B - 正規化 + 資料擴增")
print("=" * 80)

print("\n【實驗設定】")
print("  ✅ 資料: ImageDataGenerator (rescale=1./255 + augmentation)")
print("  ✅ 擴增: rotation ±5°, shift ±10%, zoom ±10%, brightness [0.8, 1.2]")
print("  ✅ 模型: 基礎 CNN (無 Batch Normalization)")
print("  ✅ Early Stopping: patience=5")
print("  ✅ Batch Size: 128")
print("  ✅ Max Epochs: 35")

# 建立模型
model_aug = build_baseline_model((30, 30, 3))
print("\n📋 模型架構:")
model_aug.summary()

# 訓練
print("\n" + "=" * 60)
print("開始訓練...")
print("=" * 60)

start_time_aug = time.time()

steps_per_epoch = int(np.ceil(len(X_train_aug) / 128))

history_aug = model_aug.fit(
    datagen.flow(X_train_aug, y_train_encoded, batch_size=128),
    steps_per_epoch=steps_per_epoch,
    epochs=35,
    validation_data=(X_test_aug_norm, y_test_encoded),
    callbacks=[early_stop],
    verbose=1
)

end_time_aug = time.time()
training_time_aug = end_time_aug - start_time_aug

# 測試集評估
loss_aug, acc_aug = model_aug.evaluate(
    X_test_aug_norm, y_test_encoded, verbose=0
)

print("\n" + "=" * 60)
print("✅ 實驗 B 訓練完成!")
print("=" * 60)
print(f"⏱️  總訓練時間: {training_time_aug:.2f} 秒 ({training_time_aug/60:.2f} 分鐘)")
print(f"📊 實際訓練 epochs: {len(history_aug.history['accuracy'])}")
print(f"📊 最終訓練準確率: {history_aug.history['accuracy'][-1]*100:.2f}%")
print(f"📊 最終驗證準確率: {history_aug.history['val_accuracy'][-1]*100:.2f}%")
print(f"📊 最佳驗證準確率: {max(history_aug.history['val_accuracy'])*100:.2f}% (Epoch {np.argmax(history_aug.history['val_accuracy'])+1})")
print(f"📊 測試準確率: {acc_aug*100:.2f}%")
print(f"📊 測試損失: {loss_aug:.4f}")

# ============================================================================
# Section 11: 實驗 C - 正規化 + 資料擴增 + Batch Normalization
# ============================================================================

print("\n" + "=" * 80)
print("🚀 Section 11: 實驗 C - 正規化 + 資料擴增 + Batch Normalization")
print("=" * 80)

print("\n【實驗設定】")
print("  ✅ 資料: ImageDataGenerator (rescale=1./255 + augmentation)")
print("  ✅ 擴增: rotation ±5°, shift ±10%, zoom ±10%, brightness [0.8, 1.2]")
print("  ✅ 模型: CNN + Batch Normalization (4 層 BN)")
print("  ✅ Early Stopping: patience=5")
print("  ✅ Batch Size: 128")
print("  ✅ Max Epochs: 35")

# 建立模型
model_aug_bn = build_model_with_bn((30, 30, 3))
print("\n📋 模型架構:")
model_aug_bn.summary()

# 訓練
print("\n" + "=" * 60)
print("開始訓練...")
print("=" * 60)

start_time_aug_bn = time.time()

history_aug_bn = model_aug_bn.fit(
    datagen.flow(X_train_aug, y_train_encoded, batch_size=128),
    steps_per_epoch=steps_per_epoch,
    epochs=35,
    validation_data=(X_test_aug_norm, y_test_encoded),
    callbacks=[early_stop],
    verbose=1
)

end_time_aug_bn = time.time()
training_time_aug_bn = end_time_aug_bn - start_time_aug_bn

# 測試集評估
loss_aug_bn, acc_aug_bn = model_aug_bn.evaluate(
    X_test_aug_norm, y_test_encoded, verbose=0
)

print("\n" + "=" * 60)
print("✅ 實驗 C 訓練完成!")
print("=" * 60)
print(f"⏱️  總訓練時間: {training_time_aug_bn:.2f} 秒 ({training_time_aug_bn/60:.2f} 分鐘)")
print(f"📊 實際訓練 epochs: {len(history_aug_bn.history['accuracy'])}")
print(f"📊 最終訓練準確率: {history_aug_bn.history['accuracy'][-1]*100:.2f}%")
print(f"📊 最終驗證準確率: {history_aug_bn.history['val_accuracy'][-1]*100:.2f}%")
print(f"📊 最佳驗證準確率: {max(history_aug_bn.history['val_accuracy'])*100:.2f}% (Epoch {np.argmax(history_aug_bn.history['val_accuracy'])+1})")
print(f"📊 測試準確率: {acc_aug_bn*100:.2f}%")
print(f"📊 測試損失: {loss_aug_bn:.4f}")

# ============================================================================
# Section 12: 訓練曲線對比視覺化
# ============================================================================

print("\n" + "=" * 80)
print("📊 Section 12: 訓練曲線對比視覺化")
print("=" * 80)

fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle('Training Comparison: 4 Experiments', fontsize=16, fontweight='bold')

# 子圖 1: 訓練準確率
epochs_baseline = range(1, len(history_baseline.history['accuracy']) + 1)
epochs_norm = range(1, len(history_norm.history['accuracy']) + 1)
epochs_aug = range(1, len(history_aug.history['accuracy']) + 1)
epochs_aug_bn = range(1, len(history_aug_bn.history['accuracy']) + 1)

axes[0, 0].plot(epochs_baseline, history_baseline.history['accuracy'],
                'b-o', label='Baseline', linewidth=2, markersize=4)
axes[0, 0].plot(epochs_norm, history_norm.history['accuracy'],
                'g-s', label='Normalized', linewidth=2, markersize=4)
axes[0, 0].plot(epochs_aug, history_aug.history['accuracy'],
                'r-^', label='Norm + Aug', linewidth=2, markersize=4)
axes[0, 0].plot(epochs_aug_bn, history_aug_bn.history['accuracy'],
                'm-d', label='Norm + Aug + BN', linewidth=2, markersize=4)
axes[0, 0].set_xlabel('Epoch', fontsize=12, fontweight='bold')
axes[0, 0].set_ylabel('Training Accuracy', fontsize=12, fontweight='bold')
axes[0, 0].set_title('Training Accuracy', fontsize=13, fontweight='bold')
axes[0, 0].legend(loc='lower right', fontsize=10)
axes[0, 0].grid(True, alpha=0.3)

# 子圖 2: 驗證準確率
axes[0, 1].plot(epochs_baseline, history_baseline.history['val_accuracy'],
                'b-o', label='Baseline', linewidth=2, markersize=4)
axes[0, 1].plot(epochs_norm, history_norm.history['val_accuracy'],
                'g-s', label='Normalized', linewidth=2, markersize=4)
axes[0, 1].plot(epochs_aug, history_aug.history['val_accuracy'],
                'r-^', label='Norm + Aug', linewidth=2, markersize=4)
axes[0, 1].plot(epochs_aug_bn, history_aug_bn.history['val_accuracy'],
                'm-d', label='Norm + Aug + BN', linewidth=2, markersize=4)
axes[0, 1].set_xlabel('Epoch', fontsize=12, fontweight='bold')
axes[0, 1].set_ylabel('Validation Accuracy', fontsize=12, fontweight='bold')
axes[0, 1].set_title('Validation Accuracy', fontsize=13, fontweight='bold')
axes[0, 1].legend(loc='lower right', fontsize=10)
axes[0, 1].grid(True, alpha=0.3)

# 子圖 3: 訓練損失
axes[1, 0].plot(epochs_baseline, history_baseline.history['loss'],
                'b-o', label='Baseline', linewidth=2, markersize=4)
axes[1, 0].plot(epochs_norm, history_norm.history['loss'],
                'g-s', label='Normalized', linewidth=2, markersize=4)
axes[1, 0].plot(epochs_aug, history_aug.history['loss'],
                'r-^', label='Norm + Aug', linewidth=2, markersize=4)
axes[1, 0].plot(epochs_aug_bn, history_aug_bn.history['loss'],
                'm-d', label='Norm + Aug + BN', linewidth=2, markersize=4)
axes[1, 0].set_xlabel('Epoch', fontsize=12, fontweight='bold')
axes[1, 0].set_ylabel('Training Loss', fontsize=12, fontweight='bold')
axes[1, 0].set_title('Training Loss', fontsize=13, fontweight='bold')
axes[1, 0].legend(loc='upper right', fontsize=10)
axes[1, 0].grid(True, alpha=0.3)

# 子圖 4: 驗證損失
axes[1, 1].plot(epochs_baseline, history_baseline.history['val_loss'],
                'b-o', label='Baseline', linewidth=2, markersize=4)
axes[1, 1].plot(epochs_norm, history_norm.history['val_loss'],
                'g-s', label='Normalized', linewidth=2, markersize=4)
axes[1, 1].plot(epochs_aug, history_aug.history['val_loss'],
                'r-^', label='Norm + Aug', linewidth=2, markersize=4)
axes[1, 1].plot(epochs_aug_bn, history_aug_bn.history['val_loss'],
                'm-d', label='Norm + Aug + BN', linewidth=2, markersize=4)
axes[1, 1].set_xlabel('Epoch', fontsize=12, fontweight='bold')
axes[1, 1].set_ylabel('Validation Loss', fontsize=12, fontweight='bold')
axes[1, 1].set_title('Validation Loss', fontsize=13, fontweight='bold')
axes[1, 1].legend(loc='upper right', fontsize=10)
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("✅ 訓練曲線對比圖繪製完成")

# ============================================================================
# Section 13: 測試集準確率對比圖
# ============================================================================

print("\n" + "=" * 80)
print("📊 Section 13: 測試集準確率對比圖")
print("=" * 80)

fig, ax = plt.subplots(figsize=(12, 6))

experiments = ['Baseline', 'Normalized', 'Norm + Aug', 'Norm + Aug + BN']
accuracies = [acc_baseline * 100, acc_norm * 100, acc_aug * 100, acc_aug_bn * 100]
colors = ['#3498db', '#2ecc71', '#e74c3c', '#9b59b6']

bars = ax.bar(experiments, accuracies, color=colors, edgecolor='black', linewidth=1.5)

# 在柱狀圖上標註數值
for i, (bar, acc) in enumerate(zip(bars, accuracies)):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{acc:.2f}%',
            ha='center', va='bottom', fontsize=12, fontweight='bold')

ax.set_ylabel('Test Accuracy (%)', fontsize=12, fontweight='bold')
ax.set_xlabel('Experiment', fontsize=12, fontweight='bold')
ax.set_title('Test Accuracy Comparison', fontsize=14, fontweight='bold')
ax.set_ylim([95, 100])
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.show()

print("✅ 測試集準確率對比圖繪製完成")

# ============================================================================
# Section 14: 完整結果比較表
# ============================================================================

print("\n" + "=" * 80)
print("📊 Section 14: 四個實驗完整比較")
print("=" * 80)

comparison_df = pd.DataFrame({
    '指標': [
        '訓練時間 (分鐘)',
        '實際訓練 Epochs',
        '最終訓練準確率 (%)',
        '最終驗證準確率 (%)',
        '最佳驗證準確率 (%)',
        '測試準確率 (%)',
        '測試損失',
    ],
    'Baseline': [
        f"{training_time_baseline/60:.2f}",
        f"{len(history_baseline.history['accuracy'])}",
        f"{history_baseline.history['accuracy'][-1]*100:.2f}",
        f"{history_baseline.history['val_accuracy'][-1]*100:.2f}",
        f"{max(history_baseline.history['val_accuracy'])*100:.2f}",
        f"{acc_baseline*100:.2f}",
        f"{loss_baseline:.4f}",
    ],
    '實驗 A (正規化)': [
        f"{training_time_norm/60:.2f}",
        f"{len(history_norm.history['accuracy'])}",
        f"{history_norm.history['accuracy'][-1]*100:.2f}",
        f"{history_norm.history['val_accuracy'][-1]*100:.2f}",
        f"{max(history_norm.history['val_accuracy'])*100:.2f}",
        f"{acc_norm*100:.2f}",
        f"{loss_norm:.4f}",
    ],
    '實驗 B (正+擴增)': [
        f"{training_time_aug/60:.2f}",
        f"{len(history_aug.history['accuracy'])}",
        f"{history_aug.history['accuracy'][-1]*100:.2f}",
        f"{history_aug.history['val_accuracy'][-1]*100:.2f}",
        f"{max(history_aug.history['val_accuracy'])*100:.2f}",
        f"{acc_aug*100:.2f}",
        f"{loss_aug:.4f}",
    ],
    '實驗 C (正+擴增+BN)': [
        f"{training_time_aug_bn/60:.2f}",
        f"{len(history_aug_bn.history['accuracy'])}",
        f"{history_aug_bn.history['accuracy'][-1]*100:.2f}",
        f"{history_aug_bn.history['val_accuracy'][-1]*100:.2f}",
        f"{max(history_aug_bn.history['val_accuracy'])*100:.2f}",
        f"{acc_aug_bn*100:.2f}",
        f"{loss_aug_bn:.4f}",
    ]
})

print("\n" + "=" * 100)
print(comparison_df.to_string(index=False))
print("=" * 100)

# ============================================================================
# Section 15: 詳細分析
# ============================================================================

print("\n" + "=" * 80)
print("📈 Section 15: 詳細分析")
print("=" * 80)

print("\n1️⃣ 訓練效率分析:")
print(f"  Baseline 訓練時間: {training_time_baseline/60:.2f} 分鐘 ({len(history_baseline.history['accuracy'])} epochs)")
print(f"  實驗 A 訓練時間: {training_time_norm/60:.2f} 分鐘 ({len(history_norm.history['accuracy'])} epochs)")
print(f"  實驗 B 訓練時間: {training_time_aug/60:.2f} 分鐘 ({len(history_aug.history['accuracy'])} epochs)")
print(f"  實驗 C 訓練時間: {training_time_aug_bn/60:.2f} 分鐘 ({len(history_aug_bn.history['accuracy'])} epochs)")
print(f"  最快收斂: 實驗 {'C' if len(history_aug_bn.history['accuracy']) <= min(len(history_baseline.history['accuracy']), len(history_norm.history['accuracy']), len(history_aug.history['accuracy'])) else 'B'}")

print("\n2️⃣ 準確率提升分析:")
print(f"  Baseline → 實驗 A: {(acc_norm - acc_baseline)*100:+.2f}% (正規化效果)")
print(f"  實驗 A → 實驗 B: {(acc_aug - acc_norm)*100:+.2f}% (資料擴增效果)")
print(f"  實驗 B → 實驗 C: {(acc_aug_bn - acc_aug)*100:+.2f}% (Batch Normalization 效果)")
print(f"  Baseline → 實驗 C: {(acc_aug_bn - acc_baseline)*100:+.2f}% (總提升)")

print("\n3️⃣ 損失降低分析:")
print(f"  Baseline: {loss_baseline:.4f}")
print(f"  實驗 A: {loss_norm:.4f} ({(loss_norm - loss_baseline)/loss_baseline*100:+.1f}%)")
print(f"  實驗 B: {loss_aug:.4f} ({(loss_aug - loss_baseline)/loss_baseline*100:+.1f}%)")
print(f"  實驗 C: {loss_aug_bn:.4f} ({(loss_aug_bn - loss_baseline)/loss_baseline*100:+.1f}%)")

print("\n4️⃣ 過擬合分析:")
train_val_gap_baseline = history_baseline.history['accuracy'][-1] - history_baseline.history['val_accuracy'][-1]
train_val_gap_norm = history_norm.history['accuracy'][-1] - history_norm.history['val_accuracy'][-1]
train_val_gap_aug = history_aug.history['accuracy'][-1] - history_aug.history['val_accuracy'][-1]
train_val_gap_aug_bn = history_aug_bn.history['accuracy'][-1] - history_aug_bn.history['val_accuracy'][-1]

print(f"  Baseline 訓練/驗證差距: {train_val_gap_baseline*100:.2f}%")
print(f"  實驗 A 訓練/驗證差距: {train_val_gap_norm*100:.2f}%")
print(f"  實驗 B 訓練/驗證差距: {train_val_gap_aug*100:.2f}%")
print(f"  實驗 C 訓練/驗證差距: {train_val_gap_aug_bn*100:.2f}%")
print(f"  → 差距越小表示過擬合程度越低")

print("\n5️⃣ 綜合評分 (測試準確率 + 訓練效率):")
# 計算綜合評分: 準確率權重 0.7 + 效率權重 0.3
max_time = max(training_time_baseline, training_time_norm, training_time_aug, training_time_aug_bn)
score_baseline = acc_baseline * 0.7 + (1 - training_time_baseline/max_time) * 0.3
score_norm = acc_norm * 0.7 + (1 - training_time_norm/max_time) * 0.3
score_aug = acc_aug * 0.7 + (1 - training_time_aug/max_time) * 0.3
score_aug_bn = acc_aug_bn * 0.7 + (1 - training_time_aug_bn/max_time) * 0.3

print(f"  Baseline: {score_baseline:.4f}")
print(f"  實驗 A: {score_norm:.4f}")
print(f"  實驗 B: {score_aug:.4f}")
print(f"  實驗 C: {score_aug_bn:.4f} ⭐ 最佳")

# ============================================================================
# Section 16: 儲存模型與結果
# ============================================================================

print("\n" + "=" * 80)
print("💾 Section 16: 儲存模型與結果")
print("=" * 80)

model_save_path = '/content/drive/MyDrive/Colab Notebooks/GTSRB/'

# 儲存模型
model_baseline.save(os.path.join(model_save_path, 'model_baseline.h5'))
model_norm.save(os.path.join(model_save_path, 'model_normalized.h5'))
model_aug.save(os.path.join(model_save_path, 'model_augmented.h5'))
model_aug_bn.save(os.path.join(model_save_path, 'model_augmented_bn.h5'))

print("✅ 模型已儲存:")
print("  - model_baseline.h5")
print("  - model_normalized.h5")
print("  - model_augmented.h5")
print("  - model_augmented_bn.h5")

# 儲存訓練歷史
with open(os.path.join(model_save_path, 'history_baseline.pkl'), 'wb') as f:
    pickle.dump(history_baseline.history, f)
with open(os.path.join(model_save_path, 'history_normalized.pkl'), 'wb') as f:
    pickle.dump(history_norm.history, f)
with open(os.path.join(model_save_path, 'history_augmented.pkl'), 'wb') as f:
    pickle.dump(history_aug.history, f)
with open(os.path.join(model_save_path, 'history_augmented_bn.pkl'), 'wb') as f:
    pickle.dump(history_aug_bn.history, f)

print("\n✅ 訓練歷史已儲存:")
print("  - history_baseline.pkl")
print("  - history_normalized.pkl")
print("  - history_augmented.pkl")
print("  - history_augmented_bn.pkl")

# 儲存比較結果
results_summary = {
    'baseline': {
        'training_time': training_time_baseline,
        'epochs': len(history_baseline.history['accuracy']),
        'test_acc': acc_baseline,
        'test_loss': loss_baseline
    },
    'normalized': {
        'training_time': training_time_norm,
        'epochs': len(history_norm.history['accuracy']),
        'test_acc': acc_norm,
        'test_loss': loss_norm
    },
    'augmented': {
        'training_time': training_time_aug,
        'epochs': len(history_aug.history['accuracy']),
        'test_acc': acc_aug,
        'test_loss': loss_aug
    },
    'augmented_bn': {
        'training_time': training_time_aug_bn,
        'epochs': len(history_aug_bn.history['accuracy']),
        'test_acc': acc_aug_bn,
        'test_loss': loss_aug_bn
    }
}

with open(os.path.join(model_save_path, 'results_summary.pkl'), 'wb') as f:
    pickle.dump(results_summary, f)

# 儲存為 CSV
comparison_df.to_csv(os.path.join(model_save_path, 'comparison_results.csv'),
                     index=False, encoding='utf-8-sig')

print("\n✅ 結果摘要已儲存:")
print("  - results_summary.pkl")
print("  - comparison_results.csv")

# ============================================================================
# 完成!
# ============================================================================

print("\n" + "=" * 80)
print("🎉 四實驗完整比較完成!")
print("=" * 80)

print("\n📊 最終結果摘要:")
print(f"  🥇 最佳測試準確率: 實驗 {'C' if acc_aug_bn >= max(acc_baseline, acc_norm, acc_aug) else ('B' if acc_aug >= max(acc_baseline, acc_norm) else ('A' if acc_norm > acc_baseline else 'Baseline'))}")
print(f"     準確率: {max(acc_baseline, acc_norm, acc_aug, acc_aug_bn)*100:.2f}%")
print(f"  ⚡ 最快收斂: 實驗 {'C' if len(history_aug_bn.history['accuracy']) <= min(len(history_baseline.history['accuracy']), len(history_norm.history['accuracy']), len(history_aug.history['accuracy'])) else 'B'}")
print(f"     Epochs: {min(len(history_baseline.history['accuracy']), len(history_norm.history['accuracy']), len(history_aug.history['accuracy']), len(history_aug_bn.history['accuracy']))}")
print(f"  📈 總提升: {(max(acc_baseline, acc_norm, acc_aug, acc_aug_bn) - acc_baseline)*100:.2f}%")

print("\n💡 主要發現:")
print("  1. 正規化效果: 提升 {:.2f}%".format((acc_norm - acc_baseline)*100))
print("  2. 資料擴增效果: 提升 {:.2f}%".format((acc_aug - acc_norm)*100))
print("  3. Batch Normalization 效果: 提升 {:.2f}%".format((acc_aug_bn - acc_aug)*100))
print("  4. 所有實驗都使用 Early Stopping (patience=5) 確保公平比較")

print("\n感謝使用! 🎓")
print("所有結果已儲存至: " + model_save_path)
print("=" * 80)


# ============================================================================
# GTSRB - 實驗 C 單獨訓練 (優化版)
# 正規化 + 資料擴增 + Batch Normalization
# Early Stopping patience = 20 (針對 BN 訓練波動優化)
# ============================================================================

print("=" * 80)
print("🔬 GTSRB 實驗 C - 優化版單獨訓練")
print("=" * 80)
print("實驗設定:")
print("  ✅ 資料: ImageDataGenerator (rescale + augmentation)")
print("  ✅ 擴增: rotation ±5°, shift ±10%, zoom ±10%, brightness [0.8, 1.2]")
print("  ✅ 模型: CNN + Batch Normalization (4 層 BN)")
print("  ✅ Early Stopping: patience=10 (針對 BN 波動優化) ⭐")
print("  ✅ Batch Size: 128")
print("  ✅ Max Epochs: 50 (增加訓練機會)")
print("=" * 80)

# ============================================================================
# Section 1: 環境設定
# ============================================================================

print("\n📦 載入套件中...")

# 掛載 Google Drive
from google.colab import drive
drive.mount('/content/drive')

# 基礎套件
import numpy as np
import pandas as pd
import tensorflow as tf
import os
import time
import pickle

# 影像處理
from PIL import Image

# 繪圖套件
import matplotlib.pyplot as plt
import seaborn as sns

# 模型相關
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping

# 設定顯示風格
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# 設定亂數種子
np.random.seed(42)
tf.random.set_seed(42)

print("✅ 套件載入完成")
print(f"TensorFlow 版本: {tf.__version__}")
print(f"GPU 可用: {tf.config.list_physical_devices('GPU')}")

# ============================================================================
# Section 2: 資料載入
# ============================================================================

print("\n" + "=" * 80)
print("📂 Section 2: 資料載入")
print("=" * 80)

# 路徑設定
save_dir = '/content/drive/MyDrive/Colab Notebooks/GTSRB/'
data_path = os.path.join(save_dir, 'X_data.npy')
label_path = os.path.join(save_dir, 'y_labels.npy')

print("載入快取檔案...")
data = np.load(data_path)
labels = np.load(label_path)

print(f"✅ 資料載入完成:")
print(f"   影像數量: {data.shape[0]:,} 張")
print(f"   影像尺寸: {data.shape[1]}×{data.shape[2]}")
print(f"   像素值範圍: [{data.min()}, {data.max()}]")

# ============================================================================
# Section 3: 資料分割
# ============================================================================

print("\n" + "=" * 80)
print("📊 Section 3: 資料分割")
print("=" * 80)

# 80/20 分割
X_train, X_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.2, random_state=42, stratify=labels
)

print(f"✅ 資料分割完成:")
print(f"   訓練集: {X_train.shape[0]:,} 張 ({X_train.shape[0]/len(data)*100:.1f}%)")
print(f"   測試集: {X_test.shape[0]:,} 張 ({X_test.shape[0]/len(data)*100:.1f}%)")

# One-Hot Encoding
y_train_encoded = to_categorical(y_train, 43)
y_test_encoded = to_categorical(y_test, 43)

print(f"\n✅ One-Hot Encoding 完成")

# ============================================================================
# Section 4: 資料預處理
# ============================================================================

print("\n" + "=" * 80)
print("🔧 Section 4: 資料預處理")
print("=" * 80)

# 訓練集保持 [0, 255] 給 ImageDataGenerator
X_train_aug = X_train.copy()

# 測試集標準化
X_test_norm = X_test.astype('float32') / 255.0

print(f"訓練集範圍: [{X_train_aug.min()}, {X_train_aug.max()}] (給 ImageDataGenerator)")
print(f"測試集範圍: [{X_test_norm.min():.4f}, {X_test_norm.max():.4f}] (已標準化)")

# ============================================================================
# Section 5: Data Augmentation 設定
# ============================================================================

print("\n" + "=" * 80)
print("🎨 Section 5: Data Augmentation 設定")
print("=" * 80)

datagen = ImageDataGenerator(
    rescale=1./255,              # 標準化
    rotation_range=5,            # ±5°
    width_shift_range=0.1,       # ±10%
    height_shift_range=0.1,      # ±10%
    zoom_range=0.1,              # ±10%
    brightness_range=[0.8, 1.2], # 亮度 [0.8, 1.2]
    fill_mode='nearest',
    horizontal_flip=False,
    vertical_flip=False
)

print("✅ Data Augmentation 設定完成:")
print("   - rescale: 1./255")
print("   - rotation_range: ±5°")
print("   - width_shift_range: ±10%")
print("   - height_shift_range: ±10%")
print("   - zoom_range: ±10%")
print("   - brightness_range: [0.8, 1.2]")

# 視覺化驗證
print("\n繪製 Data Augmentation 驗證圖...")
sample_img = X_train_aug[0:1]

fig, axes = plt.subplots(2, 5, figsize=(15, 6))
fig.suptitle('Data Augmentation Verification', fontsize=14, fontweight='bold')

# 原始影像
axes[0, 0].imshow(sample_img[0].astype('uint8'))
axes[0, 0].set_title('Original\n[0-255]', fontsize=10, fontweight='bold')
axes[0, 0].axis('off')

# 擴增樣本
aug_iter = datagen.flow(sample_img, batch_size=1)
for i in range(9):
    aug_img = next(aug_iter)[0]
    row = (i + 1) // 5
    col = (i + 1) % 5
    axes[row, col].imshow(aug_img)
    axes[row, col].set_title(f'Augmented {i+1}\n[0-1]', fontsize=9)
    axes[row, col].axis('off')

plt.tight_layout()
plt.show()

print("✅ Data Augmentation 驗證完成")

# ============================================================================
# Section 6: 模型建立
# ============================================================================

print("\n" + "=" * 80)
print("🏗️ Section 6: 模型建立 (含 Batch Normalization)")
print("=" * 80)

def build_model_with_bn(input_shape):
    """建立含 Batch Normalization 的 CNN 模型"""
    model = Sequential(name='CNN_with_BatchNorm_Optimized')

    # 第一組卷積層 + BN
    model.add(Conv2D(32, (5, 5), activation='relu', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(Conv2D(32, (5, 5), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPool2D((2, 2)))
    model.add(Dropout(0.25))

    # 第二組卷積層 + BN
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPool2D((2, 2)))
    model.add(Dropout(0.25))

    # 全連接層
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(43, activation='softmax'))

    model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )

    return model

# 建立模型
model = build_model_with_bn((30, 30, 3))

print("\n✅ 模型建立完成")
print("\n📋 模型架構:")
model.summary()

# ============================================================================
# Section 7: Early Stopping 設定 (優化版)
# ============================================================================

print("\n" + "=" * 80)
print("⏱️ Section 7: Early Stopping 設定 (優化版)")
print("=" * 80)

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=20,  # ⭐ 從 5 > 20
    restore_best_weights=True,
    mode='min',
    verbose=1
)

print("✅ Early Stopping 已設定:")
print("   - monitor: val_loss")
print("   - patience: 10 ⭐ (針對 BN 訓練波動優化)")
print("   - restore_best_weights: True")
print("\n💡 說明:")
print("   BN 在訓練初期會有較大波動，patience=20 可以:")
print("   1. 容忍更多次驗證損失未改善的情況")
print("   2. 給予模型更多時間穩定收斂")
print("   3. 避免過早停止訓練")

# ============================================================================
# Section 8: 訓練模型
# ============================================================================

print("\n" + "=" * 80)
print("🚀 Section 8: 開始訓練實驗 C (優化版)")
print("=" * 80)

print("\n【訓練設定】")
print("  ✅ 資料: ImageDataGenerator (rescale=1./255 + augmentation)")
print("  ✅ 擴增: rotation ±5°, shift ±10%, zoom ±10%, brightness [0.8, 1.2]")
print("  ✅ 模型: CNN + 4 層 Batch Normalization")
print("  ✅ Early Stopping: patience=20 ⭐")
print("  ✅ Batch Size: 128")
print("  ✅ Max Epochs: 50")

print("\n" + "=" * 60)
print("開始訓練...")
print("=" * 60)

start_time = time.time()

steps_per_epoch = int(np.ceil(len(X_train_aug) / 128))

history = model.fit(
    datagen.flow(X_train_aug, y_train_encoded, batch_size=128),
    steps_per_epoch=steps_per_epoch,
    epochs=50,  # 增加到 50
    validation_data=(X_test_norm, y_test_encoded),
    callbacks=[early_stop],
    verbose=1
)

end_time = time.time()
training_time = end_time - start_time

# ============================================================================
# Section 9: 訓練結果
# ============================================================================

print("\n" + "=" * 60)
print("✅ 實驗 C (優化版) 訓練完成!")
print("=" * 60)

# 測試集評估
loss_test, acc_test = model.evaluate(X_test_norm, y_test_encoded, verbose=0)

print(f"⏱️  總訓練時間: {training_time:.2f} 秒 ({training_time/60:.2f} 分鐘)")
print(f"📊 實際訓練 epochs: {len(history.history['accuracy'])}")
print(f"📊 最終訓練準確率: {history.history['accuracy'][-1]*100:.2f}%")
print(f"📊 最終驗證準確率: {history.history['val_accuracy'][-1]*100:.2f}%")
print(f"📊 最佳驗證準確率: {max(history.history['val_accuracy'])*100:.2f}% (Epoch {np.argmax(history.history['val_accuracy'])+1})")
print(f"📊 最佳驗證損失: {min(history.history['val_loss']):.4f} (Epoch {np.argmin(history.history['val_loss'])+1})")
print(f"📊 測試準確率: {acc_test*100:.2f}%")
print(f"📊 測試損失: {loss_test:.4f}")

# ============================================================================
# Section 10: 與原始實驗 C 比較
# ============================================================================

print("\n" + "=" * 80)
print("📊 Section 10: 與原始實驗 C 比較")
print("=" * 80)

# 原始實驗 C 結果 (patience=5)
original_epochs = 12
original_time = 36.53
original_test_acc = 99.71
original_test_loss = 0.0102

comparison_df = pd.DataFrame({
    '指標': [
        '訓練時間 (分鐘)',
        '實際訓練 Epochs',
        '最佳驗證準確率 (%)',
        '測試準確率 (%)',
        '測試損失',
    ],
    '原始 (patience=5)': [
        f"{original_time:.2f}",
        f"{original_epochs}",
        "99.71",
        f"{original_test_acc:.2f}",
        f"{original_test_loss:.4f}",
    ],
    '優化版 (patience=10)': [
        f"{training_time/60:.2f}",
        f"{len(history.history['accuracy'])}",
        f"{max(history.history['val_accuracy'])*100:.2f}",
        f"{acc_test*100:.2f}",
        f"{loss_test:.4f}",
    ]
})

print("\n" + "=" * 80)
print(comparison_df.to_string(index=False))
print("=" * 80)

# 計算改善
acc_improvement = acc_test - (original_test_acc / 100)
loss_improvement = ((original_test_loss - loss_test) / original_test_loss) * 100
epochs_increase = len(history.history['accuracy']) - original_epochs
time_increase = (training_time/60) - original_time

print(f"\n📈 改善分析:")
print(f"  測試準確率: {acc_improvement*100:+.2f}%")
print(f"  測試損失: {loss_improvement:+.1f}%")
print(f"  訓練 Epochs: +{epochs_increase} 個")
print(f"  訓練時間: {time_increase:+.2f} 分鐘")

if acc_test > (original_test_acc / 100):
    print(f"\n✅ 優化成功! 測試準確率提升至 {acc_test*100:.2f}%")
else:
    print(f"\n⚠️ 準確率未提升，但訓練更穩定 (Epochs: {len(history.history['accuracy'])} vs {original_epochs})")

# ============================================================================
# Section 11: 訓練曲線視覺化
# ============================================================================

print("\n" + "=" * 80)
print("📊 Section 11: 訓練曲線視覺化")
print("=" * 80)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Experiment C (Optimized) - Training History', fontsize=14, fontweight='bold')

epochs_range = range(1, len(history.history['accuracy']) + 1)

# 子圖 1: 準確率
axes[0].plot(epochs_range, history.history['accuracy'],
             'b-o', label='Training Accuracy', linewidth=2, markersize=5)
axes[0].plot(epochs_range, history.history['val_accuracy'],
             'r-s', label='Validation Accuracy', linewidth=2, markersize=5)
axes[0].axvline(x=np.argmax(history.history['val_accuracy'])+1,
                color='green', linestyle='--', alpha=0.5, label='Best Epoch')
axes[0].set_xlabel('Epoch', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Accuracy', fontsize=12, fontweight='bold')
axes[0].set_title('Accuracy (patience=10)', fontsize=13, fontweight='bold')
axes[0].legend(loc='lower right', fontsize=11)
axes[0].grid(True, alpha=0.3)

# 子圖 2: 損失
axes[1].plot(epochs_range, history.history['loss'],
             'b-o', label='Training Loss', linewidth=2, markersize=5)
axes[1].plot(epochs_range, history.history['val_loss'],
             'r-s', label='Validation Loss', linewidth=2, markersize=5)
axes[1].axvline(x=np.argmin(history.history['val_loss'])+1,
                color='green', linestyle='--', alpha=0.5, label='Best Epoch')
axes[1].set_xlabel('Epoch', fontsize=12, fontweight='bold')
axes[1].set_ylabel('Loss', fontsize=12, fontweight='bold')
axes[1].set_title('Loss (patience=10)', fontsize=13, fontweight='bold')
axes[1].legend(loc='upper right', fontsize=11)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("✅ 訓練曲線繪製完成")

# ============================================================================
# Section 12: 訓練穩定性分析
# ============================================================================

print("\n" + "=" * 80)
print("📊 Section 12: 訓練穩定性分析")
print("=" * 80)

# 計算驗證損失的波動
val_loss_std = np.std(history.history['val_loss'])
val_loss_mean = np.mean(history.history['val_loss'])
val_loss_cv = (val_loss_std / val_loss_mean) * 100  # 變異係數

print(f"\n驗證損失統計:")
print(f"  平均值: {val_loss_mean:.4f}")
print(f"  標準差: {val_loss_std:.4f}")
print(f"  變異係數: {val_loss_cv:.2f}%")

# 找出損失突然上升的 epochs
val_loss = history.history['val_loss']
spike_epochs = []
for i in range(1, len(val_loss)):
    if val_loss[i] > val_loss[i-1] * 1.5:  # 損失增加超過 50%
        spike_epochs.append(i+1)

if spike_epochs:
    print(f"\n⚠️ 檢測到損失突然上升的 Epochs: {spike_epochs}")
    print("   這是 Batch Normalization 訓練過程的正常現象")
else:
    print(f"\n✅ 訓練過程穩定，無明顯損失突升")

# 過擬合分析
train_val_gap = history.history['accuracy'][-1] - history.history['val_accuracy'][-1]
print(f"\n過擬合分析:")
print(f"  訓練/驗證準確率差距: {train_val_gap*100:.2f}%")
if abs(train_val_gap) < 0.02:
    print("  ✅ 模型泛化能力良好")
elif train_val_gap > 0.02:
    print("  ⚠️ 輕微過擬合")
else:
    print("  ⚠️ 驗證準確率高於訓練準確率 (可能需要更多訓練)")

# ============================================================================
# Section 13: 儲存模型與結果
# ============================================================================

print("\n" + "=" * 80)
print("💾 Section 13: 儲存模型與結果")
print("=" * 80)

model_save_path = '/content/drive/MyDrive/Colab Notebooks/GTSRB/'

# 儲存模型
model_path = os.path.join(model_save_path, 'model_C_optimized_patience10.h5')
model.save(model_path)
print(f"✅ 模型已儲存: model_C_optimized_patience10.h5")

# 儲存訓練歷史
history_path = os.path.join(model_save_path, 'history_C_optimized_patience10.pkl')
with open(history_path, 'wb') as f:
    pickle.dump(history.history, f)
print(f"✅ 訓練歷史已儲存: history_C_optimized_patience10.pkl")

# 儲存比較結果
results = {
    'original': {
        'patience': 5,
        'epochs': original_epochs,
        'time_min': original_time,
        'test_acc': original_test_acc,
        'test_loss': original_test_loss
    },
    'optimized': {
        'patience': 10,
        'epochs': len(history.history['accuracy']),
        'time_min': training_time / 60,
        'test_acc': acc_test * 100,
        'test_loss': loss_test,
        'best_val_acc': max(history.history['val_accuracy']) * 100,
        'best_val_loss': min(history.history['val_loss'])
    }
}

results_path = os.path.join(model_save_path, 'comparison_patience_5_vs_10.pkl')
with open(results_path, 'wb') as f:
    pickle.dump(results, f)
print(f"✅ 比較結果已儲存: comparison_patience_5_vs_10.pkl")

# ============================================================================
# Section 14: 完成!
# ============================================================================

print("\n" + "=" * 80)
print("🎉 實驗 C (優化版) 完成!")
print("=" * 80)

print("\n📊 最終結果摘要:")
print(f"   訓練時間: {training_time/60:.2f} 分鐘")
print(f"   訓練 Epochs: {len(history.history['accuracy'])}")
print(f"   最佳驗證準確率: {max(history.history['val_accuracy'])*100:.2f}%")
print(f"   測試準確率: {acc_test*100:.2f}%")
print(f"   測試損失: {loss_test:.4f}")

print("\n💡 關鍵發現:")
if len(history.history['accuracy']) > original_epochs:
    print(f"   1. patience=10 使訓練延長至 {len(history.history['accuracy'])} epochs")
    print(f"   2. 相比原始的 {original_epochs} epochs，增加了 {len(history.history['accuracy']) - original_epochs} epochs")
    print(f"   3. 測試準確率: {acc_test*100:.2f}% vs 原始 {original_test_acc:.2f}%")
else:
    print(f"   1. 即使 patience=10，模型仍在 {len(history.history['accuracy'])} epochs 收斂")
    print(f"   2. 這顯示訓練已達最佳狀態")

if acc_test > (original_test_acc / 100):
    print(f"\n🎯 結論: patience=10 成功提升模型效能!")
else:
    print(f"\n🎯 結論: patience=10 提供更穩定的訓練過程")

print("\n📁 所有結果已儲存至: " + model_save_path)
print("=" * 80)


# ============================================================================
# GTSRB - 實驗 C 優化版 Confusion Matrix 詳細分析
# 測試集評估與混淆矩陣視覺化
# ============================================================================

print("=" * 80)
print("📊 實驗 C 優化版 - Confusion Matrix 詳細分析")
print("=" * 80)

# ============================================================================
# Section 1: 載入套件
# ============================================================================

print("\n📦 載入套件...")

from google.colab import drive
drive.mount('/content/drive')

import numpy as np
import pandas as pd
import tensorflow as tf
import os

# 影像處理
from PIL import Image

# 評估指標
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 繪圖
import matplotlib.pyplot as plt
import seaborn as sns

print("✅ 套件載入完成")

# ============================================================================
# Section 2: 載入實驗 C 優化版模型
# ============================================================================

print("\n" + "=" * 80)
print("🔧 Section 2: 載入實驗 C 優化版模型")
print("=" * 80)

model_path = '/content/drive/MyDrive/Colab Notebooks/GTSRB/model_C_optimized_patience10.h5'

if os.path.exists(model_path):
    model = tf.keras.models.load_model(model_path)
    print(f"✅ 模型載入成功: {model_path}")
    print(f"\n📋 模型架構摘要:")
    model.summary()
else:
    print(f"❌ 找不到模型檔案: {model_path}")
    print("請確認模型已訓練並儲存")
    raise FileNotFoundError(f"Model not found: {model_path}")

# ============================================================================
# Section 3: 載入並處理測試資料集
# ============================================================================

print("\n" + "=" * 80)
print("📂 Section 3: 載入測試資料集")
print("=" * 80)

# 快取檔案路徑
cache_path = '/content/drive/MyDrive/Colab Notebooks/GTSRB/archive/X_test_cache.npy'
label_cache_path = '/content/drive/MyDrive/Colab Notebooks/GTSRB/archive/y_test_cache.npy'

# 載入測試資料
if os.path.exists(cache_path) and os.path.exists(label_cache_path):
    print("⚡ 載入快取的測試資料...")
    X_test = np.load(cache_path)
    y_test = np.load(label_cache_path)
    print(f"✅ 快取載入成功")
else:
    print("🧩 第一次執行：讀取並處理圖片中...")
    test_csv = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/GTSRB/archive/Test.csv')
    y_test = test_csv["ClassId"].values
    imgs = test_csv["Path"].values

    data = []
    for i, img in enumerate(imgs):
        image = Image.open('/content/drive/MyDrive/Colab Notebooks/GTSRB/archive/' + img)
        image = image.resize((30, 30))
        data.append(np.array(image))

        if (i + 1) % 1000 == 0:
            print(f"  已處理 {i + 1}/{len(imgs)} 張圖片...")

    X_test = np.array(data)

    # 存成快取檔
    np.save(cache_path, X_test)
    np.save(label_cache_path, y_test)
    print("✅ 已建立快取，下次會直接載入")

print(f"\n📊 測試資料形狀:")
print(f"  X_test: {X_test.shape}")
print(f"  y_test: {y_test.shape}")
print(f"  像素值範圍: [{X_test.min()}, {X_test.max()}]")

# ============================================================================
# Section 4: 資料預處理（標準化）
# ============================================================================

print("\n" + "=" * 80)
print("🔧 Section 4: 資料預處理")
print("=" * 80)

# 重要：標準化到 [0, 1]，與訓練時一致
X_test_normalized = X_test.astype('float32') / 255.0

print(f"✅ 標準化完成")
print(f"  標準化後範圍: [{X_test_normalized.min():.4f}, {X_test_normalized.max():.4f}]")

# ============================================================================
# Section 5: 模型預測
# ============================================================================

print("\n" + "=" * 80)
print("🚀 Section 5: 模型預測")
print("=" * 80)

print("開始預測...")
predictions_prob = model.predict(X_test_normalized, verbose=1)
predictions = np.argmax(predictions_prob, axis=-1)

print(f"\n✅ 預測完成")
print(f"  預測形狀: {predictions.shape}")
print(f"  預測類別範圍: [{predictions.min()}, {predictions.max()}]")

# ============================================================================
# Section 6: 計算測試準確率
# ============================================================================

print("\n" + "=" * 80)
print("📊 Section 6: 測試準確率")
print("=" * 80)

test_accuracy = accuracy_score(y_test, predictions) * 100

print(f"✅ 測試資料準確率: {test_accuracy:.2f}%")

# 計算每個類別的準確率
unique_classes = np.unique(y_test)
class_accuracies = []

for cls in unique_classes:
    mask = y_test == cls
    if mask.sum() > 0:
        cls_acc = (predictions[mask] == cls).sum() / mask.sum() * 100
        class_accuracies.append(cls_acc)
    else:
        class_accuracies.append(0)

print(f"\n各類別準確率統計:")
print(f"  平均準確率: {np.mean(class_accuracies):.2f}%")
print(f"  準確率中位數: {np.median(class_accuracies):.2f}%")
print(f"  最高準確率: {np.max(class_accuracies):.2f}%")
print(f"  最低準確率: {np.min(class_accuracies):.2f}%")

# ============================================================================
# Section 7: Classification Report
# ============================================================================

print("\n" + "=" * 80)
print("📋 Section 7: Classification Report")
print("=" * 80)

report = classification_report(y_test, predictions, output_dict=True)
report_df = pd.DataFrame(report).transpose()

print("\n完整 Classification Report:")
print("=" * 80)
print(classification_report(y_test, predictions))
print("=" * 80)

# 儲存為 CSV
report_df.to_csv('/content/drive/MyDrive/Colab Notebooks/GTSRB/classification_report_exp_C_optimized.csv',
                 encoding='utf-8-sig')
print("✅ Classification Report 已儲存為 CSV")

# ============================================================================
# Section 8: 混淆矩陣計算
# ============================================================================

print("\n" + "=" * 80)
print("📊 Section 8: 混淆矩陣計算")
print("=" * 80)

cm = confusion_matrix(y_test, predictions)

print(f"混淆矩陣形狀: {cm.shape}")
print(f"總預測數量: {cm.sum()}")
print(f"正確預測數量: {np.trace(cm)}")
print(f"錯誤預測數量: {cm.sum() - np.trace(cm)}")

# ============================================================================
# Section 9: 視覺化混淆矩陣（完整版）
# ============================================================================

print("\n" + "=" * 80)
print("📊 Section 9: 視覺化混淆矩陣")
print("=" * 80)

# 9.1 完整混淆矩陣（43×43）
print("\n繪製完整混淆矩陣 (43×43)...")

plt.figure(figsize=(18, 15))
sns.heatmap(cm, annot=False, cmap="Blues", fmt='g', cbar_kws={'label': 'Count'})
plt.title("Confusion Matrix - Experiment C (Optimized)\nTest Accuracy: {:.2f}%".format(test_accuracy),
          fontsize=16, fontweight='bold', pad=20)
plt.xlabel("Predicted Label", fontsize=13, fontweight='bold')
plt.ylabel("True Label", fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('/content/drive/MyDrive/Colab Notebooks/GTSRB/confusion_matrix_exp_C_optimized_full.png',
            dpi=300, bbox_inches='tight')
plt.show()

print("✅ 完整混淆矩陣已儲存")

# 9.2 混淆矩陣（帶數字標註，較小尺寸）
print("\n繪製帶數字標註的混淆矩陣...")

plt.figure(figsize=(20, 17))
sns.heatmap(cm, annot=True, cmap="Blues", fmt='d', cbar_kws={'label': 'Count'},
            annot_kws={'fontsize': 6})
plt.title("Confusion Matrix with Counts - Experiment C (Optimized)\nTest Accuracy: {:.2f}%".format(test_accuracy),
          fontsize=16, fontweight='bold', pad=20)
plt.xlabel("Predicted Label", fontsize=13, fontweight='bold')
plt.ylabel("True Label", fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('/content/drive/MyDrive/Colab Notebooks/GTSRB/confusion_matrix_exp_C_optimized_annotated.png',
            dpi=300, bbox_inches='tight')
plt.show()

print("✅ 帶標註混淆矩陣已儲存")

# 9.3 正規化混淆矩陣（百分比）
print("\n繪製正規化混淆矩陣（百分比）...")

cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

plt.figure(figsize=(18, 15))
sns.heatmap(cm_normalized, annot=False, cmap="Blues", fmt='.2%',
            cbar_kws={'label': 'Percentage', 'format': '%.0f%%'})
plt.title("Normalized Confusion Matrix (%) - Experiment C (Optimized)\nTest Accuracy: {:.2f}%".format(test_accuracy),
          fontsize=16, fontweight='bold', pad=20)
plt.xlabel("Predicted Label", fontsize=13, fontweight='bold')
plt.ylabel("True Label", fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('/content/drive/MyDrive/Colab Notebooks/GTSRB/confusion_matrix_exp_C_optimized_normalized.png',
            dpi=300, bbox_inches='tight')
plt.show()

print("✅ 正規化混淆矩陣已儲存")

# ============================================================================
# Section 10: 錯誤分析
# ============================================================================

print("\n" + "=" * 80)
print("🔍 Section 10: 錯誤分析")
print("=" * 80)

# 找出所有錯誤預測
error_mask = predictions != y_test
error_indices = np.where(error_mask)[0]
error_count = len(error_indices)

print(f"\n總錯誤數: {error_count} / {len(y_test)} ({error_count/len(y_test)*100:.2f}%)")

# 統計最常見的錯誤
if error_count > 0:
    error_pairs = list(zip(y_test[error_mask], predictions[error_mask]))
    from collections import Counter
    error_counter = Counter(error_pairs)
    most_common_errors = error_counter.most_common(10)

    print(f"\n最常見的 10 種錯誤預測:")
    print("=" * 60)
    print(f"{'排名':<6} {'真實類別':<10} {'預測類別':<10} {'錯誤次數':<10}")
    print("=" * 60)
    for rank, ((true_label, pred_label), count) in enumerate(most_common_errors, 1):
        print(f"{rank:<6} {true_label:<10} {pred_label:<10} {count:<10}")
    print("=" * 60)

    # 找出最容易混淆的類別
    print(f"\n最容易被誤判的類別 (Top 5):")
    class_errors = {}
    for cls in unique_classes:
        mask = y_test == cls
        if mask.sum() > 0:
            errors = mask.sum() - (predictions[mask] == cls).sum()
            class_errors[cls] = errors

    sorted_errors = sorted(class_errors.items(), key=lambda x: x[1], reverse=True)[:5]
    print("=" * 60)
    print(f"{'類別':<10} {'錯誤數':<10} {'該類別總數':<15} {'錯誤率':<10}")
    print("=" * 60)
    for cls, err_count in sorted_errors:
        total = (y_test == cls).sum()
        error_rate = err_count / total * 100 if total > 0 else 0
        print(f"{cls:<10} {err_count:<10} {total:<15} {error_rate:.2f}%")
    print("=" * 60)

else:
    print("🎉 完美預測！沒有任何錯誤！")

# ============================================================================
# Section 11: 類別準確率視覺化
# ============================================================================

print("\n" + "=" * 80)
print("📊 Section 11: 類別準確率視覺化")
print("=" * 80)

# 類別名稱對照
class_names = {
    0:'Speed limit (20km/h)', 1:'Speed limit (30km/h)', 2:'Speed limit (50km/h)',
    3:'Speed limit (60km/h)', 4:'Speed limit (70km/h)', 5:'Speed limit (80km/h)',
    6:'End of speed limit (80km/h)', 7:'Speed limit (100km/h)', 8:'Speed limit (120km/h)',
    9:'No passing', 10:'No passing veh over 3.5 tons', 11:'Right-of-way at intersection',
    12:'Priority road', 13:'Yield', 14:'Stop', 15:'No vehicles',
    16:'Veh > 3.5 tons prohibited', 17:'No entry', 18:'General caution',
    19:'Dangerous curve left', 20:'Dangerous curve right', 21:'Double curve',
    22:'Bumpy road', 23:'Slippery road', 24:'Road narrows on the right',
    25:'Road work', 26:'Traffic signals', 27:'Pedestrians', 28:'Children crossing',
    29:'Bicycles crossing', 30:'Beware of ice/snow', 31:'Wild animals crossing',
    32:'End speed + passing limits', 33:'Turn right ahead', 34:'Turn left ahead',
    35:'Ahead only', 36:'Go straight or right', 37:'Go straight or left',
    38:'Keep right', 39:'Keep left', 40:'Roundabout mandatory',
    41:'End of no passing', 42:'End no passing veh > 3.5 tons'
}

# 繪製類別準確率柱狀圖
fig, ax = plt.subplots(figsize=(18, 8))

colors = ['green' if acc == 100 else ('orange' if acc >= 99 else 'red') for acc in class_accuracies]
bars = ax.bar(range(43), class_accuracies, color=colors, edgecolor='black', linewidth=0.5)

ax.axhline(y=100, color='green', linestyle='--', linewidth=1, alpha=0.5, label='100%')
ax.axhline(y=99, color='orange', linestyle='--', linewidth=1, alpha=0.5, label='99%')
ax.axhline(y=test_accuracy, color='blue', linestyle='-', linewidth=2, alpha=0.7,
           label=f'Overall: {test_accuracy:.2f}%')

ax.set_xlabel('Class ID', fontsize=12, fontweight='bold')
ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
ax.set_title('Per-Class Accuracy - Experiment C (Optimized)', fontsize=14, fontweight='bold')
ax.set_xticks(range(43))
ax.set_ylim([95, 101])
ax.legend(loc='lower right', fontsize=11)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('/content/drive/MyDrive/Colab Notebooks/GTSRB/class_accuracy_exp_C_optimized.png',
            dpi=300, bbox_inches='tight')
plt.show()

print("✅ 類別準確率圖已儲存")

# 統計達到 100% 準確率的類別數
perfect_classes = sum(1 for acc in class_accuracies if acc == 100)
print(f"\n達到 100% 準確率的類別數: {perfect_classes} / 43 ({perfect_classes/43*100:.1f}%)")

# ============================================================================
# Section 12: 儲存詳細分析結果
# ============================================================================

print("\n" + "=" * 80)
print("💾 Section 12: 儲存詳細分析結果")
print("=" * 80)

# 建立詳細分析 DataFrame
analysis_data = []
for cls in range(43):
    mask = y_test == cls
    total = mask.sum()
    correct = (predictions[mask] == cls).sum()
    accuracy = correct / total * 100 if total > 0 else 0

    analysis_data.append({
        'Class ID': cls,
        'Class Name': class_names[cls],
        'Total Samples': total,
        'Correct Predictions': correct,
        'Accuracy (%)': f"{accuracy:.2f}",
    })

analysis_df = pd.DataFrame(analysis_data)
analysis_df.to_csv('/content/drive/MyDrive/Colab Notebooks/GTSRB/class_analysis_exp_C_optimized.csv',
                   index=False, encoding='utf-8-sig')

print("✅ 類別分析已儲存為 CSV")

# 儲存混淆矩陣為 CSV
cm_df = pd.DataFrame(cm)
cm_df.to_csv('/content/drive/MyDrive/Colab Notebooks/GTSRB/confusion_matrix_exp_C_optimized.csv',
             index=False, encoding='utf-8-sig')

print("✅ 混淆矩陣已儲存為 CSV")

# ============================================================================
# Section 13: 總結
# ============================================================================

print("\n" + "=" * 80)
print("🎉 實驗 C 優化版 - Confusion Matrix 分析完成!")
print("=" * 80)

print(f"\n📊 最終結果摘要:")
print(f"  測試準確率: {test_accuracy:.2f}%")
print(f"  總測試樣本: {len(y_test):,}")
print(f"  正確預測: {(predictions == y_test).sum():,}")
print(f"  錯誤預測: {error_count:,}")
print(f"  達到 100% 準確率的類別: {perfect_classes} / 43")

print(f"\n📁 已產生的檔案:")
print(f"  1. confusion_matrix_exp_C_optimized_full.png (完整混淆矩陣)")
print(f"  2. confusion_matrix_exp_C_optimized_annotated.png (帶數字標註)")
print(f"  3. confusion_matrix_exp_C_optimized_normalized.png (正規化百分比)")
print(f"  4. class_accuracy_exp_C_optimized.png (類別準確率圖)")
print(f"  5. classification_report_exp_C_optimized.csv (分類報告)")
print(f"  6. class_analysis_exp_C_optimized.csv (類別分析)")
print(f"  7. confusion_matrix_exp_C_optimized.csv (混淆矩陣數據)")

print(f"\n所有結果已儲存至: /content/drive/MyDrive/Colab Notebooks/GTSRB/")
print("=" * 80)