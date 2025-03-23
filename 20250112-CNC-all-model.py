# %% [markdown]
# ## 讀取 Data，只取第一段5秒，且不 scaling

# %%
import os
from pathlib import Path
from utils import data_loader_utils
import itertools 

# %%
import pickle

# load X_raw_data.pkl, y_raw_data.pkl
with open("X_raw_data.pkl", "rb") as file:
    X_raw_data = pickle.load(file)

with open("y_raw_data.pkl", "rb") as file:
    y_raw_data = pickle.load(file)

# %%
time_lengths = [x.shape[0]//2000 for x in X_raw_data]

# %%
num_bad, num_good = [1 if "good" in i else 0 for i in y_raw_data].count(0), [1 if "good" in i else 0 for i in y_raw_data].count(1)

# %%
import numpy as np
import matplotlib.pyplot as plt

def plot_data(vibration_data, color='b'):
    freq = 2000
    samples_s = len(vibration_data[:, 0]) / freq
    samples = np.linspace(0, samples_s, len(vibration_data[:, 0]))

    # plotting
    plt.figure(figsize=(20, 2))
    plt.plot(samples, vibration_data[:, 0], color)
    plt.ylabel('X-axis Vibration Data')
    plt.xlabel('Time [sec]')
    plt.locator_params(axis='y', nbins=10)
    plt.grid()
    plt.show()
    plt.figure(figsize=(20, 2))
    plt.plot(samples, vibration_data[:, 1], color)
    plt.ylabel('Y-axis Vibration Data')
    plt.xlabel('Time [sec]')
    plt.locator_params(axis='y', nbins=10)
    plt.grid()
    plt.show()
    plt.figure(figsize=(20, 2))
    plt.plot(samples, vibration_data[:, 2], color)
    plt.ylabel('Z-axis Vibration Data')
    plt.xlabel('Time [sec]')
    plt.locator_params(axis='y', nbins=10)
    plt.grid()
    plt.show()

# %%
good_indices = [i for i, path in enumerate(y_raw_data) if "good" in path]
bad_indices = [i for i, path in enumerate(y_raw_data) if "bad" in path]

# %%
for arr in X_raw_data:
    arr[:, 2] += 1000

# %%
import numpy as np

def process_time_series_with_overlap(data, sample_rate=2000, skip_seconds=5, segment_seconds=5, overlap_seconds=2):
    """
    處理 time series data，去除開頭資料並切分成固定長度、具有 overlap 的片段。
    
    :param data: numpy array, shape = (length, 3)
    :param sample_rate: 採樣率, 預設 2000 Hz
    :param skip_seconds: 開頭跳過的秒數, 預設 5 秒
    :param segment_seconds: 每段切分的秒數, 預設 5 秒
    :param overlap_seconds: 每段之間的重疊秒數, 預設 2 秒
    :return: 切分後的資料列表，每個元素為 numpy array
    """
    # 計算需要跳過的資料行數
    skip_rows = skip_seconds * sample_rate
    # 計算每段的行數
    segment_rows = segment_seconds * sample_rate
    # 計算每次移動的步數 (segment_rows - overlap_rows)
    overlap_rows = overlap_seconds * sample_rate
    step_rows = segment_rows - overlap_rows

    # 去掉開頭的資料
    data = data[skip_rows:]
    
    # 切分資料，加入 overlap
    segments = [data[i:i + segment_rows] for i in range(0, len(data) - segment_rows + 1, step_rows)]
    
    return segments[0:1]
    # return segments

# %%
X_data = []
y_data = []
SAMPLE_RATE = 2000 # Hz
SKIP_SECONDS = 5 # sec
TIME_LENGTH = 5 # sec
OVERLAP_SECONDS = 2 # sec
# TIME_LENGTH = 10 # sec
# 
for i in range(len(X_raw_data)):
    X_data.extend(process_time_series_with_overlap(X_raw_data[i], SAMPLE_RATE, SKIP_SECONDS, TIME_LENGTH, OVERLAP_SECONDS))
    if 'good' in y_raw_data[i]:
        y_data.extend([0] * len(process_time_series_with_overlap(X_raw_data[i], SAMPLE_RATE, SKIP_SECONDS, TIME_LENGTH, OVERLAP_SECONDS)))
    else:
        y_data.extend([1] * len(process_time_series_with_overlap(X_raw_data[i], SAMPLE_RATE, SKIP_SECONDS, TIME_LENGTH, OVERLAP_SECONDS)))

# %%
del X_raw_data

# %%
X_data1 = np.array(X_data)

# %%
X_data_abs = np.abs(X_data1)
# X_data_abs[:4]

# %%
x_abs_max, y_abs_max, z_abs_max = X_data_abs[:,:,0].max(), X_data_abs[:,:,1].max(), X_data_abs[:,:,2].max()
print(x_abs_max, y_abs_max, z_abs_max)

# %%
good_indices = [index for index, value in enumerate(y_data) if value == 0]
bad_indices = [index for index, value in enumerate(y_data) if value == 1]

# %% [markdown]
# ## 取 z-axis time domain

# %%
X_good = X_data1[good_indices][:,:,-1]
X_good.shape

# %%
X_bad = X_data1[bad_indices][:,:,-1]
X_bad.shape

# %%
SEED = 42

# %%
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, roc_auc_score

# 劃分正常資料為訓練集與測試集
X_good_train, X_good_test = train_test_split(X_good, test_size= len(X_bad) / len(X_good), random_state=SEED)

# %%
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt

tf.random.set_seed(SEED) 
np.random.seed(SEED)

# AutoEncoder 模型
input_dim = X_good_train.shape[1]
encoding_dim = 128

autoencoder = models.Sequential([
    layers.Input(shape=(input_dim,)),
    layers.Dense(512, activation='relu'),
    layers.Dense(encoding_dim, activation='relu'),
    layers.Dense(512, activation='relu'),
    layers.Dense(input_dim, activation='linear')
])

autoencoder.compile(optimizer='adam', loss='mse')
autoencoder.summary()

# %%
# 計算重建誤差 (MSE) 作為異常分數
def calculate_reconstruction_error(model, data):
    reconstructed = model.predict(data)
    print(data.shape)
    # print(reconstructed.shape)
    mse = np.mean(np.square(data - reconstructed), axis=1)
    return mse

# %%
import matplotlib.pyplot as plt

mse_good_train = calculate_reconstruction_error(autoencoder, X_good_train)
mse_good_test = calculate_reconstruction_error(autoencoder, X_good_test)
mse_bad = calculate_reconstruction_error(autoencoder, X_bad)

plt.figure(figsize=(8, 3))
plt.hist(mse_good_train, bins=50, alpha=0.5, label='X_good_train')
plt.hist(mse_good_test, bins=50, alpha=0.3, label='X_good_test')
plt.hist(mse_bad, bins=50, alpha=0.5, label='X_bad')
plt.legend()
plt.xlabel('Reconstruction Error')
plt.ylabel('Frequency')
plt.show()

# %%
import matplotlib.pyplot as plt

mse_good_train = calculate_reconstruction_error(autoencoder, X_good_train)
mse_good_test = calculate_reconstruction_error(autoencoder, X_good_test)
# mse_bad = calculate_reconstruction_error(autoencoder, X_bad)

plt.figure(figsize=(8, 3))
plt.hist(mse_good_train, bins=500, alpha=0.5, label='X_good_train')
plt.hist(mse_good_test, bins=100, alpha=0.3, label='X_good_test')
# plt.hist(mse_bad, bins=500, alpha=0.5, label='X_bad')
plt.legend()
plt.xlabel('Reconstruction Error')
plt.ylabel('Frequency')
plt.show()

# %%
import matplotlib.pyplot as plt

# mse_good_train = calculate_reconstruction_error(autoencoder, X_good_train)
mse_good_test = calculate_reconstruction_error(autoencoder, X_good_test)
mse_bad = calculate_reconstruction_error(autoencoder, X_bad)

plt.figure(figsize=(8, 3))
# plt.hist(mse_good_train, bins=50, alpha=0.5, label='X_good_train')
plt.hist(mse_good_test, bins=50, alpha=0.3, label='X_good_test')
plt.hist(mse_bad, bins=500, alpha=0.5, label='X_bad')
plt.legend()
plt.xlabel('Reconstruction Error')
plt.ylabel('Frequency')
plt.show()

# %% [markdown]
# ## Before training

# %%
import matplotlib.pyplot as plt
import seaborn as sns

# 計算重建誤差
mse_good_train = calculate_reconstruction_error(autoencoder, X_good_train)
mse_good_test = calculate_reconstruction_error(autoencoder, X_good_test)
mse_bad = calculate_reconstruction_error(autoencoder, X_bad)

# 只顯示 KDE 曲線
plt.figure(figsize=(8, 3))
sns.kdeplot(mse_good_train, color="blue", label='X_good_train', fill=True, alpha=0.5)
sns.kdeplot(mse_good_test, color="orange", label='X_good_test', fill=True, alpha=0.5)
sns.kdeplot(mse_bad, color="red", label='X_bad', fill=True, alpha=0.5)

# 圖表標籤與顯示
plt.xlabel('Reconstruction Error')
plt.ylabel('Density')
plt.legend()
plt.grid()
plt.show()

# %%
len(X_good_test), len(X_bad)

# %%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay

# 假設你已經有 X_good_test 和 X_bad
X_test = np.concatenate([X_good_test, X_bad])  # 組合測試集
X_test_labels = np.concatenate([np.zeros(len(X_good_test)), np.ones(len(X_bad))])  # 標註正常和異常資料

# 計算重建誤差
re_error_X_test = calculate_reconstruction_error(autoencoder, X_test)

# 計算 ROC 曲線的 FPR, TPR 和閾值
fpr, tpr, thresholds = roc_curve(X_test_labels, re_error_X_test)

# 計算 AUC
auc = roc_auc_score(X_test_labels, re_error_X_test)
print(f"AUC: {auc:.4f}")

# 繪製 ROC 曲線
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {auc:.4f})")
plt.plot([0, 1], [0, 1], 'k--', label="Random Guess")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.grid(True)
plt.title("ROC Curve")
plt.show()

# 找到最佳閾值，通常是 ROC 曲線中最靠近 (0, 1) 點的閾值
optimal_idx = np.argmax(tpr - fpr)
optimal_threshold = thresholds[optimal_idx]
print(f"Optimal Threshold: {optimal_threshold:.4f}")

# 使用最佳閾值將異常分數轉換為二元分類結果
predictions = (re_error_X_test > optimal_threshold).astype(int)

# 計算混淆矩陣
cm = confusion_matrix(X_test_labels, predictions)

# 可視化混淆矩陣
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Normal", "Anomaly"])
disp.plot(cmap=plt.cm.Blues)
plt.title(f"Confusion Matrix at Optimal Threshold ({optimal_threshold:.4f})")
plt.show()

# # 打印混淆矩陣數值
# print("Confusion Matrix:")
# print(cm)

# 取得混淆矩陣中的各項數值
TN, FP, FN, TP = cm.ravel()

# 計算 TPR、FPR、Precision 和 Recall
TPR = TP / (TP + FN)  # True Positive Rate
FPR = FP / (FP + TN)  # False Positive Rate
Precision = TP / (TP + FP)  # Precision
Recall = TP / (TP + FN)  # Recall

# 打印指標
print(f"TPR (True Positive Rate): {TPR:.4f}")
print(f"FPR (False Positive Rate): {FPR:.4f}")
print(f"Precision: {Precision:.4f}")
print(f"Recall: {Recall:.4f}")

# %%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, auc, confusion_matrix, ConfusionMatrixDisplay

# 假設你已經有 X_test_labels 和 re_error_X_test
precision, recall, thresholds = precision_recall_curve(X_test_labels, re_error_X_test)

# 計算 AUC (Precision-Recall AUC)
pr_auc = auc(recall, precision)

# 繪製 Precision-Recall Curve
plt.figure(figsize=(8, 6))
plt.plot(recall, precision, label=f"PR Curve (AUC = {pr_auc:.4f})")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.legend()
plt.grid(True)
plt.title("Precision-Recall Curve")
plt.show()

# 計算 F1 score
f1_scores = 2 * (precision * recall) / (precision + recall)

# 找到最佳閾值 (最佳 F1 score 對應的閾值)
best_threshold_index = np.argmax(f1_scores)
best_threshold = thresholds[best_threshold_index]

# 使用最佳閾值來計算預測類別
y_pred = (re_error_X_test >= best_threshold).astype(int)

# 計算混淆矩陣
cm = confusion_matrix(X_test_labels, y_pred)

# 可視化混淆矩陣
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
disp.plot(cmap=plt.cm.Blues)
plt.title(f"Confusion Matrix (Best Threshold = {best_threshold:.2f})")
plt.show()

# # 打印混淆矩陣數值
# print(f"Confusion Matrix at Best Threshold ({best_threshold:.2f}):")
# print(cm)

# 取得混淆矩陣中的各項數值
TN, FP, FN, TP = cm.ravel()

# 計算 TPR、FPR、Precision 和 Recall
TPR = TP / (TP + FN)  # True Positive Rate
FPR = FP / (FP + TN)  # False Positive Rate
Precision = TP / (TP + FP)  # Precision
Recall = TP / (TP + FN)  # Recall

# 打印指標
print(f"TPR (True Positive Rate): {TPR:.4f}")
print(f"FPR (False Positive Rate): {FPR:.4f}")
print(f"Precision: {Precision:.4f}")
print(f"Recall: {Recall:.4f}")

# %% [markdown]
# ## Training 1

# %%
from tensorflow.keras.callbacks import EarlyStopping

# 定義 Early Stopping
early_stop = EarlyStopping(
    monitor='val_loss',  # 監控的指標（可以是 'val_loss', 'loss', 或其他 metric）
    patience=5,          # 容忍幾個 epoch 無改善
    verbose=1,           # 是否顯示早停的訊息
    restore_best_weights=True  # 是否恢復為最佳權重
)

# %%
# 訓練 AutoEncoder (僅使用正常資料)
# history = autoencoder.fit(X_good, X_good, epochs=300, batch_size=32, shuffle=True, validation_split=0.2)
# 加入 Early Stopping 回調並訓練模型
history = autoencoder.fit(
    X_good_train, 
    X_good_train, 
    epochs=50, 
    batch_size=16, 
    shuffle=True, 
    validation_split=0.2, 
    callbacks=[early_stop]  # 加入 Early Stopping
)

# %%
import matplotlib.pyplot as plt

train_loss = history.history['loss']       # 訓練損失
val_loss = history.history['val_loss']    # 驗證損失

# 繪製訓練與驗證損失
plt.figure(figsize=(9, 5))
plt.plot(train_loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss', linestyle='--')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss Curve')
plt.legend()
plt.grid(True)
plt.show()

# %%
# 計算重建誤差 (MSE) 作為異常分數
def calculate_reconstruction_error(model, data):
    reconstructed = model.predict(data, verbose=0)
    print(data.shape)
    mse = np.mean(np.square(data - reconstructed), axis=1)
    return mse

# %%
import matplotlib.pyplot as plt
import seaborn as sns

# 計算重建誤差
mse_good_train = calculate_reconstruction_error(autoencoder, X_good_train)
mse_good_test = calculate_reconstruction_error(autoencoder, X_good_test)
mse_bad = calculate_reconstruction_error(autoencoder, X_bad)

# 只顯示 KDE 曲線
plt.figure(figsize=(8, 3))
sns.kdeplot(mse_good_train, color="blue", label='X_good_train', fill=True, alpha=0.5)
sns.kdeplot(mse_good_test, color="orange", label='X_good_test', fill=True, alpha=0.5)
sns.kdeplot(mse_bad, color="red", label='X_bad', fill=True, alpha=0.5)

# 圖表標籤與顯示
plt.xlabel('Reconstruction Error')
plt.ylabel('Density')
plt.legend()
plt.grid()
plt.show()

# %%
len(X_good_test), len(X_bad)

# %%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay

# 假設你已經有 X_good_test 和 X_bad
X_test = np.concatenate([X_good_test, X_bad])  # 組合測試集
X_test_labels = np.concatenate([np.zeros(len(X_good_test)), np.ones(len(X_bad))])  # 標註正常和異常資料

# 計算重建誤差
re_error_X_test = calculate_reconstruction_error(autoencoder, X_test)

# 計算 ROC 曲線的 FPR, TPR 和閾值
fpr, tpr, thresholds = roc_curve(X_test_labels, re_error_X_test)

# 計算 AUC
auc = roc_auc_score(X_test_labels, re_error_X_test)
print(f"AUC: {auc:.4f}")

# 繪製 ROC 曲線
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {auc:.4f})")
plt.plot([0, 1], [0, 1], 'k--', label="Random Guess")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.grid(True)
plt.title("ROC Curve")
plt.show()

# 找到最佳閾值，通常是 ROC 曲線中最靠近 (0, 1) 點的閾值
optimal_idx = np.argmax(tpr - fpr)
optimal_threshold = thresholds[optimal_idx]
print(f"Optimal Threshold: {optimal_threshold:.4f}")

# 使用最佳閾值將異常分數轉換為二元分類結果
predictions = (re_error_X_test > optimal_threshold).astype(int)

# 計算混淆矩陣
cm = confusion_matrix(X_test_labels, predictions)

# 可視化混淆矩陣
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Normal", "Anomaly"])
disp.plot(cmap=plt.cm.Blues)
plt.title(f"Confusion Matrix at Optimal Threshold ({optimal_threshold:.4f})")
plt.show()

# # 打印混淆矩陣數值
# print("Confusion Matrix:")
# print(cm)
# 取得混淆矩陣中的各項數值
TN, FP, FN, TP = cm.ravel()

# 計算 TPR、FPR、Precision 和 Recall
TPR = TP / (TP + FN)  # True Positive Rate
FPR = FP / (FP + TN)  # False Positive Rate
Precision = TP / (TP + FP)  # Precision
Recall = TP / (TP + FN)  # Recall

# 打印指標
print(f"TPR (True Positive Rate): {TPR:.4f}")
print(f"FPR (False Positive Rate): {FPR:.4f}")
print(f"Precision: {Precision:.4f}")
print(f"Recall: {Recall:.4f}")

# %%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, auc, confusion_matrix, ConfusionMatrixDisplay

# 假設你已經有 X_test_labels 和 re_error_X_test
precision, recall, thresholds = precision_recall_curve(X_test_labels, re_error_X_test)

# 計算 AUC (Precision-Recall AUC)
pr_auc = auc(recall, precision)

# 繪製 Precision-Recall Curve
plt.figure(figsize=(8, 6))
plt.plot(recall, precision, label=f"PR Curve (AUC = {pr_auc:.4f})")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.legend()
plt.grid(True)
plt.title("Precision-Recall Curve")
plt.show()

# 計算 F1 score
f1_scores = 2 * (precision * recall) / (precision + recall)

# 找到最佳閾值 (最佳 F1 score 對應的閾值)
best_threshold_index = np.argmax(f1_scores)
best_threshold = thresholds[best_threshold_index]

# 使用最佳閾值來計算預測類別
y_pred = (re_error_X_test >= best_threshold).astype(int)

# 計算混淆矩陣
cm = confusion_matrix(X_test_labels, y_pred)

# 可視化混淆矩陣
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
disp.plot(cmap=plt.cm.Blues)
plt.title(f"Confusion Matrix (Best Threshold = {best_threshold:.2f})")
plt.show()

# # 打印混淆矩陣數值
# print(f"Confusion Matrix at Best Threshold ({best_threshold:.2f}):")
# print(cm)
# 取得混淆矩陣中的各項數值
TN, FP, FN, TP = cm.ravel()

# 計算 TPR、FPR、Precision 和 Recall
TPR = TP / (TP + FN)  # True Positive Rate
FPR = FP / (FP + TN)  # False Positive Rate
Precision = TP / (TP + FP)  # Precision
Recall = TP / (TP + FN)  # Recall

# 打印指標
print(f"TPR (True Positive Rate): {TPR:.4f}")
print(f"FPR (False Positive Rate): {FPR:.4f}")
print(f"Precision: {Precision:.4f}")
print(f"Recall: {Recall:.4f}")

# %% [markdown]
# ## Training 2

# %%
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping

tf.random.set_seed(SEED) 
np.random.seed(SEED)

# AutoEncoder 模型
input_dim = X_good_train.shape[1]
encoding_dim = 128

autoencoder = models.Sequential([
    layers.Input(shape=(input_dim,)),
    layers.Dense(512, activation='relu'),
    layers.Dense(encoding_dim, activation='relu'),
    layers.Dense(512, activation='relu'),
    layers.Dense(input_dim, activation='linear')
])

autoencoder.compile(optimizer='adam', loss='mse')
autoencoder.summary()

# 定義 Early Stopping
early_stop = EarlyStopping(
    monitor='val_loss',  # 監控的指標（可以是 'val_loss', 'loss', 或其他 metric）
    patience=10,          # 容忍幾個 epoch 無改善
    verbose=1,           # 是否顯示早停的訊息
    restore_best_weights=True  # 是否恢復為最佳權重
)

# 訓練 AutoEncoder (僅使用正常資料)
# history = autoencoder.fit(X_good, X_good, epochs=300, batch_size=32, shuffle=True, validation_split=0.2)
# 加入 Early Stopping 回調並訓練模型
history = autoencoder.fit(
    X_good_train, 
    X_good_train, 
    epochs=50, 
    batch_size=16, 
    shuffle=True, 
    validation_split=0.2, 
    callbacks=[early_stop]  # 加入 Early Stopping
)

# %%
import matplotlib.pyplot as plt

train_loss = history.history['loss']       # 訓練損失
val_loss = history.history['val_loss']    # 驗證損失

# 繪製訓練與驗證損失
plt.figure(figsize=(9, 5))
plt.plot(train_loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss', linestyle='--')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss Curve')
plt.legend()
plt.grid(True)
plt.show()

# %%
# 計算重建誤差 (MSE) 作為異常分數
def calculate_reconstruction_error(model, data):
    reconstructed = model.predict(data, verbose=0)
    print(data.shape)
    mse = np.mean(np.square(data - reconstructed), axis=1)
    return mse

# %%
import matplotlib.pyplot as plt
import seaborn as sns

# 計算重建誤差
mse_good_train = calculate_reconstruction_error(autoencoder, X_good_train)
mse_good_test = calculate_reconstruction_error(autoencoder, X_good_test)
mse_bad = calculate_reconstruction_error(autoencoder, X_bad)

# 只顯示 KDE 曲線
plt.figure(figsize=(8, 3))
sns.kdeplot(mse_good_train, color="blue", label='X_good_train', fill=True, alpha=0.5)
sns.kdeplot(mse_good_test, color="orange", label='X_good_test', fill=True, alpha=0.5)
sns.kdeplot(mse_bad, color="red", label='X_bad', fill=True, alpha=0.5)

# 圖表標籤與顯示
plt.xlabel('Reconstruction Error')
plt.ylabel('Density')
plt.legend()
plt.grid()
plt.show()

# %%
len(X_good_test), len(X_bad)

# %%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay

# 假設你已經有 X_good_test 和 X_bad
X_test = np.concatenate([X_good_test, X_bad])  # 組合測試集
X_test_labels = np.concatenate([np.zeros(len(X_good_test)), np.ones(len(X_bad))])  # 標註正常和異常資料

# 計算重建誤差
re_error_X_test = calculate_reconstruction_error(autoencoder, X_test)

# 計算 ROC 曲線的 FPR, TPR 和閾值
fpr, tpr, thresholds = roc_curve(X_test_labels, re_error_X_test)

# 計算 AUC
auc = roc_auc_score(X_test_labels, re_error_X_test)
print(f"AUC: {auc:.4f}")

# 繪製 ROC 曲線
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {auc:.4f})")
plt.plot([0, 1], [0, 1], 'k--', label="Random Guess")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.grid(True)
plt.title("ROC Curve")
plt.show()

# 找到最佳閾值，通常是 ROC 曲線中最靠近 (0, 1) 點的閾值
optimal_idx = np.argmax(tpr - fpr)
optimal_threshold = thresholds[optimal_idx]
print(f"Optimal Threshold: {optimal_threshold:.4f}")

# 使用最佳閾值將異常分數轉換為二元分類結果
predictions = (re_error_X_test > optimal_threshold).astype(int)

# 計算混淆矩陣
cm = confusion_matrix(X_test_labels, predictions)

# 可視化混淆矩陣
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Normal", "Anomaly"])
disp.plot(cmap=plt.cm.Blues)
plt.title(f"Confusion Matrix at Optimal Threshold ({optimal_threshold:.4f})")
plt.show()

# # 打印混淆矩陣數值
# print("Confusion Matrix:")
# print(cm)
# 取得混淆矩陣中的各項數值
TN, FP, FN, TP = cm.ravel()

# 計算 TPR、FPR、Precision 和 Recall
TPR = TP / (TP + FN)  # True Positive Rate
FPR = FP / (FP + TN)  # False Positive Rate
Precision = TP / (TP + FP)  # Precision
Recall = TP / (TP + FN)  # Recall

# 打印指標
print(f"TPR (True Positive Rate): {TPR:.4f}")
print(f"FPR (False Positive Rate): {FPR:.4f}")
print(f"Precision: {Precision:.4f}")
print(f"Recall: {Recall:.4f}")

# %%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, auc, confusion_matrix, ConfusionMatrixDisplay

# 假設你已經有 X_test_labels 和 re_error_X_test
precision, recall, thresholds = precision_recall_curve(X_test_labels, re_error_X_test)

# 計算 AUC (Precision-Recall AUC)
pr_auc = auc(recall, precision)

# 繪製 Precision-Recall Curve
plt.figure(figsize=(8, 6))
plt.plot(recall, precision, label=f"PR Curve (AUC = {pr_auc:.4f})")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.legend()
plt.grid(True)
plt.title("Precision-Recall Curve")
plt.show()

# 計算 F1 score
f1_scores = 2 * (precision * recall) / (precision + recall)

# 找到最佳閾值 (最佳 F1 score 對應的閾值)
best_threshold_index = np.argmax(f1_scores)
best_threshold = thresholds[best_threshold_index]

# 使用最佳閾值來計算預測類別
y_pred = (re_error_X_test >= best_threshold).astype(int)

# 計算混淆矩陣
cm = confusion_matrix(X_test_labels, y_pred)

# 可視化混淆矩陣
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
disp.plot(cmap=plt.cm.Blues)
plt.title(f"Confusion Matrix (Best Threshold = {best_threshold:.2f})")
plt.show()

# # 打印混淆矩陣數值
# print(f"Confusion Matrix at Best Threshold ({best_threshold:.2f}):")
# print(cm)
# 取得混淆矩陣中的各項數值
TN, FP, FN, TP = cm.ravel()

# 計算 TPR、FPR、Precision 和 Recall
TPR = TP / (TP + FN)  # True Positive Rate
FPR = FP / (FP + TN)  # False Positive Rate
Precision = TP / (TP + FP)  # Precision
Recall = TP / (TP + FN)  # Recall

# 打印指標
print(f"TPR (True Positive Rate): {TPR:.4f}")
print(f"FPR (False Positive Rate): {FPR:.4f}")
print(f"Precision: {Precision:.4f}")
print(f"Recall: {Recall:.4f}")

# %%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix, f1_score

# 假設你已經有 X_good_test 和 X_bad
X_test = np.concatenate([X_good_test, X_bad])  # 組合測試集
X_test_labels = np.concatenate([np.zeros(len(X_good_test)), np.ones(len(X_bad))])  # 標註正常和異常資料

# 計算重建誤差
re_error_X_test = calculate_reconstruction_error(autoencoder, X_test)

# 計算 ROC 曲線的 FPR, TPR 和閾值
fpr, tpr, thresholds = roc_curve(X_test_labels, re_error_X_test)

# 計算 AUC
auc = roc_auc_score(X_test_labels, re_error_X_test)
print(f"AUC: {auc:.4f}")

# 計算每個閾值下的 F1 分數
f1_scores = []
for threshold in thresholds:
    predictions = (re_error_X_test > threshold).astype(int)
    f1 = f1_score(X_test_labels, predictions)
    f1_scores.append(f1)

# 繪製 F1-Score 曲線
plt.figure(figsize=(8, 6))
plt.plot(thresholds, f1_scores, label="F1-Score Curve", color="blue")
plt.xlabel("Threshold")
plt.ylabel("F1-Score")
plt.title("F1-Score vs Threshold")
plt.legend()
plt.grid(True)
plt.show()

# 找到最佳閾值
optimal_idx = np.argmax(f1_scores)
optimal_threshold = thresholds[optimal_idx]
best_f1_score = f1_scores[optimal_idx]
print(f"Best F1-Score: {best_f1_score:.4f} at Threshold: {optimal_threshold:.4f}")

# 使用最佳閾值將異常分數轉換為二元分類結果
predictions = (re_error_X_test > optimal_threshold).astype(int)

# 計算混淆矩陣
from sklearn.metrics import ConfusionMatrixDisplay
cm = confusion_matrix(X_test_labels, predictions)

# 可視化混淆矩陣
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Normal", "Anomaly"])
disp.plot(cmap=plt.cm.Blues)
plt.title(f"Confusion Matrix at Optimal Threshold ({optimal_threshold:.4f})")
plt.show()

# 取得混淆矩陣中的各項數值
TN, FP, FN, TP = cm.ravel()

# 計算 TPR、FPR、Precision、Recall 和 F1-Score
TPR = TP / (TP + FN)  # True Positive Rate
FPR = FP / (FP + TN)  # False Positive Rate
Precision = TP / (TP + FP)  # Precision
Recall = TP / (TP + FN)  # Recall
final_f1_score = 2 * (Precision * Recall) / (Precision + Recall)  # F1-Score

# 打印指標
print(f"TPR (True Positive Rate): {TPR:.4f}")
print(f"FPR (False Positive Rate): {FPR:.4f}")
print(f"Precision: {Precision:.4f}")
print(f"Recall: {Recall:.4f}")
print(f"F1-Score (at Optimal Threshold): {final_f1_score:.4f}")

# %% [markdown]
# ## Final


