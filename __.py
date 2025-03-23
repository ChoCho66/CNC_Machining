import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from sklearn.model_selection import KFold


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
X_bad = X_data1[bad_indices][:,:,-1]

# 生成範例資料
np.random.seed(42)
X = np.vstack([X_good, X_bad])
y = np.hstack([np.zeros(len(X_good)), np.ones(len(X_bad))])  # 0: 正常, 1: 異常

# k-fold cross-validation 設置
k_folds = 5
kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)

# 儲存評估結果
results = {"Autoencoder": [], "IsolationForest": [], "OneClassSVM": []}

# 定義評估函數
def evaluate_model(y_true, y_pred, y_score):
    roc_auc = roc_auc_score(y_true, y_score)
    pr_auc = average_precision_score(y_true, y_score)
    y_pred_bin = (y_score > 0.5).astype(int)  # 二元化
    f1 = f1_score(y_true, y_pred_bin)
    return roc_auc, pr_auc, f1

# 1. Autoencoder
def train_autoencoder(X_train, X_test):
    input_dim = X_train.shape[1]
    encoding_dim = 128  # 壓縮到 128 維
    # 定義 Autoencoder 模型
    autoencoder = models.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(512, activation='relu'),
        layers.Dense(encoding_dim, activation='relu'),
        layers.Dense(512, activation='relu'),
        layers.Dense(input_dim, activation='linear')
    ])
    autoencoder.compile(optimizer='adam', loss='mse')

    # 訓練模型
    autoencoder.fit(X_train, X_train, epochs=10, batch_size=32, verbose=0)

    # 計算重建誤差
    reconstructed = autoencoder.predict(X_test)
    reconstruction_error = np.mean((X_test - reconstructed) ** 2, axis=1)
    return reconstruction_error

# 2. Isolation Forest
def train_isolation_forest(X_train, X_test):
    clf = IsolationForest(contamination=len(X_bad) / len(X), random_state=42)
    clf.fit(X_train)
    scores = -clf.decision_function(X_test)  # 負的 decision_function 越大越異常
    return scores

# 3. One-Class SVM
def train_one_class_svm(X_train, X_test):
    ocsvm = OneClassSVM(kernel="rbf", nu=0.05, gamma="scale")
    ocsvm.fit(X_train)
    scores = -ocsvm.decision_function(X_test)  # 負的 decision_function 越大越異常
    return scores

# k-fold cross-validation
for train_idx, test_idx in kf.split(X_good):
    X_train, X_test = X_good[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # Autoencoder
    ae_scores = train_autoencoder(X_train, X_test)
    ae_metrics = evaluate_model(y_test, ae_scores > np.percentile(ae_scores, 95), ae_scores)
    results["Autoencoder"].append(ae_metrics)

    # Isolation Forest
    if_scores = train_isolation_forest(X_train, X_test)
    if_metrics = evaluate_model(y_test, if_scores > np.percentile(if_scores, 95), if_scores)
    results["IsolationForest"].append(if_metrics)

    # One-Class SVM
    ocsvm_scores = train_one_class_svm(X_train, X_test)
    ocsvm_metrics = evaluate_model(y_test, ocsvm_scores > np.percentile(ocsvm_scores, 95), ocsvm_scores)
    results["OneClassSVM"].append(ocsvm_metrics)
    
print()
print()
print()
print()
print(results)

# 計算平均分數
for model_name, metrics in results.items():
    metrics = np.array(metrics)
    print(f"{model_name} 平均結果:")
    print(f"- ROC AUC: {metrics[:, 0].mean():.4f}")
    print(f"- PR AUC: {metrics[:, 1].mean():.4f}")
    print(f"- F1-Score: {metrics[:, 2].mean():.4f}")
    print()