import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping


# ==================== 1. 生成/加载数据（支持真实数据替换） ====================
def generate_driving_data(n_samples=10000):
    """生成真实感无人车驾驶数据：速度、加速度、转向角、时间特征"""
    time = np.linspace(0, 100, n_samples)
    speed = np.zeros(n_samples)
    acceleration = np.zeros(n_samples)
    steering_angle = np.zeros(n_samples)

    # 模拟驾驶场景：起步→加速→匀速→减速→转弯→加速→停车
    segment1 = slice(0, 1500)
    segment2 = slice(1500, 4000)
    segment3 = slice(4000, 5000)
    segment4 = slice(5000, 6500)
    segment5 = slice(6500, 8500)
    segment6 = slice(8500, 10000)

    # 填充数据（含噪声）
    speed[segment1] = np.linspace(0, 15, len(segment1)) + np.random.normal(0, 0.3, len(segment1))
    acceleration[segment1] = 0.8 + np.random.normal(0, 0.1, len(segment1))

    speed[segment2] = 15 + np.random.normal(0, 0.2, len(segment2))
    acceleration[segment2] = np.random.normal(0, 0.05, len(segment2))

    speed[segment3] = np.linspace(15, 5, len(segment3)) + np.random.normal(0, 0.3, len(segment3))
    acceleration[segment3] = -0.6 + np.random.normal(0, 0.1, len(segment3))

    speed[segment4] = 5 + np.random.normal(0, 0.2, len(segment4))
    steering_angle[segment4] = np.linspace(-15, 15, len(segment4)) + np.random.normal(0, 0.5, len(segment4))

    speed[segment5] = np.linspace(5, 20, len(segment5)) + np.random.normal(0, 0.4, len(segment5))
    acceleration[segment5] = 0.9 + np.random.normal(0, 0.1, len(segment5))

    speed[segment6] = np.linspace(20, 0, len(segment6)) + np.random.normal(0, 0.4, len(segment6))
    acceleration[segment6] = -1.2 + np.random.normal(0, 0.15, len(segment6))

    # 构造时间特征
    timestamp = pd.date_range(start='2024-01-01', periods=n_samples, freq='10ms')
    hour = timestamp.hour
    dayofweek = timestamp.dayofweek

    # 整理数据框
    data = pd.DataFrame({
        'speed': speed.clip(0, 25),
        'acceleration': acceleration.clip(-2, 2),
        'steering_angle': steering_angle.clip(-30, 30),
        'hour': hour,
        'dayofweek': dayofweek
    })
    return data


# 生成数据（替换为 pd.read_csv("your_real_data.csv") 可使用真实数据）
data = generate_driving_data()
print("数据形状：", data.shape)
print("数据前5行：\n", data.head())


# ==================== 2. 数据预处理（时序序列转换） ====================
def create_sequences(data, seq_length=10, target_step=1):
    """将数据转换为LSTM输入格式：(样本数, 序列长度, 特征数)"""
    # 归一化特征
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    X, y = [], []
    # 生成输入序列和目标（预测未来target_step步的速度）
    for i in range(seq_length, len(scaled_data) - target_step + 1):
        X.append(scaled_data[i - seq_length:i, :])
        y.append(scaled_data[i:i + target_step, 0])  # 速度是第0列

    return np.array(X), np.array(y), scaler


# 超参数
seq_length = 20  # 输入序列长度（历史20个时间步）
target_step = 3  # 预测未来3个时间步的速度
features = data.shape[1]  # 特征数

# 创建序列
X, y, scaler = create_sequences(data, seq_length, target_step)
print(f"\n输入序列形状：{X.shape}")
print(f"输出序列形状：{y.shape}")

# 划分训练集/测试集（时序数据不打乱）
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
print(f"训练集：X={X_train.shape}, y={y_train.shape}")
print(f"测试集：X={X_test.shape}, y={y_test.shape}")

# ==================== 3. 构建LSTM模型 ====================
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(seq_length, features)),
    Dropout(0.2),
    LSTM(32, return_sequences=False),
    Dropout(0.2),
    Dense(target_step)
])

model.compile(optimizer='adam', loss='mse')
model.summary()

# ==================== 4. 训练模型（含早停机制） ====================
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

history = model.fit(
    X_train, y_train,
    batch_size=32,
    epochs=30,
    validation_data=(X_test, y_test),
    callbacks=[early_stopping],
    verbose=1
)


# ==================== 5. 预测与反归一化 ====================
def inverse_transform_speed(pred, scaler):
    """反归一化速度预测结果"""
    dummy = np.zeros((pred.shape[0], pred.shape[1], features))
    dummy[:, :, 0] = pred
    inv_pred = scaler.inverse_transform(dummy.reshape(-1, features))[:, 0].reshape(pred.shape)
    return inv_pred


# 预测
y_train_pred = model.predict(X_train, verbose=0)
y_test_pred = model.predict(X_test, verbose=0)

# 反归一化
y_train_true = inverse_transform_speed(y_train, scaler)
y_train_pred = inverse_transform_speed(y_train_pred, scaler)
y_test_true = inverse_transform_speed(y_test, scaler)
y_test_pred = inverse_transform_speed(y_test_pred, scaler)

# ==================== 6. 模型评估与可视化 ====================
# 损失曲线
plt.figure(figsize=(12, 8))
plt.subplot(2, 1, 1)
plt.plot(history.history['loss'], label='训练损失', color='#2E86AB')
plt.plot(history.history['val_loss'], label='验证损失', color='#A23B72')
plt.xlabel('Epoch')
plt.ylabel('MSE损失')
plt.title('训练损失曲线')
plt.legend()
plt.grid(alpha=0.3)

# 速度预测对比
plt.subplot(2, 1, 2)
plt.plot(y_test_true[:500, 0], label='真实速度', color='#F18F01')
plt.plot(y_test_pred[:500, 0], label='预测速度', color='#C73E1D', alpha=0.8)
plt.xlabel('时间步')
plt.ylabel('速度 (m/s)')
plt.title(f'速度预测结果（预测未来{target_step}步，展示第1步）')
plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plt.show()

# 量化评估
mae = mean_absolute_error(y_test_true[:, 0], y_test_pred[:, 0])
mse = mean_squared_error(y_test_true[:, 0], y_test_pred[:, 0])
rmse = np.sqrt(mse)
r2 = r2_score(y_test_true[:, 0], y_test_pred[:, 0])

print("\n模型评估结果：")
print(f"MAE: {mae:.4f} m/s")
print(f"MSE: {mse:.4f}")
print(f"RMSE: {rmse:.4f} m/s")
print(f"R²: {r2:.4f}")

# 保存模型
model.save("unmanned_vehicle_speed_model.h5")
print("\n模型已保存为：unmanned_vehicle_speed_model.h5")