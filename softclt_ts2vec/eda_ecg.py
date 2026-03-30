import pandas as pd
import matplotlib.pyplot as plt

# 1. 定位并读取刚才千辛万苦下载的真实数据集
file_path = "datasets/UCR/ECG200/ECG200_TRAIN.tsv"
print(f"🔍 正在读取真实数据: {file_path}")

# UCR格式的数据集没有表头，且是用制表符 '\t' 分隔的
df = pd.read_csv(file_path, sep='\t', header=None)

# 2. 看看这个数据的全貌
print(f"📊 数据集的形状是: {df.shape} (代表有 {df.shape[0]} 个样本，每个样本有 {df.shape[1]} 列)")
print("💡 提示：第一列是疾病标签(1正常，-1异常)，后面的列是真实的电压采样点！")

# 3. 提取第一号病人的心电信号（去掉第一列的标签）
real_ecg_signal = df.iloc[0, 1:].values
label = df.iloc[0, 0]

# 4. 把这根真实的信号画出来
plt.figure(figsize=(10, 4))
plt.plot(real_ecg_signal, color='crimson', linewidth=2)
plt.title(f"Real ECG Signal from ECG200 (Patient Label: {label})")
plt.xlabel("Time Steps")
plt.ylabel("Voltage")
plt.grid(True, linestyle='--', alpha=0.7)

# 5. 在云端保存图片
save_name = "real_ecg_step1.png"
plt.savefig(save_name, dpi=300, bbox_inches='tight')
print(f"✅ 图片已成功保存为: {save_name}，快去左侧文件栏下载查看吧！")