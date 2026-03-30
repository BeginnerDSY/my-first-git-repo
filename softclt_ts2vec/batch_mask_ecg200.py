import pandas as pd
import numpy as np
import neurokit2 as nk
import sys
import os

def print_now(text):
    print(text)
    sys.stdout.flush()

# 1. 确认文件路径 (这里以 UCR 格式的 TEST 集为例)
# 如果你是在 AutoDL 的 softclt_ts2vec 目录下运行，这个路径是直接生效的
file_path = "datasets/UCR/ECG200/ECG200_TEST.tsv"

if not os.path.exists(file_path):
    print_now(f"!!! 找不到文件: {file_path}，请确认你在这个文件的上级目录运行！")
    sys.exit()

print_now(">>> 1. 正在读取 ECG200 原始数据...")
# UCR 的 tsv 文件：第 0 列是标签(患病/正常)，后面的列全是波形数据
df = pd.read_csv(file_path, sep='\t', header=None)
labels = df.iloc[:, 0].values
signals = df.iloc[:, 1:].values  # 形状应该是 (100, 96)

num_samples, seq_len = signals.shape
print_now(f"   -> 成功加载 {num_samples} 条数据，每条长度 {seq_len} 个点。")

# 2. 准备我们的“装小抄的麻袋” (初始化一个全 0 的矩阵矩阵)
all_masks = np.zeros((num_samples, seq_len), dtype=int)

print_now("\n>>> 2. 启动流水线，开始批量生成 Phase Mask...")
success_count = 0

# 3. 开始无情地循环遍历
for i in range(num_samples):
    # 取出第 i 条心电图
    current_signal = signals[i]
    
    # 穿上防弹装甲 (try-except)，防止 NeuroKit2 因为数据太短而崩溃
    try:
        # UCR 数据没有真实的采样率，我们强行假设它是 100Hz 骗过 NeuroKit2
        fake_fs = 100 
        
        # 清洗 + 找峰 + 切割边界
        clean_sig = nk.ecg_clean(current_signal, sampling_rate=fake_fs)
        _, r_info = nk.ecg_peaks(clean_sig, sampling_rate=fake_fs)
        
        # 如果连一个 R 峰都找不到，直接放弃这条数据
        if len(r_info["ECG_R_Peaks"]) == 0:
            raise ValueError("No R-peaks found")

        _, waves_info = nk.ecg_delineate(clean_sig, r_info["ECG_R_Peaks"], 
                                         sampling_rate=fake_fs, method="dwt")
        
        # 涂色环节
        for start, end in zip(waves_info["ECG_P_Onsets"], waves_info["ECG_P_Offsets"]):
            if not np.isnan(start) and not np.isnan(end): all_masks[i, int(start):int(end)] = 1
        for start, end in zip(waves_info["ECG_R_Onsets"], waves_info["ECG_R_Offsets"]):
            if not np.isnan(start) and not np.isnan(end): all_masks[i, int(start):int(end)] = 2
        for start, end in zip(waves_info["ECG_T_Onsets"], waves_info["ECG_T_Offsets"]):
            if not np.isnan(start) and not np.isnan(end): all_masks[i, int(start):int(end)] = 3
            
        success_count += 1
        
    except Exception as e:
        # 如果崩溃了，啥也不干，保留全 0，静默跳过 (为了不刷屏，只在后台记录)
        pass

    # 打印个简单的进度条
    if (i + 1) % 20 == 0:
        print_now(f"   -> 已处理 {i+1}/{num_samples} 条...")

print_now(f"\n>>> 流水线运行完毕！成功打上标签的数量: {success_count}/{num_samples}")
print_now("   (注：失败的样本已被自动填充为全 0 基线)")

# 4. 将整个矩阵永久落盘！
out_filename = "ECG200_TEST_masks.npy"
np.save(out_filename, all_masks)
print_now(f">>> 3. 文件保存为: {out_filename}")
print_now(f"   -> 矩阵形状: {all_masks.shape}")