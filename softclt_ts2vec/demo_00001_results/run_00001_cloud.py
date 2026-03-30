import urllib.request
import os
import wfdb
import neurokit2 as nk
import matplotlib.pyplot as plt
import numpy as np
import sys

def print_now(text):
    print(text)
    sys.stdout.flush()

print_now("========== 云端 00001 号斩首行动 ==========")

# 1. 云端自动拉取数据 (因为之前你是在本地下的，云端还没有)
base_url = "https://physionet.org/files/ptb-xl/1.0.3/records500/00000/"
files_to_download = ["00001_hr.dat", "00001_hr.hea"]

print_now(">>> 1. 正在从 PhysioNet 拉取 00001 号真实临床数据...")
for file_name in files_to_download:
    if not os.path.exists(file_name):
        urllib.request.urlretrieve(base_url + file_name, file_name)
        print_now(f"   -> {file_name} 下载完成！")
    else:
        print_now(f"   -> {file_name} 已存在。")

# 2. 读取并处理数据
print_now("\n>>> 2. 启动 NeuroKit2 医学信号解析引擎...")
try:
    record = wfdb.rdrecord('00001_hr')
    sampling_rate = record.fs
    ecg_signal = record.p_signal[:2500, 1]  # 截取前 5 秒
    print_now(f"   -> 数据读取成功！长度: {len(ecg_signal)} 点")

    # 清洗 + 找峰 + 小波变换切割
    ecg_cleaned = nk.ecg_clean(ecg_signal, sampling_rate=sampling_rate)
    _, r_info = nk.ecg_peaks(ecg_cleaned, sampling_rate=sampling_rate)
    _, waves_info = nk.ecg_delineate(ecg_cleaned, r_info["ECG_R_Peaks"], 
                                     sampling_rate=sampling_rate, method="dwt")

    # 3. 涂色生成 Mask
    print_now(">>> 3. 正在生成 Phase Mask 黄金调料...")
    mask = np.zeros(len(ecg_cleaned), dtype=int)
    
    def fill_mask(onsets, offsets, label_val):
        for start, end in zip(onsets, offsets):
            if not np.isnan(start) and not np.isnan(end):
                mask[int(start):int(end)] = label_val

    fill_mask(waves_info["ECG_P_Onsets"], waves_info["ECG_P_Offsets"], 1)
    fill_mask(waves_info["ECG_R_Onsets"], waves_info["ECG_R_Offsets"], 2)
    fill_mask(waves_info["ECG_T_Onsets"], waves_info["ECG_T_Offsets"], 3)

    # 4. 永久落盘 (核心！)
    print_now(">>> 4. 正在保存特征矩阵与标签矩阵...")
    np.save('cloud_00001_cleaned_signal.npy', ecg_cleaned)
    np.save('cloud_00001_phase_mask.npy', mask)
    print_now("   -> [大捷] .npy 标签文件已成功落盘！")

    # 5. 云端专属绘图 (只存图，不弹窗)
    print_now(">>> 5. 正在渲染可视化战报...")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    ax1.plot(ecg_cleaned, color='black', linewidth=1.5)
    ax1.set_title("Cloud PTB-XL ECG Signal (Patient 00001, Cleaned Lead II)")
    ax1.grid(True, linestyle='--', alpha=0.6)
    
    ax2.plot(mask, color='blue', drawstyle='steps-pre', linewidth=2)
    ax2.set_title("Phase Mask (0=Baseline, 1=P, 2=QRS, 3=T)")
    ax2.set_yticks([0, 1, 2, 3])
    ax2.set_yticklabels(['0: Base', '1: P', '2: QRS', '3: T'])
    ax2.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig('cloud_00001_result.png', dpi=300)
    print_now("   -> [大捷] 战报图已保存为 cloud_00001_result.png！")
    print_now("========== 行动圆满结束 ==========")

except Exception as e:
    print_now(f"!!! 运行出错: {e}")