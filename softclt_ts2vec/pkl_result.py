import pickle
import os

# 1. 定义实验配置列表 (只需在这里添加新运行的文件夹名)
# 格式: (文件夹标识符, 显示名称)
experiments = [
    ("TEMP0_INST0", "Baseline (TS2Vec, tau=0)"),
    ("TEMP1_INST1_tau_temp0.5_tau_inst0.5", "SoftCLT (tau_temp0.5_tau_inst0.5)"),
    ("TEMP1_INST1_tau_temp1.0_tau_inst1.0", "SoftCLT (tau_temp1.0_tau_inst1.0)"),
    ("TEMP1_INST1_tau_temp1.5_tau_inst2.0", "SoftCLT (tau_temp1.5_tau_inst2.0)"),
    ("TEMP1_INST1_tau_temp1.0_tau_inst1.0", "SoftCLT (tau_temp1.0_tau_inst1.0,BS16)"), # 需注意路径中的 bs16
    ("TEMP1_INST1_tau_temp5.0_tau_inst10.0", "SoftCLT (tau_temp5.0_tau_inst10.0)"),
    ("TEMP1_INST1_tau_temp2.0_tau_inst20.0", "SoftCLT (tau_temp2.0_tau_inst20.0)")
]

# 2. 基础路径配置
base_dir = "results_classification_UCR"
dataset = "ECG200"
run_id = "run2" # 确保和你跑出的 run 编号对应

def get_acc_auprc(temp_folder, bs="bs8"):
    """自动化路径拼接与数据读取"""
    filepath = os.path.join(base_dir, temp_folder, dataset, bs, run_id, "eval_res_DTW.pkl")
    try:
        with open(filepath, 'rb') as f:
            res = pickle.load(f)
            acc = res.get('acc', 0.0)
            auprc = res.get('auprc', 0.0)
            return f"{acc:.4f}", f"{auprc:.4f}"
    except FileNotFoundError:
        return None, None

# 3. 打印学术实验表格
print("=" * 85)
print(f"Summary of Experiment Results (Dataset: {dataset})")
print("-" * 85)
print(f"{'Method / Hyperparameters':<45} |  Accuracy  |   AUPRC")
print("-" * 85)

for folder, name in experiments:
    # 特殊处理 BS16 的路径
    current_bs = "bs16" if "BS=16" in name else "bs8"
    acc, auprc = get_acc_auprc(folder, current_bs)
    
    if acc:
        print(f"{name:<45} |   {acc}   |   {auprc}")
    else:
        print(f"{name:<45} |   [Data Not Found]   ")

print("=" * 85)