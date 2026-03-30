import pickle

# 定义文件路径
file_baseline = "results_classification_UCR/TEMP0_INST0/ECG200/bs8/run2/eval_res_DTW.pkl"
file_softclt = "results_classification_UCR/TEMP1_INST1_tau_temp0.5_tau_inst0.5/ECG200/bs8/run2/eval_res_DTW.pkl"

def load_and_print(filepath, method_name):
    try:
        with open(filepath, 'rb') as f:
            res = pickle.load(f)
            # 学术规范：保留四位小数进行对比
            acc = f"{res.get('acc', 0.0):.4f}"
            auprc = f"{res.get('auprc', 0.0):.4f}"
            # 格式化输出，保证对齐
            print(f"{method_name:<35} | Accuracy: {acc} | AUPRC: {auprc}")
    except FileNotFoundError:
        print(f"{method_name:<35} | Status: File Not Found")

# 打印严谨的纯文本实验表格
print("-" * 80)
print("Experiment Results (Dataset: ECG200)")
print("-" * 80)
load_and_print(file_baseline, "Baseline (TS2Vec, tau=0)")
load_and_print(file_softclt, "Proposed Method (SoftCLT, tau=0.5)")
print("-" * 80)