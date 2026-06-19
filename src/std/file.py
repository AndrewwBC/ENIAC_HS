import json
import numpy as np
from pathlib import Path
from base import BASE_DIR

def collect_fold_metrics(checkpoints_dir):
    macro_f1s = []
    hs_f1s = []

    for fold in range(5):
        metrics_path = Path(checkpoints_dir) / f"outer_fold_{fold}" / "metrics.json"
        with open(metrics_path) as f:
            data = json.load(f)

        report = data["test_evaluation_report"]
        macro_f1s.append(report["macro avg"]["f1-score"])
        hs_f1s.append(report["Hate Speech"]["f1-score"])

    return np.array(macro_f1s), np.array(hs_f1s)


for name, path in [
    ("LLM Augmentation + BERT", BASE_DIR / "checkpoints_aug"),
    ("BERT (Imbalanced)",        BASE_DIR / "checkpoints_original"),
]:
    macro_f1s, hs_f1s = collect_fold_metrics(path)
    print(f"\n{name}")
    print(f"  Macro F1:       {np.mean(macro_f1s):.4f} ± {np.std(macro_f1s):.4f}")
    print(f"  Hate Speech F1: {np.mean(hs_f1s):.4f} ± {np.std(hs_f1s):.4f}")
