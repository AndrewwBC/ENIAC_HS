import json
import pandas as pd 
import torch
import optuna
import numpy as np

from base import BASE_DIR
from optuna import Trial
from pathlib import Path
from src.train_and_eval.file import TrainAndEval
from sklearn.model_selection import StratifiedKFold

from src.minabr.augmented.use_aug_data import train as train_data, test as test_data

results_dir = Path(BASE_DIR) / "results"
results_dir.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
skfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)    
CHECKPOINT = 'checkpoints_aug_mbert'

outer_results = []

def rebuild_hparams(best_params):
    return {
        "epochs": best_params["epochs"],
        "batch_size": best_params["batch_size"],
        "warmup_ratio": best_params["warmup_ratio"],
        "lr_scheduler": best_params["scheduler"],
        "bert": {
            "learning_rate": best_params["bert_lr"],
            "weight_decay":  best_params["bert_wd"],
        },
        "nn": {
            "learning_rate": best_params["nn_lr"],
            "weight_decay": best_params["nn_wd"],
            "dropout": best_params["nn_dropout"],
        }
    }

for outer_fold, (out_train_idx, out_test_idx) in enumerate(skfold.split(train_data, train_data['label'])):
    outer_train = train_data.iloc[out_train_idx]
    outer_test = train_data.iloc[out_test_idx]
    
    def objective(trial: Trial):
        hparams = {
            "epochs":       trial.suggest_int("epochs", 4, 8),
            "batch_size": trial.suggest_int("batch_size", 8, 32),
            "warmup_ratio": trial.suggest_float("warmup_ratio", 0.05, 0.15),
            "lr_scheduler": trial.suggest_categorical("scheduler", ["linear", "cosine"]),

            "bert": {
                "learning_rate": trial.suggest_float("bert_lr", 1e-5, 5e-5, log=True),
                "weight_decay":  trial.suggest_float("bert_wd", 1e-4, 1e-2, log=True),
            },

            "nn": {
                "learning_rate": trial.suggest_float("nn_lr", 1e-4, 5e-4, log=True),
                "weight_decay":  trial.suggest_float("nn_wd", 1e-4, 1e-2, log=True),
                "dropout":       trial.suggest_float("nn_dropout", 0.1, 0.3),
            }
        }
                
        inner_f1_scores = []
        
        for inner_fold, (inner_train_idx, inner_eval_idx) in enumerate(skfold.split(outer_train, outer_train['label'])):
            inner_train = outer_train.iloc[inner_train_idx]
            inner_eval = outer_train.iloc[inner_eval_idx]         
                       
            train_and_eval = TrainAndEval(
                train_df=inner_train,
                eval_df=inner_eval,
                hparams=hparams,
                device=DEVICE,
            )
            
            metrics_dict, state = train_and_eval()
            
            f1 = metrics_dict.get("macro avg", {}).get("f1-score", 0.0)
            current_acc = metrics_dict.get("accuracy", 0.0)
            inner_f1_scores.append(f1)
            print(f"Inner fold {inner_fold}: F1 = {f1:.4f} | Acc = {current_acc:.4f}")
            
            inner_dir = Path(BASE_DIR) / CHECKPOINT / f"outer_{outer_fold}" / f"inner_{inner_fold}"
            inner_dir.mkdir(parents=True, exist_ok=True)
            
            metrics_path = inner_dir / "metrics.json"
            should_save = True
            
            if metrics_path.exists():
                try:
                    with open(metrics_path, "r", encoding="utf-8") as f:
                        saved_data = json.load(f)
                    old_acc = saved_data.get("evaluation_report", {}).get("accuracy", 0.0)
                    
                    if current_acc > old_acc:
                        print(f"   🔥 Nova melhor acurácia no Outer {outer_fold} Inner {inner_fold}: {old_acc:.4f} -> {current_acc:.4f}")
                        should_save = True
                    else:
                        should_save = False
                except Exception:
                    should_save = True
            
            if should_save:
                inner_json_data = {
                    "trial_number": trial.number,
                    "hparams": hparams,
                    "evaluation_report": metrics_dict
                }
                with open(metrics_path, "w", encoding="utf-8") as f:
                    json.dump(inner_json_data, f, indent=4, ensure_ascii=False)
        
        mean_f1 = np.mean(inner_f1_scores)
        print(f"Trial mean F1: {mean_f1:.4f}")
        return mean_f1
    
    study = optuna.create_study(
        study_name=f"multi_task_outer_fold_{outer_fold}", 
        direction="maximize"
    )
    study.optimize(objective, n_trials=5)
    
    best_params = study.best_params
    print(f"\nBest params for outer fold {outer_fold}: {best_params}")
    print(f"Best CV F1: {study.best_value:.4f}")
    
    final_tae = TrainAndEval(
        train_df=outer_train,
        eval_df=outer_test,
        hparams=rebuild_hparams(best_params),
        device=DEVICE,
    )
    
    test_metrics, final_state = final_tae()
    
    outer_results.append({
        "outer_fold": outer_fold,
        "best_params": best_params,
        "cv_f1": study.best_value,
        "test_metrics": test_metrics,
    })
    
    # Mantivemos o salvamento do modelo de teste do fold externo (Avaliação de Generalização)
    fold_dir = Path(BASE_DIR) / CHECKPOINT / f"outer_fold_{outer_fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    
    metrics_json_dict = {
        'outer_fold': outer_fold,
        'best_params': best_params,
        'cv_best_macro_f1': float(study.best_value),
        'test_evaluation_report': test_metrics
    }
    
    with open(fold_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_json_dict, f, indent=4, ensure_ascii=False)    

best_result = max(
    outer_results,
    key=lambda x: x["cv_f1"]
)

best_global_params = best_result["best_params"]
print("\nFINAL PARAMS:", best_global_params)

final_tae = TrainAndEval(
    train_df=train_data,
    eval_df=test_data,
    hparams=rebuild_hparams(best_global_params),
    device=DEVICE,
)

holdout_metrics, final_state = final_tae()
print(f"\nFINAL HOLDOUT F1: {holdout_metrics.get('macro avg', {}).get('f1-score', 0.0):.4f}")

final_model_dir = Path(BASE_DIR) / CHECKPOINT / "final_model"
final_model_dir.mkdir(parents=True, exist_ok=True)

final_metrics_json_dict = {
    "best_params": best_global_params,
    "holdout_evaluation_report": holdout_metrics
}

with open(final_model_dir / "metrics.json", "w", encoding="utf-8") as f:
    json.dump(final_metrics_json_dict, f, indent=4, ensure_ascii=False)

torch.save(final_state, final_model_dir / "model.pt")