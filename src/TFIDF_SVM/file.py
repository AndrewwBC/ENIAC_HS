import json
import pandas as pd
from base import BASE_DIR
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

original  = BASE_DIR / "src/minabr/dataset/minabr.csv"

df = pd.read_csv(original)
df = df[["comment", "odio"]].rename(columns={"comment": "text", "odio": "label"})
df = df.dropna(subset=["text", "label"])

train_df, test_df = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=42)

X_train = train_df["text"].astype(str).to_numpy()
y_train = train_df["label"].astype(int).to_numpy()
X_test  = test_df["text"].astype(str).to_numpy()
y_test  = test_df["label"].astype(int).to_numpy()

pipeline = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("svm",   SVC(kernel="linear", random_state=42)),
])

param_grid = {
    "tfidf__max_features": [3000, 5000, 10000],
    "tfidf__ngram_range":  [(1, 1), (1, 2)],
    "svm__C":              [0.1, 1.0, 10.0],
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

grid_search = GridSearchCV(
    pipeline,
    param_grid,
    cv=cv,
    scoring="f1_macro",
    n_jobs=-1,
    verbose=1,
)

grid_search.fit(X_train, y_train)

print(f"Best params: {grid_search.best_params_}")
print(f"Best CV macro F1: {grid_search.best_score_:.4f}")

y_pred = grid_search.predict(X_test)
report_dict = classification_report(
    y_test, y_pred,
    target_names=["Não-HS", "Hate Speech"],
    output_dict=True
)

results_dir = BASE_DIR / "results"
results_dir.mkdir(parents=True, exist_ok=True)

output = {
    "best_params": grid_search.best_params_,
    "best_cv_macro_f1": grid_search.best_score_,
    "holdout_evaluation_report": report_dict,
}

with open(results_dir / "tfidf_svm_results.json", "w", encoding="utf-8") as f:
    json.dump(output, f, indent=4, ensure_ascii=False)

print(f"Salvo em {results_dir / 'tfidf_svm_results.json'}")