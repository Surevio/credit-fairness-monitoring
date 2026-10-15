"""
ШАГ 4 (продолжение): Обучение модели
- GradientBoostingClassifier
- Сохраняем модель, scaler и список признаков в data/
"""

import pandas as pd
import numpy as np
import joblib
import json
import os
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, roc_auc_score,
    classification_report, confusion_matrix
)

os.makedirs("data", exist_ok=True)

# --- Загрузка данных ---
df = pd.read_csv("data/credit_data.csv")

FEATURE_COLS = [
    "age", "gender", "income", "credit_score",
    "loan_amount", "employment_years",
    "debt_to_income", "num_prev_loans", "num_defaults"
]
TARGET_COL = "approved"

X = df[FEATURE_COLS]
y = df[TARGET_COL]

# --- Разбивка на train/test ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --- Масштабирование ---
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# --- Обучение модели ---
model = GradientBoostingClassifier(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    random_state=42
)
model.fit(X_train_scaled, y_train)

# --- Метрики ---
y_pred      = model.predict(X_test_scaled)
y_pred_prob = model.predict_proba(X_test_scaled)[:, 1]

acc = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred_prob)

print("=" * 50)
print("Модель обучена!")
print(f"Accuracy : {acc:.4f}")
print(f"ROC-AUC  : {auc:.4f}")
print()
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=["Отказ", "Одобрено"]))

# --- Сохранение ---
joblib.dump(model,        "data/model.pkl")
joblib.dump(scaler,       "data/scaler.pkl")
joblib.dump(X_train,      "data/X_train.pkl")   # нужен для baseline SHAP
joblib.dump(y_train,      "data/y_train.pkl")

with open("data/feature_cols.json", "w") as f:
    json.dump(FEATURE_COLS, f)

# Сохраним тестовую выборку для экспериментов
X_test_df = X_test.copy()
X_test_df["approved"] = y_test.values
X_test_df.to_csv("data/test_data.csv", index=False)

print()
print("Сохранено:")
print("  data/model.pkl")
print("  data/scaler.pkl")
print("  data/X_train.pkl")
print("  data/y_train.pkl")
print("  data/feature_cols.json")
print("  data/test_data.csv")
print("=" * 50)
