"""
ШАГ 5: Вычисление эталонных метрик (baseline)
"""

import pandas as pd
import numpy as np
import joblib
import json
import os
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from fairlearn.metrics import (
    demographic_parity_difference,
    equalized_odds_difference
)

os.makedirs("data",    exist_ok=True)
os.makedirs("reports", exist_ok=True)

FEATURE_LABELS = {
    "age":              "Возраст",
    "gender":           "Пол",
    "income":           "Доход",
    "credit_score":     "Кред. скор",
    "loan_amount":      "Сумма займа",
    "employment_years": "Стаж работы",
    "debt_to_income":   "Долг/доход",
    "num_prev_loans":   "Пред. займы",
    "num_defaults":     "Просрочки",
}

model       = joblib.load("data/model.pkl")
scaler      = joblib.load("data/scaler.pkl")
X_train     = joblib.load("data/X_train.pkl")
with open("data/feature_cols.json") as f:
    feature_cols = json.load(f)

X_train_scaled = scaler.transform(X_train)
labels = [FEATURE_LABELS.get(f, f) for f in feature_cols]

# 1. ГЛОБАЛЬНЫЕ SHAP
print("Считаем SHAP (это займёт ~30 секунд)...")
explainer   = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_train_scaled)

mean_abs_shap = np.abs(shap_values).mean(axis=0)
global_shap   = dict(zip(feature_cols, mean_abs_shap.tolist()))

sorted_shap = sorted(global_shap.items(), key=lambda x: x[1], reverse=True)
print("\nГлобальная важность признаков (SHAP):")
for feat, val in sorted_shap:
    print(f"  {feat:20s}: {val:.4f}")

fig, ax = plt.subplots(figsize=(8, 5))
names  = [FEATURE_LABELS.get(x[0], x[0]) for x in sorted_shap]
values = [x[1] for x in sorted_shap]
ax.barh(names[::-1], values[::-1], color="#4C72B0")
ax.set_xlabel("Среднее |SHAP-значение|")
ax.set_ylabel("Признак")
ax.set_title("Глобальная важность признаков (эталон)")
plt.tight_layout()
plt.savefig("reports/baseline_shap_global.png", dpi=150)
plt.close()
print("  → reports/baseline_shap_global.png")

# 2. FAIRNESS-МЕТРИКИ
df_train = X_train.copy()
y_train  = joblib.load("data/y_train.pkl")
y_pred   = model.predict(X_train_scaled)
sensitive = df_train["gender"]

dp_diff  = demographic_parity_difference(
    y_true=y_train, y_pred=y_pred, sensitive_features=sensitive
)
eo_diff  = equalized_odds_difference(
    y_true=y_train, y_pred=y_pred, sensitive_features=sensitive
)

print(f"\nFairness (baseline):")
print(f"  Демографический паритет (разница) : {dp_diff:.4f}")
print(f"  Равенство шансов (разница)        : {eo_diff:.4f}")
print("  (0.0 = идеальная справедливость, >0.1 = проблема)")

# 3. SHAP ПО ГРУППАМ
gender_col = df_train["gender"].values
shap_women = shap_values[gender_col == 0]
shap_men   = shap_values[gender_col == 1]

mean_shap_women = dict(zip(feature_cols, np.abs(shap_women).mean(axis=0).tolist()))
mean_shap_men   = dict(zip(feature_cols, np.abs(shap_men  ).mean(axis=0).tolist()))

x     = np.arange(len(feature_cols))
w     = 0.35
fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(x - w/2, [mean_shap_women[f] for f in feature_cols], w,
       label="Женщины", color="#DD8452")
ax.bar(x + w/2, [mean_shap_men[f]   for f in feature_cols], w,
       label="Мужчины",  color="#4C72B0")
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=35, ha="right")
ax.set_ylabel("Среднее |SHAP-значение|")
ax.set_xlabel("Признак")
ax.set_title("Важность признаков по группам (эталон)")
ax.legend()
plt.tight_layout()
plt.savefig("reports/baseline_shap_by_group.png", dpi=150)
plt.close()
print("  → reports/baseline_shap_by_group.png")

# 4. EDA-ГРАФИКИ
df_full = pd.read_csv("data/credit_data.csv")

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

approval_by_gender = df_full.groupby("gender")["approved"].mean()
axes[0].bar(["Женщины", "Мужчины"], approval_by_gender.values,
            color=["#DD8452", "#4C72B0"])
axes[0].set_ylabel("Доля одобрений")
axes[0].set_xlabel("Группа")
axes[0].set_title("Доля одобрений по полу")
axes[0].set_ylim(0, 1)
for i, v in enumerate(approval_by_gender.values):
    axes[0].text(i, v + 0.02, f"{v:.1%}", ha="center", fontsize=11)

women = df_full[df_full["gender"] == 0]["credit_score"]
men   = df_full[df_full["gender"] == 1]["credit_score"]
axes[1].hist(women, bins=30, alpha=0.6, label="Женщины", color="#DD8452")
axes[1].hist(men,   bins=30, alpha=0.6, label="Мужчины",  color="#4C72B0")
axes[1].set_xlabel("Кредитный скор")
axes[1].set_ylabel("Количество клиентов")
axes[1].set_title("Распределение кредитного скора по полу")
axes[1].legend()

plt.tight_layout()
plt.savefig("reports/eda_overview.png", dpi=150)
plt.close()
print("  → reports/eda_overview.png")

# 5. СОХРАНЕНИЕ
baseline = {
    "global_shap":         global_shap,
    "shap_feature_rank":   [x[0] for x in sorted_shap],
    "mean_shap_women":     mean_shap_women,
    "mean_shap_men":       mean_shap_men,
    "demographic_parity_difference": float(dp_diff),
    "equalized_odds_difference":     float(eo_diff),
    "n_train_samples":     int(len(X_train)),
}

with open("data/baseline_metrics.json", "w", encoding="utf-8") as f:
    json.dump(baseline, f, indent=2, ensure_ascii=False)

print("\nBaseline сохранён: data/baseline_metrics.json")
print("=" * 50)
print("ШАГ 5 завершён!")
