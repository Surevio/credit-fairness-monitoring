"""
ШАГ 7: Модуль мониторинга
Сравнивает baseline с текущими данными:
  1. Дрейф глобальных SHAP-значений (порядок важности признаков)
  2. Дрейф fairness-метрик (demographic parity, equalized odds)
  3. Систематические различия объяснений между группами
Вывод: консоль + HTML-отчёт в reports/
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
import matplotlib.ticker as mticker
from scipy.stats import spearmanr
from fairlearn.metrics import (
    demographic_parity_difference,
    equalized_odds_difference
)
from datetime import datetime

os.makedirs("reports", exist_ok=True)

# Русские названия признаков для графиков
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

model   = joblib.load("data/model.pkl")
scaler  = joblib.load("data/scaler.pkl")
with open("data/feature_cols.json") as f:
    FEATURE_COLS = json.load(f)
with open("data/baseline_metrics.json") as f:
    baseline = json.load(f)

PROTECTED = "gender"

def run_monitoring(data_path: str, scenario_name: str = "current") -> dict:
    print(f"\n{'='*60}")
    print(f"  МОНИТОРИНГ: {scenario_name}")
    print(f"{'='*60}")

    df = pd.read_csv(data_path)
    X  = df[FEATURE_COLS]
    y  = df["approved"]
    X_scaled = scaler.transform(X)

    print("\n[1] Считаем SHAP...")
    explainer   = shap.TreeExplainer(model)
    shap_vals   = explainer.shap_values(X_scaled)

    mean_abs_shap  = np.abs(shap_vals).mean(axis=0)
    current_shap   = dict(zip(FEATURE_COLS, mean_abs_shap.tolist()))
    current_rank   = sorted(current_shap, key=current_shap.get, reverse=True)
    baseline_rank  = baseline["shap_feature_rank"]

    rank_current  = [current_rank.index(f)  for f in FEATURE_COLS]
    rank_baseline = [baseline_rank.index(f) for f in FEATURE_COLS]
    spearman_corr, _ = spearmanr(rank_baseline, rank_current)

    print(f"  Корреляция порядка важности с baseline: {spearman_corr:.4f}")
    print(f"  (1.0 = нет изменений, <0.8 = существенный дрейф)")

    shap_drift_alert = spearman_corr < 0.8

    print("\n[2] Fairness-метрики...")
    y_pred = model.predict(X_scaled)
    sensitive = df[PROTECTED]

    dp_current = demographic_parity_difference(
        y_true=y, y_pred=y_pred, sensitive_features=sensitive
    )
    eo_current = equalized_odds_difference(
        y_true=y, y_pred=y_pred, sensitive_features=sensitive
    )

    dp_baseline_val = abs(baseline["demographic_parity_difference"])
    eo_baseline_val = abs(baseline["equalized_odds_difference"])

    dp_drift = abs(dp_current) - dp_baseline_val
    eo_drift = abs(eo_current) - eo_baseline_val

    print(f"  Demographic Parity Diff  — baseline: {dp_baseline_val:.4f}  |  сейчас: {abs(dp_current):.4f}  |  Δ={dp_drift:+.4f}")
    print(f"  Equalized Odds Diff      — baseline: {eo_baseline_val:.4f}  |  сейчас: {abs(eo_current):.4f}  |  Δ={eo_drift:+.4f}")

    fairness_alert = dp_drift > 0.05 or eo_drift > 0.05

    print("\n[3] Различия объяснений между группами...")
    gender_vals = df[PROTECTED].values
    shap_women  = shap_vals[gender_vals == 0]
    shap_men    = shap_vals[gender_vals == 1]

    mean_women  = np.abs(shap_women).mean(axis=0)
    mean_men    = np.abs(shap_men  ).mean(axis=0)
    shap_gap    = np.abs(mean_women - mean_men)

    max_gap_feat = FEATURE_COLS[np.argmax(shap_gap)]
    max_gap_val  = shap_gap.max()

    print(f"  Наибольшее расхождение SHAP между группами:")
    print(f"  Признак: {max_gap_feat}, Δ={max_gap_val:.4f}")

    group_alert = max_gap_val > 0.1

    approval = df.groupby(PROTECTED)["approved"].mean()
    appr_women = approval.get(0, np.nan)
    appr_men   = approval.get(1, np.nan)
    print(f"\n  Одобрение — женщины: {appr_women:.1%}  |  мужчины: {appr_men:.1%}")

    print(f"\n{'─'*60}")
    print("  АЛЕРТЫ:")
    print(f"  {'🔴' if shap_drift_alert  else '🟢'} SHAP-дрейф:        {'ОБНАРУЖЕН' if shap_drift_alert  else 'норма'}")
    print(f"  {'🔴' if fairness_alert    else '🟢'} Fairness-дрейф:    {'ОБНАРУЖЕН' if fairness_alert    else 'норма'}")
    print(f"  {'🔴' if group_alert       else '🟢'} SHAP между группами: {'РАСХОЖДЕНИЕ' if group_alert   else 'норма'}")
    print(f"{'─'*60}")

    _save_monitoring_plots(
        scenario_name, FEATURE_COLS,
        baseline["global_shap"], current_shap,
        baseline["mean_shap_women"], baseline["mean_shap_men"],
        {f: float(mean_women[i]) for i, f in enumerate(FEATURE_COLS)},
        {f: float(mean_men[i])   for i, f in enumerate(FEATURE_COLS)},
        dp_baseline_val, abs(dp_current),
        eo_baseline_val, abs(eo_current),
    )

    results = {
        "scenario":             scenario_name,
        "n_samples":            len(df),
        "spearman_corr":        float(spearman_corr),
        "shap_drift_alert":     shap_drift_alert,
        "dp_baseline":          dp_baseline_val,
        "dp_current":           float(abs(dp_current)),
        "dp_drift":             float(dp_drift),
        "eo_baseline":          eo_baseline_val,
        "eo_current":           float(abs(eo_current)),
        "eo_drift":             float(eo_drift),
        "fairness_alert":       fairness_alert,
        "max_shap_gap_feature": max_gap_feat,
        "max_shap_gap_value":   float(max_gap_val),
        "group_alert":          group_alert,
        "approval_women":       float(appr_women),
        "approval_men":         float(appr_men),
    }
    return results


def _save_monitoring_plots(
    name, features,
    shap_base, shap_curr,
    shap_base_w, shap_base_m,
    shap_curr_w, shap_curr_m,
    dp_base, dp_curr,
    eo_base, eo_curr,
):
    safe_name = name.replace(" ", "_")
    labels = [FEATURE_LABELS.get(f, f) for f in features]

    # График 1: сравнение глобального SHAP
    fig, ax = plt.subplots(figsize=(9, 5))
    x  = np.arange(len(features))
    w  = 0.35
    ax.bar(x - w/2, [shap_base.get(f, 0) for f in features], w,
           label="Эталон (baseline)", color="#4C72B0", alpha=0.8)
    ax.bar(x + w/2, [shap_curr.get(f, 0) for f in features], w,
           label="Текущий период",    color="#DD8452", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylabel("Среднее |SHAP-значение|")
    ax.set_xlabel("Признак")
    ax.set_title(f"Дрейф важности признаков — {name}")
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"reports/{safe_name}_shap_drift.png", dpi=150)
    plt.close()

    # График 2: fairness-метрики baseline vs текущее
    fig, ax = plt.subplots(figsize=(6, 4))
    metrics = ["Демографический\nпаритет", "Равенство\nшансов"]
    base_v  = [dp_base, eo_base]
    curr_v  = [dp_curr, eo_curr]
    x2 = np.arange(len(metrics))
    ax.bar(x2 - 0.2, base_v, 0.35, label="Эталон (baseline)", color="#4C72B0", alpha=0.8)
    ax.bar(x2 + 0.2, curr_v, 0.35, label="Текущий период",    color="#DD8452", alpha=0.8)
    ax.axhline(0.1, color="red", linestyle="--", linewidth=1.2, label="Порог тревоги (0.1)")
    ax.set_xticks(x2)
    ax.set_xticklabels(metrics)
    ax.set_ylabel("Значение метрики")
    ax.set_xlabel("Метрика справедливости")
    ax.set_title(f"Метрики справедливости — {name}")
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"reports/{safe_name}_fairness.png", dpi=150)
    plt.close()

    # График 3: SHAP по группам
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    for ax, (w_data, m_data, title) in zip(axes, [
        (shap_base_w, shap_base_m, "Эталон (baseline)"),
        (shap_curr_w, shap_curr_m, "Текущий период"),
    ]):
        xp = np.arange(len(features))
        ax.bar(xp - 0.2, [w_data.get(f, 0) for f in features], 0.35,
               label="Женщины", color="#DD8452", alpha=0.8)
        ax.bar(xp + 0.2, [m_data.get(f, 0) for f in features], 0.35,
               label="Мужчины",  color="#4C72B0", alpha=0.8)
        ax.set_xticks(xp)
        ax.set_xticklabels(labels, rotation=35, ha="right")
        ax.set_ylabel("Среднее |SHAP-значение|")
        ax.set_xlabel("Признак")
        ax.set_title(f"Важность признаков по группам\n({title})")
        ax.legend()
    plt.suptitle(name, fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"reports/{safe_name}_shap_groups.png", dpi=150)
    plt.close()

    print(f"\n  Графики сохранены в reports/{safe_name}_*.png")


def generate_html_report(results_a: dict, results_b: dict):
    def status(alert: bool) -> str:
        return '<span style="color:red;font-weight:bold">🔴 ПРОБЛЕМА</span>' if alert \
          else '<span style="color:green;font-weight:bold">🟢 НОРМА</span>'

    html = f"""<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="UTF-8">
<title>Отчёт мониторинга — Credit Fairness</title>
<style>
  body {{ font-family: Arial, sans-serif; max-width: 1100px; margin: 40px auto; color: #222; }}
  h1   {{ color: #2c3e50; }}
  h2   {{ color: #34495e; border-bottom: 2px solid #3498db; padding-bottom: 6px; }}
  table {{ border-collapse: collapse; width: 100%; margin: 16px 0; }}
  th, td {{ border: 1px solid #ccc; padding: 8px 12px; text-align: left; }}
  th   {{ background: #3498db; color: white; }}
  tr:nth-child(even) {{ background: #f4f8fb; }}
  img  {{ max-width: 100%; margin: 10px 0; border: 1px solid #ddd; border-radius: 6px; }}
</style>
</head>
<body>
<h1>Отчёт мониторинга интерпретируемости и справедливости</h1>
<p>Дата: {datetime.now().strftime('%d.%m.%Y %H:%M')}</p>

<h2>Сравнительная таблица сценариев</h2>
<table>
  <tr>
    <th>Метрика</th>
    <th>Эталон (baseline)</th>
    <th>Сценарий А (норма)</th>
    <th>Сценарий Б (деградация)</th>
  </tr>
  <tr><td>Корреляция SHAP-рангов</td><td>1.0000</td><td>{results_a['spearman_corr']:.4f}</td><td>{results_b['spearman_corr']:.4f}</td></tr>
  <tr><td>Демографический паритет (разница)</td><td>{results_a['dp_baseline']:.4f}</td><td>{results_a['dp_current']:.4f} (Δ={results_a['dp_drift']:+.4f})</td><td>{results_b['dp_current']:.4f} (Δ={results_b['dp_drift']:+.4f})</td></tr>
  <tr><td>Равенство шансов (разница)</td><td>{results_a['eo_baseline']:.4f}</td><td>{results_a['eo_current']:.4f} (Δ={results_a['eo_drift']:+.4f})</td><td>{results_b['eo_current']:.4f} (Δ={results_b['eo_drift']:+.4f})</td></tr>
  <tr><td>Доля одобрений — женщины</td><td>—</td><td>{results_a['approval_women']:.1%}</td><td>{results_b['approval_women']:.1%}</td></tr>
  <tr><td>Доля одобрений — мужчины</td><td>—</td><td>{results_a['approval_men']:.1%}</td><td>{results_b['approval_men']:.1%}</td></tr>
  <tr><td>Алерт: дрейф SHAP-важности</td><td>—</td><td>{status(results_a['shap_drift_alert'])}</td><td>{status(results_b['shap_drift_alert'])}</td></tr>
  <tr><td>Алерт: дрейф справедливости</td><td>—</td><td>{status(results_a['fairness_alert'])}</td><td>{status(results_b['fairness_alert'])}</td></tr>
  <tr><td>Алерт: разрыв SHAP между группами</td><td>—</td><td>{status(results_a['group_alert'])}</td><td>{status(results_b['group_alert'])}</td></tr>
</table>

<h2>Сценарий А — Норма</h2>
<img src="Сценарий_А_(норма)_shap_drift.png"  alt="SHAP drift A">
<img src="Сценарий_А_(норма)_fairness.png"     alt="Fairness A">
<img src="Сценарий_А_(норма)_shap_groups.png"  alt="SHAP groups A">

<h2>Сценарий Б — Деградация</h2>
<img src="Сценарий_Б_(деградация)_shap_drift.png"  alt="SHAP drift B">
<img src="Сценарий_Б_(деградация)_fairness.png"     alt="Fairness B">
<img src="Сценарий_Б_(деградация)_shap_groups.png"  alt="SHAP groups B">

</body>
</html>"""

    path = "reports/monitoring_report.html"
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"\n  HTML-отчёт: {path}")
    return path
