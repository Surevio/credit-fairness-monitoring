"""
ШАГ 8: Эксперимент — два сценария
  Сценарий А (норма):       данные близки к обучающей выборке
  Сценарий Б (деградация):  дрейф данных усиливает дискриминацию
Запускает мониторинг на обоих и сравнивает результаты.
"""

import numpy as np
import pandas as pd
import os, sys

os.makedirs("data",    exist_ok=True)
os.makedirs("reports", exist_ok=True)

# Добавляем папку scripts в путь (для импорта monitor)
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from monitor import run_monitoring, generate_html_report

np.random.seed(99)
N = 500

def _base_features(n, seed=None):
    """Генерирует базовые финансовые признаки."""
    if seed:
        np.random.seed(seed)
    gender           = np.random.binomial(1, 0.5, n)
    age              = np.random.randint(21, 65, n)
    income           = np.random.normal(50000, 15000, n).clip(15000, 120000)
    credit_score     = np.random.normal(650, 80, n).clip(300, 850)
    loan_amount      = np.random.normal(15000, 7000, n).clip(1000, 50000)
    employment_years = np.random.exponential(5, n).clip(0, 30)
    debt_to_income   = np.random.beta(2, 5, n)
    num_prev_loans   = np.random.poisson(1.5, n)
    num_defaults     = np.random.poisson(0.2, n)
    return (gender, age, income, credit_score, loan_amount,
            employment_years, debt_to_income, num_prev_loans, num_defaults)


# ════════════════════════════════════════════════════════════
# СЦЕНАРИЙ А — НОРМА
# Данные похожи на обучающую выборку, небольшой дрейф
# ════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  Генерируем Сценарий А (норма)...")
print("="*60)

(gender, age, income, credit_score, loan_amount,
 employment_years, debt_to_income, num_prev_loans, num_defaults) = _base_features(N, seed=1)

# Такой же скор как при обучении + тот же гендерный штраф
score_a = (
      (credit_score - 300) / 550 * 3.0
    + (income - 15000) / 105000 * 2.0
    + employment_years / 30 * 1.5
    - debt_to_income * 2.5
    - num_defaults * 1.5
    - (loan_amount - 1000) / 49000 * 1.0
    + np.random.normal(0, 0.3, N)
)
gender_penalty_a = np.where(gender == 0, -1.2, 0.0)   # тот же что при обучении
prob_a    = 1 / (1 + np.exp(-(score_a + gender_penalty_a)))
approved_a = np.random.binomial(1, prob_a, N)

df_a = pd.DataFrame({
    "age": age, "gender": gender,
    "income": income.round(0), "credit_score": credit_score.round(0),
    "loan_amount": loan_amount.round(0), "employment_years": employment_years.round(1),
    "debt_to_income": debt_to_income.round(3),
    "num_prev_loans": num_prev_loans, "num_defaults": num_defaults,
    "approved": approved_a
})
df_a.to_csv("data/scenario_a.csv", index=False)

appr_a = df_a.groupby("gender")["approved"].mean()
print(f"Одобрение — женщины: {appr_a.get(0,0):.1%} | мужчины: {appr_a.get(1,0):.1%}")
print("Сохранено: data/scenario_a.csv")


# ════════════════════════════════════════════════════════════
# СЦЕНАРИЙ Б — ДЕГРАДАЦИЯ
# Дрейф данных: женщины с хорошими показателями всё чаще получают отказ,
# мужчины с плохими показателями — одобрение
# ════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  Генерируем Сценарий Б (деградация)...")
print("="*60)

(gender, age, income, credit_score, loan_amount,
 employment_years, debt_to_income, num_prev_loans, num_defaults) = _base_features(N, seed=2)

score_b = (
      (credit_score - 300) / 550 * 3.0
    + (income - 15000) / 105000 * 2.0
    + employment_years / 30 * 1.5
    - debt_to_income * 2.5
    - num_defaults * 1.5
    - (loan_amount - 1000) / 49000 * 1.0
    + np.random.normal(0, 0.3, N)
)

# Усиленный штраф для женщин: -2.5 вместо -1.2
# Дополнительно «бонус» мужчинам с плохими показателями
gender_penalty_b = np.where(gender == 0, -2.5, +0.8)

# Добавляем шум в признаки: имитируем изменение распределения данных
credit_score  = (credit_score  + np.where(gender == 0, -30, +20)).clip(300, 850)
income        = (income        + np.where(gender == 0, -3000, +2000)).clip(15000, 120000)

prob_b     = 1 / (1 + np.exp(-(score_b + gender_penalty_b)))
approved_b = np.random.binomial(1, prob_b, N)

df_b = pd.DataFrame({
    "age": age, "gender": gender,
    "income": income.round(0), "credit_score": credit_score.round(0),
    "loan_amount": loan_amount.round(0), "employment_years": employment_years.round(1),
    "debt_to_income": debt_to_income.round(3),
    "num_prev_loans": num_prev_loans, "num_defaults": num_defaults,
    "approved": approved_b
})
df_b.to_csv("data/scenario_b.csv", index=False)

appr_b = df_b.groupby("gender")["approved"].mean()
print(f"Одобрение — женщины: {appr_b.get(0,0):.1%} | мужчины: {appr_b.get(1,0):.1%}")
print("Сохранено: data/scenario_b.csv")


# ════════════════════════════════════════════════════════════
# ЗАПУСК МОНИТОРИНГА
# ════════════════════════════════════════════════════════════
results_a = run_monitoring("data/scenario_a.csv", "Сценарий А (норма)")
results_b = run_monitoring("data/scenario_b.csv", "Сценарий Б (деградация)")

# ════════════════════════════════════════════════════════════
# СВОДНАЯ ТАБЛИЦА В КОНСОЛИ
# ════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  ИТОГОВОЕ СРАВНЕНИЕ")
print("="*60)

rows = [
    ("Корреляция SHAP-рангов",   f"{results_a['spearman_corr']:.4f}",  f"{results_b['spearman_corr']:.4f}"),
    ("Demographic Parity Diff",  f"{results_a['dp_current']:.4f}",     f"{results_b['dp_current']:.4f}"),
    ("Equalized Odds Diff",      f"{results_a['eo_current']:.4f}",     f"{results_b['eo_current']:.4f}"),
    ("Одобрение женщин",         f"{results_a['approval_women']:.1%}", f"{results_b['approval_women']:.1%}"),
    ("Одобрение мужчин",         f"{results_a['approval_men']:.1%}",   f"{results_b['approval_men']:.1%}"),
    ("⚠ SHAP-дрейф",             "✅ нет" if not results_a['shap_drift_alert']  else "🔴 ДА", "✅ нет" if not results_b['shap_drift_alert']  else "🔴 ДА"),
    ("⚠ Fairness-дрейф",         "✅ нет" if not results_a['fairness_alert']    else "🔴 ДА", "✅ нет" if not results_b['fairness_alert']    else "🔴 ДА"),
    ("⚠ Разрыв SHAP по группам", "✅ нет" if not results_a['group_alert']       else "🔴 ДА", "✅ нет" if not results_b['group_alert']       else "🔴 ДА"),
]

col_w = 30
print(f"{'Метрика':<{col_w}} {'Сценарий А':^20} {'Сценарий Б':^20}")
print("─" * (col_w + 42))
for metric, val_a, val_b in rows:
    print(f"{metric:<{col_w}} {val_a:^20} {val_b:^20}")

# ════════════════════════════════════════════════════════════
# HTML-ОТЧЁТ
# ════════════════════════════════════════════════════════════
generate_html_report(results_a, results_b)

print("\n" + "="*60)
print("  ШАГ 8 ЗАВЕРШЁН!")
print("  Открой reports/monitoring_report.html в браузере")
print("="*60)
