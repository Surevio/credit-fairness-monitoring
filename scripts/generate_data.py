"""
ШАГ 4: Генерация синтетического датасета
Задача: предсказание одобрения кредита
- 2000 записей
- Защищённый признак: gender (0=женщина, 1=мужчина)
- Намеренный дисбаланс: женщины получают отказ чаще при тех же финансовых показателях
"""

import numpy as np
import pandas as pd
import os

np.random.seed(42)

N = 2000

# --- Генерация признаков ---
gender = np.random.binomial(1, 0.5, N)          # 0=женщина, 1=мужчина
age = np.random.randint(21, 65, N)
income = np.random.normal(50000, 15000, N).clip(15000, 120000)
credit_score = np.random.normal(650, 80, N).clip(300, 850)
loan_amount = np.random.normal(15000, 7000, N).clip(1000, 50000)
employment_years = np.random.exponential(5, N).clip(0, 30)
debt_to_income = np.random.beta(2, 5, N)         # отношение долга к доходу (0..1)
num_prev_loans = np.random.poisson(1.5, N)
num_defaults = np.random.poisson(0.2, N)

# --- Базовый скор одобрения (финансовые показатели) ---
score = (
      (credit_score - 300) / 550 * 3.0
    + (income - 15000) / 105000 * 2.0
    + employment_years / 30 * 1.5
    - debt_to_income * 2.5
    - num_defaults * 1.5
    - (loan_amount - 1000) / 49000 * 1.0
    + np.random.normal(0, 0.3, N)   # небольшой шум
)

# --- Намеренный дисбаланс: штраф для женщин ---
# Женщины (gender=0) получают искусственный штраф -1.2
# Это имитирует историческую предвзятость в обучающих данных
gender_penalty = np.where(gender == 0, -1.2, 0.0)
score_biased = score + gender_penalty

# --- Перевод скора в бинарный таргет ---
prob = 1 / (1 + np.exp(-score_biased))
approved = np.random.binomial(1, prob, N)

# --- Сборка датафрейма ---
df = pd.DataFrame({
    "age":              age,
    "gender":           gender,           # защищённый признак
    "income":           income.round(0),
    "credit_score":     credit_score.round(0),
    "loan_amount":      loan_amount.round(0),
    "employment_years": employment_years.round(1),
    "debt_to_income":   debt_to_income.round(3),
    "num_prev_loans":   num_prev_loans,
    "num_defaults":     num_defaults,
    "approved":         approved          # целевая переменная
})

os.makedirs("data", exist_ok=True)
df.to_csv("data/credit_data.csv", index=False)

# --- Вывод статистики ---
print("=" * 50)
print("Датасет сгенерирован: data/credit_data.csv")
print(f"Всего записей: {len(df)}")
print()
print("Одобрение по полу:")
stats = df.groupby("gender")["approved"].agg(["count", "mean"])
stats.index = ["Женщины (0)", "Мужчины (1)"]
stats.columns = ["Кол-во", "Доля одобрений"]
stats["Доля одобрений"] = stats["Доля одобрений"].map("{:.1%}".format)
print(stats.to_string())
print()
print("Признаки:", list(df.columns[:-1]))
print("=" * 50)
