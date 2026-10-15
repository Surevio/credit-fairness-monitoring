"""
Отправляет 50 тестовых запросов к API:
  - 25 женщин (gender=0) с хорошими показателями
  - 25 мужчин (gender=1) с такими же показателями
Потом выводит ссылку на /fairness
"""

import urllib.request
import urllib.error
import json
import random
import time

random.seed(42)
BASE_URL = "http://127.0.0.1:8000"

def send_predict(data: dict) -> dict:
    body = json.dumps(data).encode("utf-8")
    req  = urllib.request.Request(
        f"{BASE_URL}/predict",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST"
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.loads(resp.read())

print("Отправляем запросы к API...\n")

results = {"women": [], "men": []}

for i in range(50):
    gender = 0 if i < 25 else 1
    label  = "Женщина" if gender == 0 else "Мужчина "

    data = {
        "age":              random.randint(25, 55),
        "gender":           gender,
        "income":           round(random.uniform(40000, 80000)),
        "credit_score":     round(random.uniform(620, 750)),
        "loan_amount":      round(random.uniform(8000, 20000)),
        "employment_years": round(random.uniform(2, 10), 1),
        "debt_to_income":   round(random.uniform(0.15, 0.40), 3),
        "num_prev_loans":   random.randint(0, 3),
        "num_defaults":     0,
    }

    try:
        result = send_predict(data)
        decision = result["decision"]
        prob     = result["probability"]
        if gender == 0:
            results["women"].append(decision)
        else:
            results["men"].append(decision)
        print(f"  [{i+1:02d}] {label} | {decision:8s} | вероятность: {prob:.2%}")
    except Exception as e:
        print(f"  [{i+1:02d}] Ошибка: {e}")

    time.sleep(0.05)

# Итог
print("\n" + "="*50)
print("ИТОГ:")
w_approved = results["women"].count("approved")
m_approved = results["men"].count("approved")
print(f"  Женщины одобрено: {w_approved}/25 ({w_approved/25:.0%})")
print(f"  Мужчины одобрено: {m_approved}/25 ({m_approved/25:.0%})")
print(f"\nТеперь открой в браузере:")
print(f"  {BASE_URL}/fairness")
print("="*50)
