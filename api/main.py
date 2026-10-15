"""
ШАГ 6: REST API на FastAPI
  POST /predict  — предсказание + топ-3 признака (локальный SHAP)
  GET  /fairness — текущие fairness-метрики по накопленным запросам
  GET  /health   — проверка работоспособности
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import pandas as pd
import joblib
import json
import shap
import uvicorn
from typing import List, Dict, Any
from datetime import datetime
from collections import defaultdict

# ── Загрузка модели и артефактов ─────────────────────────────
model   = joblib.load("data/model.pkl")
scaler  = joblib.load("data/scaler.pkl")
with open("data/feature_cols.json") as f:
    FEATURE_COLS = json.load(f)

explainer = shap.TreeExplainer(model)

# ── Хранилище запросов (в памяти) ────────────────────────────
request_log: List[Dict[str, Any]] = []

# ── FastAPI приложение ────────────────────────────────────────
app = FastAPI(
    title="Credit Fairness Monitor",
    description="ML API с мониторингом интерпретируемости и справедливости",
    version="1.0.0"
)

# ── Схема входных данных ──────────────────────────────────────
class ClientData(BaseModel):
    age:               float
    gender:            int    # 0=женщина, 1=мужчина
    income:            float
    credit_score:      float
    loan_amount:       float
    employment_years:  float
    debt_to_income:    float
    num_prev_loans:    int
    num_defaults:      int

# ── Эндпоинты ─────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "requests_logged": len(request_log)}


@app.post("/predict")
def predict(client: ClientData):
    """
    Принимает данные клиента.
    Возвращает:
      - decision: "approved" / "denied"
      - probability: вероятность одобрения
      - top_features: топ-3 признака, повлиявших на решение (локальный SHAP)
    """
    # Формируем вектор признаков
    row = pd.DataFrame([{
        "age":              client.age,
        "gender":           client.gender,
        "income":           client.income,
        "credit_score":     client.credit_score,
        "loan_amount":      client.loan_amount,
        "employment_years": client.employment_years,
        "debt_to_income":   client.debt_to_income,
        "num_prev_loans":   client.num_prev_loans,
        "num_defaults":     client.num_defaults,
    }])[FEATURE_COLS]

    row_scaled = scaler.transform(row)

    # Предсказание
    prob     = float(model.predict_proba(row_scaled)[0, 1])
    decision = "approved" if prob >= 0.5 else "denied"

    # Локальный SHAP
    shap_vals = explainer.shap_values(row_scaled)[0]           # (n_features,)
    top_idx   = np.argsort(np.abs(shap_vals))[::-1][:3]
    top_features = [
        {
            "feature": FEATURE_COLS[i],
            "shap_value": round(float(shap_vals[i]), 4),
            "direction": "увеличивает одобрение" if shap_vals[i] > 0 else "снижает одобрение"
        }
        for i in top_idx
    ]

    # Логируем запрос
    request_log.append({
        "timestamp":    datetime.utcnow().isoformat(),
        "gender":       client.gender,
        "decision":     decision,
        "probability":  prob,
        "shap_values":  dict(zip(FEATURE_COLS, shap_vals.tolist())),
        "features":     row.iloc[0].to_dict()
    })

    return {
        "decision":     decision,
        "probability":  round(prob, 4),
        "top_features": top_features
    }


@app.get("/fairness")
def fairness():
    """
    Возвращает текущие fairness-метрики
    по всем накопленным запросам.
    """
    if len(request_log) < 10:
        return {
            "warning": f"Мало данных: {len(request_log)} запросов (нужно минимум 10)",
            "requests_total": len(request_log)
        }

    df = pd.DataFrame(request_log)

    # Approval rate по группам
    approval_by_group = (
        df.groupby("gender")["decision"]
        .apply(lambda x: (x == "approved").mean())
        .to_dict()
    )

    # Demographic Parity Difference
    rates  = list(approval_by_group.values())
    dp_diff = max(rates) - min(rates) if len(rates) == 2 else None

    # Средний SHAP по группам
    shap_df    = pd.DataFrame(df["shap_values"].tolist())
    shap_means = {}
    for g, label in [(0, "women"), (1, "men")]:
        subset = shap_df[df["gender"] == g]
        if len(subset) > 0:
            shap_means[label] = subset.mean().to_dict()

    # Загружаем baseline для сравнения
    try:
        with open("data/baseline_metrics.json") as f:
            baseline = json.load(f)
        dp_baseline = baseline["demographic_parity_difference"]
        dp_drift    = round(abs(dp_diff - abs(dp_baseline)), 4) if dp_diff else None
    except FileNotFoundError:
        dp_baseline = None
        dp_drift    = None

    return {
        "requests_total":           len(request_log),
        "approval_rate_by_gender":  {
            "women": round(approval_by_group.get(0, 0), 4),
            "men":   round(approval_by_group.get(1, 0), 4),
        },
        "demographic_parity_difference": round(dp_diff, 4) if dp_diff else None,
        "dp_baseline":              round(abs(dp_baseline), 4) if dp_baseline else None,
        "dp_drift_from_baseline":   dp_drift,
        "alert":                    dp_diff is not None and dp_diff > 0.15,
        "shap_means_by_group":      shap_means,
    }


@app.get("/logs")
def get_logs(limit: int = 20):
    """Последние N запросов из лога"""
    return {"logs": request_log[-limit:], "total": len(request_log)}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
